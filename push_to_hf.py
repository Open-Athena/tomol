"""
Stream a HuggingFace molecular dataset, serialize to text, and push to HuggingFace Hub.

This script:
1. Streams molecule data from a HuggingFace dataset (e.g., colabfit/OMol25_train)
2. Tokenizes using RVQ codebooks
3. Converts to text representation
4. Pushes to HuggingFace Hub in shards to keep memory bounded
"""

import argparse
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from pathlib import Path
from queue import Queue
from threading import Thread

import datasets
import numpy as np
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, get_token
from tqdm import tqdm

from serialize_molecules import MoleculeTokenizer

DEFAULT_SHARD_SIZE = 100_000
DEFAULT_NUM_WORKERS = 16
BATCH_SIZE = 256  # Number of examples to batch before parallel processing
PREFETCH_BATCHES = 4  # Number of batches to prefetch from stream

# Global worker state for multiprocessing
_worker_tokenizer: MoleculeTokenizer | None = None
_worker_shuffle_sections: bool = False


def _init_worker(codebook_path: str, shuffle_sections: bool):
    """Initialize worker process with tokenizer."""
    global _worker_tokenizer, _worker_shuffle_sections
    _worker_tokenizer = MoleculeTokenizer(codebook_path)
    _worker_shuffle_sections = shuffle_sections


def _serialize_in_worker(args: tuple[dict, int | None]) -> str:
    """Serialize a single example in worker process."""
    example, seed = args
    rng = np.random.default_rng(seed) if _worker_shuffle_sections and seed is not None else None

    atomic_numbers = example["atomic_numbers"]
    positions = np.array(example["positions"])
    forces = np.array(example["atomic_forces"])
    energy = float(example["energy"])

    tokens = _worker_tokenizer.encode_molecule(
        atomic_numbers,
        positions,
        forces,
        energy,
        shuffle_sections=_worker_shuffle_sections,
        rng=rng,
    )
    return _worker_tokenizer.tokens_to_string(tokens)


def prefetch_batches(
    ds,
    batch_size: int,
    queue: Queue,
    seed_rng: np.random.Generator | None,
    max_rows: int | None,
):
    """Prefetch batches from streaming dataset into a queue."""
    batch = []
    batch_seeds = []
    count = 0

    for example in ds:
        batch.append(example)
        batch_seeds.append(
            int(seed_rng.integers(0, 2**31)) if seed_rng is not None else None
        )
        count += 1

        if len(batch) >= batch_size:
            queue.put((batch, batch_seeds, count))
            batch = []
            batch_seeds = []

        if max_rows is not None and count >= max_rows:
            break

    # Send final partial batch
    if batch:
        queue.put((batch, batch_seeds, count))

    # Signal end of stream
    queue.put(None)


def serialize_and_push(
    source_dataset: str,
    codebook_path: str,
    repo_id: str,
    source_revision: str = "main",
    source_split: str = "train",
    target_split: str = "train",
    max_rows: int | None = None,
    skip_rows: int = 0,
    shuffle_sections: bool = False,
    shuffle_seed: int = 42,
    private: bool = False,
    shard_size: int = DEFAULT_SHARD_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
):
    """
    Stream a HuggingFace dataset, serialize molecules, and push to HuggingFace Hub.

    Pushes in shards to keep memory usage bounded.

    Args:
        source_dataset: Source HuggingFace dataset ID (e.g., "colabfit/OMol25_train")
        codebook_path: Path to codebook pickle file
        repo_id: Target HuggingFace repo ID (e.g., "username/dataset-name")
        source_revision: Source dataset revision/branch
        source_split: Source dataset split to use
        target_split: Target split name in output dataset (e.g., "train", "validation")
        max_rows: Maximum rows to process (None for all)
        skip_rows: Number of rows to skip from the beginning
        shuffle_sections: Whether to shuffle molecule sections
        shuffle_seed: Random seed for section shuffling
        private: Whether to create a private repository
        shard_size: Number of examples per shard (default: 100k)
        num_workers: Number of parallel workers for serialization (default: 16)
    """
    # Set up HF token
    token = get_token()
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)

    tokenizer = MoleculeTokenizer(codebook_path)

    print("Vocabulary Info:")
    vocab_info = tokenizer.get_vocab_info()
    for key, value in vocab_info.items():
        print(f"  {key}: {value}")

    print(f"\nLoading dataset from {source_dataset} (split={source_split})...")

    # Load dataset - check if local path or HuggingFace dataset
    datasets.disable_caching()
    source_path = Path(source_dataset)
    if source_path.exists():
        # Local file or directory
        if source_path.is_file():
            ds = load_dataset("parquet", data_files=str(source_path), split="train")
        else:
            # Directory of parquet files
            ds = load_dataset("parquet", data_dir=str(source_path), split="train")
        print(f"Loaded local dataset with {len(ds):,} rows")
        ds = ds.to_iterable_dataset()
    else:
        # HuggingFace dataset - use streaming
        ds = load_dataset(
            source_dataset,
            revision=source_revision,
            split=source_split,
            streaming=True,
        )

    # Select only the columns we need
    ds = ds.select_columns(["atomic_numbers", "positions", "atomic_forces", "energy"])

    # Skip rows if specified
    if skip_rows > 0:
        ds = ds.skip(skip_rows)
        print(f"Skipping first {skip_rows:,} molecules")

    # Limit rows if specified
    if max_rows is not None:
        ds = ds.take(max_rows)
        print(f"Processing up to {max_rows:,} molecules")

    # Set up seed generator for thread-safe RNG (each example gets unique seed)
    seed_rng = np.random.default_rng(shuffle_seed) if shuffle_sections else None

    # Create or get the repository
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    print(f"\nStreaming and pushing to HuggingFace Hub: {repo_id} (split: {target_split})")
    print(f"Shard size: {shard_size:,} molecules (~{shard_size * 3 // 1000} MB per shard)")
    print(f"Using {num_workers} workers for parallel serialization")

    def upload_shard(shard_path: Path, shard_idx: int, count: int) -> tuple[int, int]:
        """Upload a shard and return (shard_idx, count) on success."""
        api.upload_file(
            path_or_fileobj=str(shard_path),
            path_in_repo=f"data/{target_split}-{shard_idx:05d}.parquet",
            repo_id=repo_id,
            repo_type="dataset",
        )
        shard_path.unlink()
        return shard_idx, count

    # Start prefetching batches from stream in background thread
    batch_queue: Queue = Queue(maxsize=PREFETCH_BATCHES)
    prefetch_thread = Thread(
        target=prefetch_batches,
        args=(ds, BATCH_SIZE, batch_queue, seed_rng, max_rows),
        daemon=True,
    )
    prefetch_thread.start()

    # Process in shards
    texts = []
    shard_idx = 0
    total_processed = 0
    total_serialized = 0
    first_text = None
    pending_uploads: list[Future] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        with (
            ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_init_worker,
                initargs=(codebook_path, shuffle_sections),
            ) as proc_executor,
            ThreadPoolExecutor(max_workers=2) as upload_executor,
        ):
            with tqdm(total=max_rows, desc="Processing molecules") as pbar:
                while True:
                    # Get next batch from prefetch queue
                    batch_data = batch_queue.get()
                    if batch_data is None:
                        break  # End of stream

                    batch, batch_seeds, cumulative_count = batch_data

                    # Process batch in parallel using process pool
                    batch_texts = list(
                        proc_executor.map(_serialize_in_worker, zip(batch, batch_seeds))
                    )
                    texts.extend(batch_texts)
                    total_serialized += len(batch_texts)

                    # Update progress bar
                    pbar.update(len(batch))

                    if first_text is None and batch_texts:
                        first_text = batch_texts[0]

                    # When shard is full, write and upload in background
                    while len(texts) >= shard_size:
                        shard_texts = texts[:shard_size]
                        texts = texts[shard_size:]

                        shard_path = tmpdir / f"{target_split}-{shard_idx:05d}.parquet"
                        shard_dataset = Dataset.from_dict({"text": shard_texts})
                        shard_dataset.to_parquet(shard_path)

                        # Submit upload to background thread
                        future = upload_executor.submit(
                            upload_shard, shard_path, shard_idx, len(shard_texts)
                        )
                        pending_uploads.append(future)

                        # Check for completed uploads and report progress
                        done = [f for f in pending_uploads if f.done()]
                        for f in done:
                            idx, count = f.result()
                            total_processed += count
                            tqdm.write(
                                f"  Uploaded {target_split} shard {idx} ({total_processed:,} molecules total)"
                            )
                            pending_uploads.remove(f)

                        shard_idx += 1

            # Upload remaining full shards
            while len(texts) >= shard_size:
                shard_texts = texts[:shard_size]
                texts = texts[shard_size:]

                shard_path = tmpdir / f"{target_split}-{shard_idx:05d}.parquet"
                shard_dataset = Dataset.from_dict({"text": shard_texts})
                shard_dataset.to_parquet(shard_path)

                future = upload_executor.submit(
                    upload_shard, shard_path, shard_idx, len(shard_texts)
                )
                pending_uploads.append(future)
                shard_idx += 1

            # Upload final partial shard if any
            if texts:
                shard_path = tmpdir / f"{target_split}-{shard_idx:05d}.parquet"
                shard_dataset = Dataset.from_dict({"text": texts})
                shard_dataset.to_parquet(shard_path)

                future = upload_executor.submit(
                    upload_shard, shard_path, shard_idx, len(texts)
                )
                pending_uploads.append(future)

            # Wait for all pending uploads to complete
            for future in pending_uploads:
                idx, count = future.result()
                total_processed += count
                tqdm.write(
                    f"  Uploaded {target_split} shard {idx} ({total_processed:,} molecules total)"
                )

    prefetch_thread.join()

    print(f"\nSuccessfully pushed {total_processed:,} molecules to '{target_split}' split in {shard_idx + 1} shards")
    print(f"Dataset URL: https://huggingface.co/datasets/{repo_id}")
    if first_text:
        print(f"Sample text (first 200 chars): {first_text[:200]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Stream HuggingFace molecular dataset, serialize, and push to Hub"
    )
    parser.add_argument(
        "source_dataset", type=str,
        help="Source HuggingFace dataset ID (e.g., colabfit/OMol25_train)"
    )
    parser.add_argument(
        "repo_id", type=str,
        help="Target HuggingFace repo ID (e.g., username/dataset-name)"
    )
    parser.add_argument(
        "--codebook", "-c", type=str, default="codebook_mol_1m.pkl",
        help="Path to codebook pickle file"
    )
    parser.add_argument(
        "--source-revision", type=str, default="main",
        help="Source dataset revision/branch"
    )
    parser.add_argument(
        "--source-split", type=str, default="train",
        help="Source dataset split"
    )
    parser.add_argument(
        "--target-split", type=str, default="train",
        help="Target split name in output dataset (e.g., train, validation)"
    )
    parser.add_argument(
        "--max-rows", type=int, default=None,
        help="Maximum rows to process"
    )
    parser.add_argument(
        "--skip-rows", type=int, default=0,
        help="Number of rows to skip from the beginning"
    )
    parser.add_argument(
        "--shuffle-sections", action="store_true",
        help="Shuffle molecule sections (atoms, pos, force, energy)"
    )
    parser.add_argument(
        "--shuffle-seed", type=int, default=42,
        help="Random seed for section shuffling"
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create a private repository"
    )
    parser.add_argument(
        "--shard-size", type=int, default=DEFAULT_SHARD_SIZE,
        help=f"Number of molecules per shard (default: {DEFAULT_SHARD_SIZE:,})"
    )
    parser.add_argument(
        "--num-workers", "-j", type=int, default=DEFAULT_NUM_WORKERS,
        help=f"Number of parallel workers for serialization (default: {DEFAULT_NUM_WORKERS})"
    )
    args = parser.parse_args()

    serialize_and_push(
        source_dataset=args.source_dataset,
        codebook_path=args.codebook,
        repo_id=args.repo_id,
        source_revision=args.source_revision,
        source_split=args.source_split,
        target_split=args.target_split,
        max_rows=args.max_rows,
        skip_rows=args.skip_rows,
        shuffle_sections=args.shuffle_sections,
        shuffle_seed=args.shuffle_seed,
        private=args.private,
        shard_size=args.shard_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

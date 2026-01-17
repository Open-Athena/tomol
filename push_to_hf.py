"""
Serialize molecular dataset and push to HuggingFace Hub.

Streams molecule data, tokenizes using RVQ codebooks, and uploads train/val splits.
"""

import argparse
import tempfile
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from pathlib import Path
from queue import Queue
from threading import Thread

import datasets
import numpy as np
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

from serialize_molecules import MoleculeTokenizer

DEFAULT_SHARD_SIZE = 100_000
DEFAULT_NUM_WORKERS = 16
BATCH_SIZE = 512
PREFETCH_BATCHES = 8

# Global worker state for multiprocessing
_worker_tokenizer: MoleculeTokenizer | None = None


def _init_worker(codebook_path: str):
    """Initialize worker process with tokenizer."""
    global _worker_tokenizer
    _worker_tokenizer = MoleculeTokenizer(codebook_path)


def _serialize_in_worker(example: dict) -> str:
    """Serialize a single example in worker process."""
    tokens = _worker_tokenizer.encode_molecule(
        example["atomic_numbers"],
        np.array(example["positions"]),
        np.array(example["atomic_forces"]),
        float(example["energy"]),
    )
    return _worker_tokenizer.tokens_to_string(tokens)


def prefetch_batches(ds, batch_size: int, queue: Queue):
    """Prefetch batches from dataset into a queue."""
    batch = []
    for example in ds:
        batch.append(example)
        if len(batch) >= batch_size:
            queue.put(batch)
            batch = []
    if batch:
        queue.put(batch)
    queue.put(None)  # Signal end


def serialize_and_push(
    source_dataset: str,
    codebook_path: str,
    repo_id: str,
    val_size: int = 100_000,
    shard_size: int = DEFAULT_SHARD_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    seed: int = 42,
):
    """
    Serialize molecules and push to HuggingFace Hub with train/val splits.

    First val_size molecules go to validation, rest to train.
    """
    tokenizer = MoleculeTokenizer(codebook_path)
    print("Vocabulary Info:")
    for key, value in tokenizer.get_vocab_info().items():
        print(f"  {key}: {value}")

    print(f"\nLoading dataset from {source_dataset}...")
    datasets.disable_caching()
    source_path = Path(source_dataset)
    if source_path.exists():
        if source_path.is_file():
            ds = load_dataset("parquet", data_files=str(source_path), split="train", num_proc=num_workers)
        else:
            ds = load_dataset("parquet", data_dir=str(source_path), split="train", num_proc=num_workers)
        total_rows = len(ds)
        print(f"Loaded {total_rows:,} rows, shuffling with seed={seed}...")
        ds = ds.shuffle(seed=seed)
        ds = ds.to_iterable_dataset()
    else:
        ds = load_dataset(source_dataset, split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        total_rows = None
        print(f"Streaming from HuggingFace (buffer shuffle, seed={seed})")

    ds = ds.select_columns(["atomic_numbers", "positions", "atomic_forces", "energy"])

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    print(f"\nPushing to: {repo_id}")
    print(f"Validation: first {val_size:,} molecules")
    print(f"Train: remaining molecules")
    print(f"Using {num_workers} workers")

    def upload_shard(shard_path: Path, split: str, shard_idx: int, count: int):
        api.upload_file(
            path_or_fileobj=str(shard_path),
            path_in_repo=f"data/{split}-{shard_idx:05d}.parquet",
            repo_id=repo_id,
            repo_type="dataset",
        )
        shard_path.unlink()
        return split, shard_idx, count

    batch_queue: Queue = Queue(maxsize=PREFETCH_BATCHES)
    prefetch_thread = Thread(target=prefetch_batches, args=(ds, BATCH_SIZE, batch_queue), daemon=True)
    prefetch_thread.start()

    texts = []
    processed = 0
    val_shard_idx = 0
    train_shard_idx = 0
    pending_uploads: list[Future] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        with (
            ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker, initargs=(codebook_path,)) as proc_executor,
            ThreadPoolExecutor(max_workers=2) as upload_executor,
        ):
            pbar = tqdm(total=total_rows, desc="Processing")

            while True:
                batch = batch_queue.get()
                if batch is None:
                    break

                batch_texts = list(proc_executor.map(_serialize_in_worker, batch))
                pbar.update(len(batch))

                for text in batch_texts:
                    texts.append(text)
                    processed += 1

                    # Determine current split
                    in_val = processed <= val_size
                    split = "validation" if in_val else "train"
                    shard_idx = val_shard_idx if in_val else train_shard_idx

                    # Flush shard when full OR when transitioning from val to train
                    flush = len(texts) >= shard_size or (processed == val_size + 1 and len(texts) > 1)

                    if flush:
                        # When transitioning, the last text belongs to train, so exclude it
                        if processed == val_size + 1:
                            shard_texts = texts[:-1]
                            texts = texts[-1:]
                            split = "validation"
                            shard_idx = val_shard_idx
                        else:
                            shard_texts = texts
                            texts = []

                        shard_path = tmpdir / f"{split}-{shard_idx:05d}.parquet"
                        Dataset.from_dict({"text": shard_texts}).to_parquet(shard_path)

                        future = upload_executor.submit(upload_shard, shard_path, split, shard_idx, len(shard_texts))
                        pending_uploads.append(future)

                        if split == "validation":
                            val_shard_idx += 1
                        else:
                            train_shard_idx += 1

                        # Report completed uploads
                        for f in [f for f in pending_uploads if f.done()]:
                            s, idx, cnt = f.result()
                            tqdm.write(f"  Uploaded {s} shard {idx}")
                            pending_uploads.remove(f)

            pbar.close()

            # Final shard
            if texts:
                split = "validation" if processed <= val_size else "train"
                shard_idx = val_shard_idx if split == "validation" else train_shard_idx
                shard_path = tmpdir / f"{split}-{shard_idx:05d}.parquet"
                Dataset.from_dict({"text": texts}).to_parquet(shard_path)
                future = upload_executor.submit(upload_shard, shard_path, split, shard_idx, len(texts))
                pending_uploads.append(future)

            # Wait for uploads
            for f in pending_uploads:
                s, idx, cnt = f.result()
                tqdm.write(f"  Uploaded {s} shard {idx}")

    prefetch_thread.join()
    print(f"\nDone! Validation: {min(processed, val_size):,}, Train: {max(0, processed - val_size):,}")
    print(f"https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Serialize molecules and push to HuggingFace Hub")
    parser.add_argument("source_dataset", help="Local path or HuggingFace dataset ID")
    parser.add_argument("repo_id", help="Target HuggingFace repo ID")
    parser.add_argument("--codebook", "-c", default="codebook_mol_1m.pkl", help="Codebook path")
    parser.add_argument("--val-size", "-v", type=int, default=100_000, help="Validation set size (default: 100k)")
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE, help="Molecules per shard")
    parser.add_argument("--num-workers", "-j", type=int, default=DEFAULT_NUM_WORKERS, help="Parallel workers")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Shuffle seed (default: 42)")
    args = parser.parse_args()

    serialize_and_push(
        source_dataset=args.source_dataset,
        codebook_path=args.codebook,
        repo_id=args.repo_id,
        val_size=args.val_size,
        shard_size=args.shard_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

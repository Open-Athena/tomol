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
from pathlib import Path

import datasets
import numpy as np
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, get_token
from tqdm import tqdm

from serialize_molecules import MoleculeTokenizer

DEFAULT_SHARD_SIZE = 100_000


def serialize_example(
    example: dict,
    tokenizer: MoleculeTokenizer,
    shuffle_sections: bool,
    rng: np.random.Generator | None,
) -> str:
    """Serialize a single molecule example to text."""
    atomic_numbers = example["atomic_numbers"]
    positions = np.array(example["positions"])
    forces = np.array(example["atomic_forces"])
    energy = float(example["energy"])

    tokens = tokenizer.encode_molecule(
        atomic_numbers,
        positions,
        forces,
        energy,
        shuffle_sections=shuffle_sections,
        rng=rng,
    )
    return tokenizer.tokens_to_string(tokens)


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

    # Load dataset in streaming mode
    datasets.disable_caching()
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

    # Set up RNG for section shuffling
    rng = np.random.default_rng(shuffle_seed) if shuffle_sections else None

    # Create or get the repository
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)

    print(f"\nStreaming and pushing to HuggingFace Hub: {repo_id} (split: {target_split})")
    print(f"Shard size: {shard_size:,} molecules (~{shard_size * 3 // 1000} MB per shard)")

    # Process in shards
    texts = []
    shard_idx = 0
    total_processed = 0
    first_text = None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for example in tqdm(ds, total=max_rows, desc="Processing molecules"):
            text = serialize_example(example, tokenizer, shuffle_sections, rng)
            texts.append(text)

            if first_text is None:
                first_text = text

            # When shard is full, write and upload
            if len(texts) >= shard_size:
                shard_path = tmpdir / f"{target_split}-{shard_idx:05d}.parquet"
                shard_dataset = Dataset.from_dict({"text": texts})
                shard_dataset.to_parquet(shard_path)

                # Upload shard
                api.upload_file(
                    path_or_fileobj=str(shard_path),
                    path_in_repo=f"data/{target_split}-{shard_idx:05d}.parquet",
                    repo_id=repo_id,
                    repo_type="dataset",
                )

                total_processed += len(texts)
                tqdm.write(f"  Uploaded {target_split} shard {shard_idx} ({total_processed:,} molecules total)")

                # Clear memory
                shard_path.unlink()
                texts = []
                shard_idx += 1

        # Upload final partial shard if any
        if texts:
            shard_path = tmpdir / f"{target_split}-{shard_idx:05d}.parquet"
            shard_dataset = Dataset.from_dict({"text": texts})
            shard_dataset.to_parquet(shard_path)

            api.upload_file(
                path_or_fileobj=str(shard_path),
                path_in_repo=f"data/{target_split}-{shard_idx:05d}.parquet",
                repo_id=repo_id,
                repo_type="dataset",
            )

            total_processed += len(texts)
            tqdm.write(f"  Uploaded {target_split} shard {shard_idx} ({total_processed:,} molecules total)")

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
    )


if __name__ == "__main__":
    main()

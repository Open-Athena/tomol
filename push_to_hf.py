"""
Serialize molecular dataset and push to HuggingFace Hub.

Dataset downloaded locally with: uv run huggingface-cli download facebook/OMol25
"""

import argparse
import multiprocessing as mp
import random
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from tqdm import tqdm

from serialize_molecules import MoleculeTokenizer

_tokenizer: MoleculeTokenizer | None = None
_counter = None


def _init_worker(codebook_path: str, counter: mp.Value):
    global _tokenizer, _counter
    _tokenizer = MoleculeTokenizer(codebook_path)
    _counter = counter


def _process_file(args: tuple) -> tuple[list[str], list[str]]:
    """Process one parquet file in batches, return (train_texts, val_texts)."""
    path, start_idx, val_indices = args
    pf = pq.ParquetFile(path)

    train, val = [], []
    row_idx = start_idx

    for batch in pf.iter_batches(batch_size=1024, columns=["atomic_numbers", "positions", "atomic_forces", "energy"]):
        batch_list = batch.to_pylist()
        for row in batch_list:
            tokens = _tokenizer.encode_molecule(
                row["atomic_numbers"],
                np.array(row["positions"]),
                np.array(row["atomic_forces"]),
                float(row["energy"]),
            )
            text = _tokenizer.tokens_to_string(tokens)
            if row_idx in val_indices:
                val.append(text)
            else:
                train.append(text)
            row_idx += 1
        with _counter.get_lock():
            _counter.value += len(batch_list)
    return train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("--codebook", "-c", default="codebook_mol_1m.pkl")
    parser.add_argument("--val-size", "-v", type=int, default=100_000)
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--num-workers", "-j", type=int, default=16)
    parser.add_argument("--seed", "-s", type=int, default=42)
    args = parser.parse_args()

    # Print vocab info
    tokenizer = MoleculeTokenizer(args.codebook)
    print("Vocabulary:", tokenizer.get_vocab_info())

    # Scan files and compute offsets
    files = sorted(args.source_dir.glob("*.parquet"))
    offsets, total = [], 0
    for f in files:
        offsets.append(total)
        total += pq.read_metadata(f).num_rows
    print(f"Found {len(files)} files, {total:,} rows")

    # Pre-select validation indices
    random.seed(args.seed)
    val_indices = frozenset(random.sample(range(total), min(args.val_size, total)))
    print(f"Validation: {len(val_indices):,}, Train: {total - len(val_indices):,}")

    # Process files in parallel with live progress
    tasks = [(f, off, val_indices) for f, off in zip(files, offsets)]
    train_all, val_all = [], []
    counter = mp.Value("q", 0)  # Shared counter for progress

    with ProcessPoolExecutor(args.num_workers, initializer=_init_worker, initargs=(args.codebook, counter)) as pool:
        futures = [pool.submit(_process_file, t) for t in tasks]
        pbar = tqdm(total=total, desc="Processing", unit="mol")

        # Poll counter until all futures complete
        done = 0
        while done < len(futures):
            time.sleep(0.1)
            pbar.n = counter.value
            pbar.refresh()
            done = sum(f.done() for f in futures)

        # Collect results
        for future in futures:
            train, val = future.result()
            train_all.extend(train)
            val_all.extend(val)

        pbar.n = counter.value
        pbar.close()

    # Upload shards
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)

    def write_and_upload(texts: list[str], split: str, shard_idx: int, tmpdir: Path):
        path = tmpdir / f"{split}-{shard_idx:05d}.parquet"
        pq.write_table(pa.table({"text": texts}), path)
        api.upload_file(str(path), f"data/{split}-{shard_idx:05d}.parquet", args.repo_id, repo_type="dataset")
        path.unlink()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        uploads = []

        with ThreadPoolExecutor(8) as pool:
            # Train shards
            for i in range(0, len(train_all), args.shard_size):
                shard = train_all[i:i + args.shard_size]
                uploads.append(pool.submit(write_and_upload, shard, "train", i // args.shard_size, tmpdir))

            # Validation shards
            for i in range(0, len(val_all), args.shard_size):
                shard = val_all[i:i + args.shard_size]
                uploads.append(pool.submit(write_and_upload, shard, "validation", i // args.shard_size, tmpdir))

            for f in tqdm(uploads, desc="Uploading"):
                f.result()

    print(f"\nDone: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()

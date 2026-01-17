"""
Serialize molecular dataset and push to HuggingFace Hub.

Dataset downloaded locally with: uv run huggingface-cli download facebook/OMol25
"""

import argparse
import multiprocessing as mp
import random
import shutil
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


class ShardUploader:
    """Streams shards to HuggingFace when they reach target size."""

    def __init__(self, api: HfApi, repo_id: str, target_bytes: int):
        self.api = api
        self.repo_id = repo_id
        self.target_bytes = target_bytes
        self.tmpdir = Path(tempfile.mkdtemp())

        self.train_buffer: list[str] = []
        self.train_bytes = 0
        self.train_shard_idx = 0

        self.val_buffer: list[str] = []
        self.val_bytes = 0
        self.val_shard_idx = 0

        self.upload_pool = ThreadPoolExecutor(4)
        self.pending_uploads: list = []

    def add_train(self, texts: list[str]):
        self.train_buffer.extend(texts)
        self.train_bytes += sum(len(t) for t in texts)
        if self.train_bytes >= self.target_bytes:
            self._flush_train()

    def add_val(self, texts: list[str]):
        self.val_buffer.extend(texts)
        self.val_bytes += sum(len(t) for t in texts)
        if self.val_bytes >= self.target_bytes:
            self._flush_val()

    def _flush_train(self):
        if not self.train_buffer:
            return
        shard, idx = self.train_buffer, self.train_shard_idx
        self.train_buffer = []
        self.train_bytes = 0
        self.train_shard_idx += 1
        self.pending_uploads.append(self.upload_pool.submit(self._write_and_upload, shard, "train", idx))

    def _flush_val(self):
        if not self.val_buffer:
            return
        shard, idx = self.val_buffer, self.val_shard_idx
        self.val_buffer = []
        self.val_bytes = 0
        self.val_shard_idx += 1
        self.pending_uploads.append(self.upload_pool.submit(self._write_and_upload, shard, "validation", idx))

    def _write_and_upload(self, texts: list[str], split: str, shard_idx: int):
        path = self.tmpdir / f"{split}-{shard_idx:05d}.parquet"
        pq.write_table(pa.table({"text": texts}), path)
        self.api.upload_file(str(path), f"data/{split}-{shard_idx:05d}.parquet", self.repo_id, repo_type="dataset")
        path.unlink()

    def finish(self):
        """Flush remaining data and wait for all uploads."""
        self._flush_train()
        self._flush_val()
        for f in tqdm(as_completed(self.pending_uploads), total=len(self.pending_uploads), desc="Uploading"):
            f.result()
        self.upload_pool.shutdown()
        shutil.rmtree(self.tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("--codebook", "-c", default="codebook_mol_1m.pkl")
    parser.add_argument("--val-size", "-v", type=int, default=100_000)
    parser.add_argument("--shard-mb", type=int, default=500, help="Target shard size in MB")
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

    # Setup uploader
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
    uploader = ShardUploader(api, args.repo_id, target_bytes=args.shard_mb * 1024 * 1024)

    # Process files in parallel, streaming results to uploader as they complete
    tasks = [(f, off, val_indices) for f, off in zip(files, offsets)]
    counter = mp.Value("q", 0)

    with ProcessPoolExecutor(args.num_workers, initializer=_init_worker, initargs=(args.codebook, counter)) as pool:
        futures = [pool.submit(_process_file, t) for t in tasks]
        pbar = tqdm(total=total, desc="Processing", unit="mol")
        pending = set(futures)

        while pending:
            pbar.n = counter.value
            pbar.refresh()

            # Stream completed results to uploader (frees memory)
            done = {f for f in pending if f.done()}
            for future in done:
                train, val = future.result()
                uploader.add_train(train)
                uploader.add_val(val)
            pending -= done

            if pending:
                time.sleep(0.1)

        pbar.n = counter.value
        pbar.close()

    # Flush remaining and wait for uploads
    uploader.finish()
    print(f"\nDone: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()

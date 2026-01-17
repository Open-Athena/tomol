"""
Serialize molecular dataset and push to HuggingFace Hub.

Dataset downloaded locally with: uv run huggingface-cli download facebook/OMol25
"""

import argparse
import multiprocessing as mp
import random
import shutil
import tempfile
from concurrent.futures import Future
from pathlib import Path
from queue import Empty

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from huggingface_hub.utils import enable_progress_bars
from tqdm import tqdm

from serialize_molecules import MoleculeTokenizer

_tokenizer: MoleculeTokenizer | None = None
_counter = None
_result_queue = None
_tmpdir = None
_batch_counter = None
_disk_batch_size = None


def _init_worker(codebook_path: str, counter: mp.Value, result_queue: mp.Queue, tmpdir: str, batch_counter: mp.Value, disk_batch_size: int):
    global _tokenizer, _counter, _result_queue, _tmpdir, _batch_counter, _disk_batch_size
    _tokenizer = MoleculeTokenizer(codebook_path)
    _counter = counter
    _result_queue = result_queue
    _tmpdir = Path(tmpdir)
    _batch_counter = batch_counter
    _disk_batch_size = disk_batch_size


def _flush_to_disk(train: list[str], val: list[str]) -> None:
    """Write accumulated batch to temp file and queue the path."""
    with _batch_counter.get_lock():
        batch_id = _batch_counter.value
        _batch_counter.value += 1

    batch_file = _tmpdir / f"batch_{batch_id:08d}.parquet"
    pq.write_table(pa.table({"train": [train], "val": [val]}), batch_file)
    _result_queue.put(str(batch_file))


def _process_file(args: tuple) -> None:
    """Process one parquet file in batches, writing results to temp files."""
    path, start_idx, val_indices = args
    pf = pq.ParquetFile(path)
    row_idx = start_idx

    # Accumulate until disk_batch_size before writing to disk
    train_accum, val_accum = [], []

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
                val_accum.append(text)
            else:
                train_accum.append(text)
            row_idx += 1

        with _counter.get_lock():
            _counter.value += len(batch_list)

        # Flush to disk when accumulated enough
        if len(train_accum) + len(val_accum) >= _disk_batch_size:
            _flush_to_disk(train_accum, val_accum)
            train_accum, val_accum = [], []

    # Flush remaining
    if train_accum or val_accum:
        _flush_to_disk(train_accum, val_accum)

    # Signal this file is done
    _result_queue.put(None)


class ShardUploader:
    """Streams shards to HuggingFace when they reach target size."""

    def __init__(self, api: HfApi, repo_id: str, shard_size: int):
        self.api = api
        self.repo_id = repo_id
        self.shard_size = shard_size
        self.tmpdir = Path(tempfile.mkdtemp(prefix="shard_upload_", dir="."))

        self.train_buffer: list[str] = []
        self.train_shard_idx = 0

        self.val_buffer: list[str] = []
        self.val_shard_idx = 0

        self.pending_uploads: list[Future] = []

    def add_train(self, texts: list[str]):
        self.train_buffer.extend(texts)
        tqdm.write(f"[add_train] +{len(texts):,} -> buffer={len(self.train_buffer):,} (threshold={self.shard_size:,})")
        while len(self.train_buffer) >= self.shard_size:
            self._flush_train()

    def add_val(self, texts: list[str]):
        self.val_buffer.extend(texts)
        tqdm.write(f"[add_val] +{len(texts):,} -> buffer={len(self.val_buffer):,} (threshold={self.shard_size:,})")
        while len(self.val_buffer) >= self.shard_size:
            self._flush_val()

    def _flush_train(self):
        if not self.train_buffer:
            return
        shard = self.train_buffer[:self.shard_size]
        self.train_buffer = self.train_buffer[self.shard_size:]
        self._write_and_upload(shard, "train", self.train_shard_idx)
        self.train_shard_idx += 1

    def _flush_val(self):
        if not self.val_buffer:
            return
        shard = self.val_buffer[:self.shard_size]
        self.val_buffer = self.val_buffer[self.shard_size:]
        self._write_and_upload(shard, "validation", self.val_shard_idx)
        self.val_shard_idx += 1

    def _cleanup_completed_uploads(self):
        """Remove completed uploads from pending list and delete their temp files."""
        still_pending = []
        for future, path, name in self.pending_uploads:
            if future.done():
                try:
                    future.result()  # Raise if failed
                    tqdm.write(f"[upload] Done {name}.parquet")
                except Exception as e:
                    tqdm.write(f"[upload] FAILED {name}.parquet: {e}")
                    raise
                path.unlink(missing_ok=True)
            else:
                still_pending.append((future, path, name))
        self.pending_uploads = still_pending

    def _write_and_upload(self, texts: list[str], split: str, shard_idx: int):
        # Clean up any completed uploads first
        self._cleanup_completed_uploads()

        path = self.tmpdir / f"{split}-{shard_idx:05d}.parquet"
        tqdm.write(f"[upload] Writing {split}-{shard_idx:05d}.parquet ({len(texts):,} rows)...")
        pq.write_table(pa.table({"text": texts}), path)
        size_mb = path.stat().st_size / 1e6
        tqdm.write(f"[upload] Uploading {split}-{shard_idx:05d}.parquet ({size_mb:.1f}MB)...")
        # Use HF's run_as_future for background upload with progress
        future = self.api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"data/{split}-{shard_idx:05d}.parquet",
            repo_id=self.repo_id,
            repo_type="dataset",
            commit_message=f"Add {split}-{shard_idx:05d}",
            run_as_future=True,
        )
        self.pending_uploads.append((future, path, f"{split}-{shard_idx:05d}"))

    def finish(self):
        """Flush remaining data and wait for all uploads."""
        if self.train_buffer:
            tqdm.write(f"Flushing final train shard {self.train_shard_idx}: {len(self.train_buffer):,} examples")
            self._write_and_upload(self.train_buffer, "train", self.train_shard_idx)
        if self.val_buffer:
            tqdm.write(f"Flushing final val shard {self.val_shard_idx}: {len(self.val_buffer):,} examples")
            self._write_and_upload(self.val_buffer, "validation", self.val_shard_idx)
        # Wait for all uploads to complete
        for future, path, name in tqdm(self.pending_uploads, desc="Waiting for uploads"):
            future.result()
            tqdm.write(f"[upload] Done {name}.parquet")
            path.unlink()
        shutil.rmtree(self.tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("repo_id")
    parser.add_argument("--codebook", "-c", default="codebook_mol_1m.pkl")
    parser.add_argument("--val-size", "-v", type=int, default=100_000)
    parser.add_argument("--shard-size", type=int, default=250_000, help="Records per shard")
    parser.add_argument("--disk-batch-size", type=int, default=50_000, help="Records per temp file")
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
    enable_progress_bars()  # Ensure HF upload progress bars are shown
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
    uploader = ShardUploader(api, args.repo_id, shard_size=args.shard_size)

    # Process files in parallel, streaming batch results via temp files
    tasks = [(f, off, val_indices) for f, off in zip(files, offsets)]
    counter = mp.Value("q", 0)
    batch_counter = mp.Value("q", 0)
    result_queue = mp.Queue()

    # Temp directory for batch files (separate from uploader's tmpdir)
    batch_tmpdir = tempfile.mkdtemp(prefix="batch_", dir=".")

    pool = mp.Pool(
        args.num_workers,
        initializer=_init_worker,
        initargs=(args.codebook, counter, result_queue, batch_tmpdir, batch_counter, args.disk_batch_size),
    )
    pool.map_async(_process_file, tasks)

    pbar = tqdm(total=total, desc="Processing", unit="mol")
    files_done = 0

    while files_done < len(files):
        pbar.n = counter.value
        pbar.refresh()

        try:
            result = result_queue.get(timeout=0.1)
            if result is None:
                files_done += 1
            else:
                # Result is a path to a temp parquet file
                batch_path = Path(result)
                table = pq.read_table(batch_path)
                train = table["train"][0].as_py()
                val = table["val"][0].as_py()
                batch_path.unlink()  # Delete immediately after reading

                uploader.add_train(train)
                uploader.add_val(val)
        except Empty:
            pass

    pbar.n = counter.value
    pbar.close()
    pool.close()
    pool.join()

    # Clean up batch tmpdir
    shutil.rmtree(batch_tmpdir, ignore_errors=True)

    # Flush remaining and wait for uploads
    uploader.finish()
    print(f"\nDone: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()

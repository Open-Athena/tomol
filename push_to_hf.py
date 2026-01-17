"""
Serialize molecular dataset and push to HuggingFace Hub.

Dataset downloaded locally with: uv run huggingface-cli download facebook/OMol25
"""

import argparse
import multiprocessing as mp
import random
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi
from tqdm import tqdm

from serialize_molecules import MoleculeTokenizer

_tokenizer: MoleculeTokenizer | None = None
_counter = None
_result_queue = None


def _init_worker(codebook_path: str, counter: mp.Value, result_queue: mp.Queue):
    global _tokenizer, _counter, _result_queue
    _tokenizer = MoleculeTokenizer(codebook_path)
    _counter = counter
    _result_queue = result_queue


def _process_file(args: tuple) -> None:
    """Process one parquet file in batches, streaming results to queue."""
    path, start_idx, val_indices = args
    pf = pq.ParquetFile(path)
    row_idx = start_idx

    for batch in pf.iter_batches(batch_size=1024, columns=["atomic_numbers", "positions", "atomic_forces", "energy"]):
        train, val = [], []
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

        # Stream batch results to queue
        _result_queue.put((train, val))

        with _counter.get_lock():
            _counter.value += len(batch_list)

    # Signal this file is done
    _result_queue.put(None)


class ShardUploader:
    """Streams shards to HuggingFace when they reach target size."""

    def __init__(self, api: HfApi, repo_id: str, shard_size: int):
        self.api = api
        self.repo_id = repo_id
        self.shard_size = shard_size
        self.tmpdir = Path(tempfile.mkdtemp())

        self.train_buffer: list[str] = []
        self.train_shard_idx = 0

        self.val_buffer: list[str] = []
        self.val_shard_idx = 0

        self.upload_pool = ThreadPoolExecutor(4)
        self.pending_uploads: list = []

    def add_train(self, texts: list[str]):
        self.train_buffer.extend(texts)
        print(f"[add_train] +{len(texts):,} -> buffer={len(self.train_buffer):,} (threshold={self.shard_size:,})")
        while len(self.train_buffer) >= self.shard_size:
            self._flush_train()

    def add_val(self, texts: list[str]):
        self.val_buffer.extend(texts)
        print(f"[add_val] +{len(texts):,} -> buffer={len(self.val_buffer):,} (threshold={self.shard_size:,})")
        while len(self.val_buffer) >= self.shard_size:
            self._flush_val()

    def _flush_train(self):
        if not self.train_buffer:
            return
        shard = self.train_buffer[:self.shard_size]
        self.train_buffer = self.train_buffer[self.shard_size:]
        print(f"Flushing train shard {self.train_shard_idx}: {len(shard):,} examples")
        self.pending_uploads.append(self.upload_pool.submit(self._write_and_upload, shard, "train", self.train_shard_idx))
        self.train_shard_idx += 1

    def _flush_val(self):
        if not self.val_buffer:
            return
        shard = self.val_buffer[:self.shard_size]
        self.val_buffer = self.val_buffer[self.shard_size:]
        print(f"Flushing val shard {self.val_shard_idx}: {len(shard):,} examples")
        self.pending_uploads.append(self.upload_pool.submit(self._write_and_upload, shard, "validation", self.val_shard_idx))
        self.val_shard_idx += 1

    def _write_and_upload(self, texts: list[str], split: str, shard_idx: int):
        path = self.tmpdir / f"{split}-{shard_idx:05d}.parquet"
        pq.write_table(pa.table({"text": texts}), path)
        self.api.upload_file(str(path), f"data/{split}-{shard_idx:05d}.parquet", self.repo_id, repo_type="dataset")
        path.unlink()

    def finish(self):
        """Flush remaining data and wait for all uploads."""
        if self.train_buffer:
            print(f"Flushing final train shard {self.train_shard_idx}: {len(self.train_buffer):,} examples")
            self.pending_uploads.append(self.upload_pool.submit(self._write_and_upload, self.train_buffer, "train", self.train_shard_idx))
        if self.val_buffer:
            print(f"Flushing final val shard {self.val_shard_idx}: {len(self.val_buffer):,} examples")
            self.pending_uploads.append(self.upload_pool.submit(self._write_and_upload, self.val_buffer, "validation", self.val_shard_idx))
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
    parser.add_argument("--shard-size", type=int, default=250_000, help="Records per shard")
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
    uploader = ShardUploader(api, args.repo_id, shard_size=args.shard_size)

    # Process files in parallel, streaming batch results via queue
    tasks = [(f, off, val_indices) for f, off in zip(files, offsets)]
    counter = mp.Value("q", 0)
    result_queue = mp.Queue()

    pool = mp.Pool(args.num_workers, initializer=_init_worker, initargs=(args.codebook, counter, result_queue))
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
                train, val = result
                uploader.add_train(train)
                uploader.add_val(val)
        except Empty:
            pass

    pbar.n = counter.value
    pbar.close()
    pool.close()
    pool.join()

    # Flush remaining and wait for uploads
    uploader.finish()
    print(f"\nDone: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()

# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Sample 1M rows from local OMol25 parquet files and write to CSV.

Uses local parquet files from HuggingFace cache.
"""

import csv
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

# Local parquet directory
PARQUET_DIR = Path("./hf_cache/hub/datasets--colabfit--OMol25_train/snapshots/cf406cd59287d88e41df6f354c0d41a34cb13dc3/co/")
SAMPLE_SIZE = 1_000_000
OUTPUT_PATH = Path("omol25_train_sample_1m.csv")
SHUFFLE_SEED = 0
COLUMNS = [
    "atomic_numbers",
    "positions",
    "atomic_forces",
    "energy",
]


def _jsonify(value: object) -> object:
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def _write_csv() -> None:
    random.seed(SHUFFLE_SEED)

    # Find all parquet files
    files = sorted(PARQUET_DIR.glob("*.parquet"))
    print(f"Found {len(files)} parquet files")

    # Count total rows and build index
    print("Counting rows...")
    file_info = []
    total_rows = 0
    for f in tqdm(files, desc="Scanning files"):
        pf = pq.ParquetFile(f)
        num_rows = pf.metadata.num_rows
        file_info.append((f, total_rows, num_rows))
        total_rows += num_rows

    print(f"Total rows available: {total_rows:,}")
    sample_size = min(SAMPLE_SIZE, total_rows)
    print(f"Sampling {sample_size:,} rows...")

    # Generate random indices to sample
    sample_indices = set(random.sample(range(total_rows), sample_size))

    # Read and write samples
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    samples_written = 0

    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()

        global_idx = 0
        for filepath, start_idx, num_rows in tqdm(file_info, desc="Processing files"):
            # Check if any samples are in this file
            file_end = start_idx + num_rows
            file_samples = [i for i in range(start_idx, file_end) if i in sample_indices]

            if not file_samples:
                global_idx = file_end
                continue

            # Read the file
            table = pq.read_table(filepath, columns=COLUMNS)
            batch = table.to_pylist()

            for local_idx, row in enumerate(batch):
                global_row_idx = start_idx + local_idx
                if global_row_idx in sample_indices:
                    writer.writerow({key: _jsonify(row.get(key)) for key in COLUMNS})
                    samples_written += 1

            global_idx = file_end

    print(f"\nWrote {samples_written:,} samples to {OUTPUT_PATH}")


def main() -> None:
    _write_csv()


if __name__ == "__main__":
    main()

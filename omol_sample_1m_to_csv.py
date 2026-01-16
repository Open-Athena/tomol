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
Stream a 1M approximate random sample from colabfit/OMol25_train and write it to CSV.

Uses Hugging Face streaming shuffle with a fixed buffer for single-pass sampling.
"""

import csv
import json
import os
from pathlib import Path

import datasets
from datasets import load_dataset
from huggingface_hub import get_token
from tqdm.auto import tqdm

DATASET_ID = "colabfit/OMol25_train"
DATASET_REVISION = "main"
SAMPLE_SIZE = 1_000_000
OUTPUT_PATH = Path("omol25_train_sample_1m.csv")
SHUFFLE_SEED = 0
SHUFFLE_BUFFER = 10_000
COLUMNS = [
    "property_id",
    "configuration_id",
    "dataset_id",
    "atomic_numbers",
    "positions",
    "atomic_forces",
    "energy",
    "multiplicity",
    "cell",
    "pbc",
]


def _jsonify(value: object) -> object:
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def _write_csv() -> None:
    token = get_token()
    if token:
        os.environ.setdefault("HF_TOKEN", token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", token)

    datasets.disable_caching()
    ds = (
        load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        split="train",
        streaming=True,
        )
        .select_columns(COLUMNS)
        .shuffle(seed=SHUFFLE_SEED, buffer_size=SHUFFLE_BUFFER)
        .take(SAMPLE_SIZE)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in tqdm(ds, total=SAMPLE_SIZE, desc="Writing sample"):
            writer.writerow({key: _jsonify(row.get(key)) for key in COLUMNS})


def main() -> None:
    _write_csv()


if __name__ == "__main__":
    main()

"""Push a small sample of serialized molecules to HuggingFace."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from huggingface_hub import HfApi
from serialize_molecules import MoleculeTokenizer

SOURCE_DIR = Path("./hf_cache/hub/datasets--colabfit--OMol25_train/snapshots/cf406cd59287d88e41df6f354c0d41a34cb13dc3/co/")
REPO_ID = "WillHeld/tomol-test"
CONFIG = "fp16_config.json"
SAMPLE_SIZE = 1000
VAL_SIZE = 100

def main():
    tokenizer = MoleculeTokenizer(CONFIG)
    print("Vocabulary:", tokenizer.get_vocab_info())

    # Find parquet files
    files = sorted(SOURCE_DIR.glob("*.parquet"))
    print(f"Found {len(files)} files")

    # Read and serialize samples
    texts = []
    for f in files:
        if len(texts) >= SAMPLE_SIZE:
            break
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=256, columns=["atomic_numbers", "positions", "atomic_forces", "energy"]):
            for row in batch.to_pylist():
                tokens = tokenizer.encode_molecule(
                    row["atomic_numbers"],
                    np.array(row["positions"]),
                    np.array(row["atomic_forces"]),
                    float(row["energy"]),
                )
                texts.append(tokenizer.tokens_to_string(tokens))
                if len(texts) >= SAMPLE_SIZE:
                    break
            if len(texts) >= SAMPLE_SIZE:
                break

    print(f"Serialized {len(texts)} molecules")

    # Split into train/val
    train_texts = texts[:-VAL_SIZE]
    val_texts = texts[-VAL_SIZE:]
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

    # Create and upload
    api = HfApi()
    api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True)

    # Write and upload train
    train_path = Path("train-00000.parquet")
    pq.write_table(pa.table({"text": train_texts}), train_path)
    print(f"Uploading train ({train_path.stat().st_size / 1e6:.2f}MB)...")
    api.upload_file(
        path_or_fileobj=str(train_path),
        path_in_repo="data/train-00000.parquet",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Add train split",
    )
    train_path.unlink()

    # Write and upload val
    val_path = Path("validation-00000.parquet")
    pq.write_table(pa.table({"text": val_texts}), val_path)
    print(f"Uploading validation ({val_path.stat().st_size / 1e6:.2f}MB)...")
    api.upload_file(
        path_or_fileobj=str(val_path),
        path_in_repo="data/validation-00000.parquet",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Add validation split",
    )
    val_path.unlink()

    print(f"\nDone: https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    main()

# moltok

Tokenizer and data prep utilities for turning molecular structures into token sequences using
Residual Vector Quantization (RVQ) codebooks. This repo focuses on:
- sampling OMol25 into CSV
- building RVQ codebooks for positions, forces, and energy
- serializing molecules into token sequences
- pushing tokenized shards and the tokenizer to the Hugging Face Hub

## Repo Contents
- `omol_sample_1m_to_csv.py`: stream a 1M sample from `colabfit/OMol25_train` into CSV.
- `build_rvq_codebooks.py`: build RVQ codebooks (quantile or k-means) and evaluate reconstruction.
- `serialize_molecules.py`: `MoleculeTokenizer` + CSV serialization to token sequences.
- `push_to_hf.py`: tokenize parquet shards and upload dataset shards to HF.
- `push_tokenizer_to_hf.py`: upload the tokenizer as a HF model repo.
- `test_roundtrip.py`: round-trip test through HF tokenizer and decode back to molecules.
- `omol25_train_sample_1k.csv`: small sample for quick tests.
- `codebook_mol_1m.pkl`: prebuilt codebook used by the scripts.

## Setup
Python 3.12+ is required.

Install dependencies (one of):
```bash
uv sync
```
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

## Token Format
Tokens are organized into sections so they can be shuffled for flexible conditioning:
```
[BOS]
[ATOMS] [Z=6] [Z=1] ... [ATOMS_END]
[POS]   [P0:...] ... [NL] ... [POS_END]
[FORCE] [FX0:...] [FY0:...] [FZ0:...] [NL] ... [FORCE_END]
[ENERGY] [E0:...] [NL] [ENERGY_END]
[EOS]
```
`[NL]` is a newline token that separates per-atom rows in the position/force sections.

## Quickstart (using the included sample)
```bash
python3 serialize_molecules.py omol25_train_sample_1k.csv \
  --codebook codebook_mol_1m.pkl \
  --output tokenized_molecules.pkl \
  --show-examples 1
```

## Workflows
### 1) Sample OMol25 to CSV
```bash
python3 omol_sample_1m_to_csv.py
```
This streams a 1M-row sample from HF `colabfit/OMol25_train` and writes `omol25_train_sample_1m.csv`.
If you are authenticated with HF, the script will pick up `HF_TOKEN` automatically.

### 2) Build RVQ codebooks
```bash
python3 build_rvq_codebooks.py omol25_train_sample_1m.csv \
  --output codebook_mol_1m.pkl \
  --method quantile \
  --pos-levels 8 \
  --n-levels 4 \
  --codebook-size 256
```
Quantile mode uses Morton (Z-order) binning for 3D positions. Use `--method kmeans` for k-means.

### 3) Serialize molecules to tokens
```bash
python3 serialize_molecules.py omol25_train_sample_1m.csv \
  --codebook codebook_mol_1m.pkl \
  --output tokenized_molecules.pkl
```

### 4) Push tokenized dataset to HF
`push_to_hf.py` expects local OMol25 parquet shards (downloaded with `huggingface-cli download facebook/OMol25`)
and writes train/validation shards to a dataset repo.
```bash
python3 push_to_hf.py /path/to/omol25_parquet_dir WillHeld/Tomol25 \
  --codebook codebook_mol_1m.pkl
```
Dataset repo: https://huggingface.co/datasets/WillHeld/Tomol25

### 5) Push tokenizer to HF
```bash
python3 push_tokenizer_to_hf.py --repo-id WillHeld/marin-tomol
```
Tokenizer repo: https://huggingface.co/WillHeld/marin-tomol

### 6) Round-trip test
```bash
python3 test_roundtrip.py
```
This uses `omol25_train_sample_1k.csv`, `codebook_mol_1m.pkl`, and the HF tokenizer repo
`WillHeld/marin-tomol`.

#!/usr/bin/env python3
"""
Analyze token distribution and reconstruction from local parquet files.

Usage:
    python analyze_from_parquet.py ./hf_cache/hub/datasets--colabfit--OMol25_train/snapshots/cf406cd59287d88e41df6f354c0d41a34cb13dc3/co/ \
        --config fp16_config.json \
        --max-rows 1000000
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from serialize_molecules import (
    MoleculeTokenizer,
    VOCAB_SIZE,
    SPECIAL_TOKENS,
    ATOM_TOKEN_OFFSET,
    MAX_ATOMIC_NUMBER,
    DIM_STARTS,
    DIM_PREFIXES,
    SIGNED_DIM_TOKENS,
    UNSIGNED_DIM_TOKENS,
)
from build_fp16_config import build_config as make_config


def analyze_parquet(
    parquet_dir: str,
    config_path: str,
    max_rows: int = 1_000_000,
    build_new_config: bool = False,
):
    """Analyze token distribution and reconstruction quality from parquet files."""

    print("=" * 70)
    print("Token Distribution and Reconstruction Analysis (from Parquet)")
    print("=" * 70)

    # Find parquet files
    parquet_dir = Path(parquet_dir)
    files = sorted(parquet_dir.glob("*.parquet"))
    print(f"\nFound {len(files)} parquet files in {parquet_dir}")

    # Count total rows
    total_available = 0
    file_rows = []
    for f in files:
        pf = pq.ParquetFile(f)
        rows = pf.metadata.num_rows
        file_rows.append((f, rows))
        total_available += rows
    print(f"Total rows available: {total_available:,}")
    print(f"Will process: {min(max_rows, total_available):,}")

    # First pass: collect data for config building if needed
    if build_new_config:
        print(f"\nBuilding FP16 config from data...")
        positions_list = []
        forces_list = []
        energies_list = []

        rows_read = 0
        config_sample = min(100_000, max_rows)  # Use 100k for config

        for filepath, num_rows in tqdm(file_rows, desc="Reading for config"):
            if rows_read >= config_sample:
                break

            table = pq.read_table(filepath, columns=["atomic_numbers", "positions", "atomic_forces", "energy"])
            batch = table.to_pylist()

            for row in batch:
                if rows_read >= config_sample:
                    break
                positions_list.append(np.array(row["positions"]))
                forces_list.append(np.array(row["atomic_forces"]))
                energies_list.append(row["energy"])
                rows_read += 1

        energies = np.array(energies_list)
        config = make_config(positions_list, forces_list, energies)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to {config_path}")

    # Load tokenizer
    print(f"\nLoading tokenizer with config: {config_path}")
    tokenizer = MoleculeTokenizer(config_path)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Collect statistics
    token_counts = Counter()
    total_tokens = 0

    pos_errors = []
    force_errors = []
    energy_errors = []

    tokens_per_molecule = []
    atoms_per_molecule = []

    rows_processed = 0

    print(f"\nProcessing molecules...")
    pbar = tqdm(total=min(max_rows, total_available), desc="Analyzing")

    for filepath, num_rows in file_rows:
        if rows_processed >= max_rows:
            break

        # Read in batches for memory efficiency
        pf = pq.ParquetFile(filepath)
        for batch in pf.iter_batches(batch_size=1024, columns=["atomic_numbers", "positions", "atomic_forces", "energy"]):
            if rows_processed >= max_rows:
                break

            batch_list = batch.to_pylist()

            for row in batch_list:
                if rows_processed >= max_rows:
                    break

                atomic_numbers = row["atomic_numbers"]
                positions = np.array(row["positions"])
                forces = np.array(row["atomic_forces"])
                energy = row["energy"]

                # Encode
                tokens = tokenizer.encode_molecule(atomic_numbers, positions, forces, energy)
                tokens_per_molecule.append(len(tokens))
                atoms_per_molecule.append(len(atomic_numbers))

                # Count tokens
                for tok in tokens:
                    token_counts[tok] += 1
                    total_tokens += 1

                # Decode and compute reconstruction error
                decoded = tokenizer.decode_molecule(tokens)

                # Position error (centered)
                pos_orig = positions - positions.mean(axis=0)
                pos_dec = decoded["positions"]
                if len(pos_dec) > 0:
                    pos_dec = pos_dec - pos_dec.mean(axis=0)
                    pos_errors.append(np.mean(np.abs(pos_orig - pos_dec)))

                # Force error
                if len(decoded["forces"]) == len(forces):
                    force_errors.append(np.mean(np.abs(forces - decoded["forces"])))

                # Energy error
                energy_errors.append(abs(energy - decoded["energy"]))

                rows_processed += 1
                pbar.update(1)

    pbar.close()

    # Print reconstruction statistics
    print("\n" + "=" * 70)
    print("RECONSTRUCTION QUALITY")
    print("=" * 70)

    pos_errors = np.array(pos_errors)
    force_errors = np.array(force_errors)
    energy_errors = np.array(energy_errors)

    print(f"\nPosition MAE:")
    print(f"  Mean:   {np.mean(pos_errors)*1000:.4f} mÅ")
    print(f"  Median: {np.median(pos_errors)*1000:.4f} mÅ")
    print(f"  Max:    {np.max(pos_errors)*1000:.4f} mÅ")
    print(f"  P99:    {np.percentile(pos_errors, 99)*1000:.4f} mÅ")

    print(f"\nForce MAE:")
    print(f"  Mean:   {np.mean(force_errors)*1000:.4f} meV/Å")
    print(f"  Median: {np.median(force_errors)*1000:.4f} meV/Å")
    print(f"  Max:    {np.max(force_errors)*1000:.4f} meV/Å")
    print(f"  P99:    {np.percentile(force_errors, 99)*1000:.4f} meV/Å")

    print(f"\nEnergy error:")
    print(f"  Mean:   {np.mean(energy_errors)*1000:.4f} meV")
    print(f"  Median: {np.median(energy_errors)*1000:.4f} meV")
    print(f"  Max:    {np.max(energy_errors)*1000:.4f} meV")
    print(f"  P99:    {np.percentile(energy_errors, 99)*1000:.4f} meV")

    # Token statistics
    print("\n" + "=" * 70)
    print("TOKEN STATISTICS")
    print("=" * 70)

    print(f"\nTotal tokens: {total_tokens:,}")
    print(f"Unique tokens used: {len(token_counts):,} / {VOCAB_SIZE}")
    print(f"Tokens per molecule: mean={np.mean(tokens_per_molecule):.1f}, "
          f"median={np.median(tokens_per_molecule):.1f}, "
          f"min={np.min(tokens_per_molecule)}, max={np.max(tokens_per_molecule)}")
    print(f"Atoms per molecule: mean={np.mean(atoms_per_molecule):.1f}, "
          f"median={np.median(atoms_per_molecule):.1f}, "
          f"min={np.min(atoms_per_molecule)}, max={np.max(atoms_per_molecule)}")

    # Analyze distribution by token type
    print("\n" + "-" * 50)
    print("TOKEN DISTRIBUTION BY TYPE")
    print("-" * 50)

    # Special tokens
    special_count = sum(token_counts[tid] for tid in SPECIAL_TOKENS.values())
    print(f"\nSpecial tokens: {special_count:,} ({100*special_count/total_tokens:.2f}%)")

    # Atom tokens
    atom_count = sum(token_counts[ATOM_TOKEN_OFFSET + z - 1] for z in range(1, MAX_ATOMIC_NUMBER + 1))
    print(f"Atom tokens: {atom_count:,} ({100*atom_count/total_tokens:.2f}%)")

    # Per-dimension analysis
    for dim_name, dim_start in DIM_STARTS.items():
        prefix = DIM_PREFIXES[dim_name]
        if dim_name == "energy":
            # Energy: Exp(256) + M0(256) + M1(256) + M2(256) = 1024 tokens
            exp_start = dim_start
            m0_start = dim_start + 256
            m1_start = dim_start + 512
            m2_start = dim_start + 768

            exp_count = sum(token_counts[exp_start + i] for i in range(256))
            m0_count = sum(token_counts[m0_start + i] for i in range(256))
            m1_count = sum(token_counts[m1_start + i] for i in range(256))
            m2_count = sum(token_counts[m2_start + i] for i in range(256))

            total_dim = exp_count + m0_count + m1_count + m2_count
            print(f"\n{prefix}: {total_dim:,} ({100*total_dim/total_tokens:.2f}%)")
            print(f"  Exp:   {exp_count:,}")
            print(f"  Mant0: {m0_count:,}")
            print(f"  Mant1: {m1_count:,}")
            print(f"  Mant2: {m2_count:,}")
        else:
            # Signed dims: SignExp(512) + M0(256) + M1(256) = 1024 tokens
            # SignExp: first 256 are positive, next 256 are negative
            signexp_pos_count = sum(token_counts[dim_start + i] for i in range(256))
            signexp_neg_count = sum(token_counts[dim_start + 256 + i] for i in range(256))
            m0_start = dim_start + 512
            m1_start = dim_start + 768

            m0_count = sum(token_counts[m0_start + i] for i in range(256))
            m1_count = sum(token_counts[m1_start + i] for i in range(256))

            total_dim = signexp_pos_count + signexp_neg_count + m0_count + m1_count
            print(f"\n{prefix}: {total_dim:,} ({100*total_dim/total_tokens:.2f}%)")
            signexp_total = signexp_pos_count + signexp_neg_count
            if signexp_total > 0:
                print(f"  SignExp+: {signexp_pos_count:,} ({100*signexp_pos_count/signexp_total:.1f}%)")
                print(f"  SignExp-: {signexp_neg_count:,} ({100*signexp_neg_count/signexp_total:.1f}%)")
            print(f"  Mant0: {m0_count:,}")
            print(f"  Mant1: {m1_count:,}")

    # Uniformity analysis for mantissa tokens
    print("\n" + "-" * 50)
    print("MANTISSA UNIFORMITY ANALYSIS")
    print("-" * 50)
    print("(Ideal uniform distribution: each bin ~0.39%)")

    for dim_name, dim_start in DIM_STARTS.items():
        prefix = DIM_PREFIXES[dim_name]

        if dim_name == "energy":
            m0_start = dim_start + 256
            m1_start = dim_start + 512
        else:
            # New layout: SignExp(512) + M0(256) + M1(256)
            m0_start = dim_start + 512
            m1_start = dim_start + 768

        # Get M0 distribution
        m0_counts = [token_counts[m0_start + i] for i in range(256)]
        m0_total = sum(m0_counts)
        if m0_total > 0:
            m0_pcts = [100 * c / m0_total for c in m0_counts]
            m0_std = np.std(m0_pcts)
            m0_min = min(m0_pcts)
            m0_max = max(m0_pcts)
            print(f"\n{prefix}_M0: std={m0_std:.4f}%, range=[{m0_min:.3f}%, {m0_max:.3f}%]")

        # Get M1 distribution
        m1_counts = [token_counts[m1_start + i] for i in range(256)]
        m1_total = sum(m1_counts)
        if m1_total > 0:
            m1_pcts = [100 * c / m1_total for c in m1_counts]
            m1_std = np.std(m1_pcts)
            m1_min = min(m1_pcts)
            m1_max = max(m1_pcts)
            print(f"{prefix}_M1: std={m1_std:.4f}%, range=[{m1_min:.3f}%, {m1_max:.3f}%]")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze token distribution from parquet files")
    parser.add_argument("parquet_dir", type=str, help="Directory containing parquet files")
    parser.add_argument("--config", "-c", type=str, default="fp16_config.json",
                        help="Path to FP16 config JSON file")
    parser.add_argument("--max-rows", type=int, default=1_000_000,
                        help="Maximum rows to process")
    parser.add_argument("--build-config", action="store_true",
                        help="Build new config from data before analysis")
    args = parser.parse_args()

    analyze_parquet(
        args.parquet_dir,
        args.config,
        max_rows=args.max_rows,
        build_new_config=args.build_config,
    )


if __name__ == "__main__":
    main()

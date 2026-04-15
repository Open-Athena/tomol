#!/usr/bin/env python3
"""
Analyze token distribution and reconstruction quality on a large sample.

Usage:
    python analyze_token_distribution.py omol25_train_sample_1m.csv --config fp16_config.json
"""

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
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


def parse_array(s: str) -> np.ndarray:
    return np.array(ast.literal_eval(s))


def analyze_distribution(
    csv_path: str,
    config_path: str,
    max_rows: int | None = None,
    build_config: bool = False,
):
    """Analyze token distribution and reconstruction quality."""

    print("=" * 70)
    print("Token Distribution and Reconstruction Analysis")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from {csv_path}...")
    df = pd.read_csv(csv_path, nrows=max_rows)
    print(f"Loaded {len(df):,} molecules")

    # Optionally build config from this data
    if build_config:
        print(f"\nBuilding FP16 config from data...")
        from build_fp16_config import build_config as make_config, load_data
        positions, forces, energies = load_data(csv_path, max_rows)
        config = make_config(positions, forces, energies)
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

    print(f"\nProcessing molecules...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing"):
        atomic_numbers = ast.literal_eval(row["atomic_numbers"])
        positions = parse_array(row["positions"])
        forces = parse_array(row["atomic_forces"])
        energy = float(row["energy"])

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
            dim_size = UNSIGNED_DIM_TOKENS
            # Energy has no sign tokens
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
            dim_size = SIGNED_DIM_TOKENS
            sign_pos = token_counts[dim_start]
            sign_neg = token_counts[dim_start + 1]
            exp_start = dim_start + 2
            m0_start = dim_start + 2 + 256
            m1_start = dim_start + 2 + 512

            exp_count = sum(token_counts[exp_start + i] for i in range(256))
            m0_count = sum(token_counts[m0_start + i] for i in range(256))
            m1_count = sum(token_counts[m1_start + i] for i in range(256))

            total_dim = sign_pos + sign_neg + exp_count + m0_count + m1_count
            print(f"\n{prefix}: {total_dim:,} ({100*total_dim/total_tokens:.2f}%)")
            print(f"  Sign+: {sign_pos:,} ({100*sign_pos/(sign_pos+sign_neg):.1f}%)")
            print(f"  Sign-: {sign_neg:,} ({100*sign_neg/(sign_pos+sign_neg):.1f}%)")
            print(f"  Exp:   {exp_count:,}")
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
            m0_start = dim_start + 2 + 256
            m1_start = dim_start + 2 + 512

        # Get M0 distribution
        m0_counts = [token_counts[m0_start + i] for i in range(256)]
        m0_total = sum(m0_counts)
        if m0_total > 0:
            m0_pcts = [100 * c / m0_total for c in m0_counts]
            m0_std = np.std(m0_pcts)
            m0_min = min(m0_pcts)
            m0_max = max(m0_pcts)
            print(f"\n{prefix}_M0: std={m0_std:.3f}%, range=[{m0_min:.2f}%, {m0_max:.2f}%]")

        # Get M1 distribution
        m1_counts = [token_counts[m1_start + i] for i in range(256)]
        m1_total = sum(m1_counts)
        if m1_total > 0:
            m1_pcts = [100 * c / m1_total for c in m1_counts]
            m1_std = np.std(m1_pcts)
            m1_min = min(m1_pcts)
            m1_max = max(m1_pcts)
            print(f"{prefix}_M1: std={m1_std:.3f}%, range=[{m1_min:.2f}%, {m1_max:.2f}%]")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return {
        "total_tokens": total_tokens,
        "unique_tokens": len(token_counts),
        "pos_mae_mean": np.mean(pos_errors),
        "force_mae_mean": np.mean(force_errors),
        "energy_error_mean": np.mean(energy_errors),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze token distribution and reconstruction")
    parser.add_argument("csv_path", type=str, help="Path to CSV file")
    parser.add_argument("--config", "-c", type=str, default="fp16_config.json",
                        help="Path to FP16 config JSON file")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Maximum rows to process")
    parser.add_argument("--build-config", action="store_true",
                        help="Build config from data before analysis")
    args = parser.parse_args()

    analyze_distribution(
        args.csv_path,
        args.config,
        max_rows=args.max_rows,
        build_config=args.build_config,
    )


if __name__ == "__main__":
    main()

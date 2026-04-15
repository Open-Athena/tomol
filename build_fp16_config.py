#!/usr/bin/env python3
"""
Build FP16-like encoding config for molecular data.

Computes log_min/log_max ranges per dimension from training data.
These ranges are used to encode floating-point values into tokens
with uniform distribution across mantissa bins.

Usage:
    python build_fp16_config.py omol25_train_sample_1k.csv \
        --output fp16_config.json \
        --train-rows 900 \
        --val-rows 100
"""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_array(s: str) -> np.ndarray:
    """Parse a string representation of a nested list into a numpy array."""
    return np.array(ast.literal_eval(s))


def load_data(csv_path: str, max_rows: int | None = None) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """
    Load molecular data from CSV.

    Returns:
        positions: List of (N_atoms, 3) arrays, one per molecule
        forces: List of (N_atoms, 3) arrays, one per molecule
        energies: (N_molecules,) array of scalar energies
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, nrows=max_rows)

    positions = []
    forces = []
    energies = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing molecules"):
        pos = parse_array(row["positions"])
        force = parse_array(row["atomic_forces"])
        energy = float(row["energy"])

        positions.append(pos)
        forces.append(force)
        energies.append(energy)

    return positions, forces, np.array(energies)


def compute_log_range(values: np.ndarray, percentile_low: float = 0.01, percentile_high: float = 99.99) -> tuple[float, float]:
    """
    Compute log_min and log_max for a set of values.

    Uses percentiles to avoid extreme outliers affecting the range.
    Values are treated as absolute values (magnitudes).

    Args:
        values: Array of values (can be positive or negative)
        percentile_low: Lower percentile for range
        percentile_high: Upper percentile for range

    Returns:
        (log_min, log_max): Tuple of log10 bounds for the magnitude range
    """
    # Take absolute values for magnitude
    magnitudes = np.abs(values)

    # Filter out exact zeros
    nonzero = magnitudes[magnitudes > 0]

    if len(nonzero) == 0:
        # All zeros - use default range
        return -10.0, 0.0

    # Get percentile range
    low_mag = np.percentile(nonzero, percentile_low)
    high_mag = np.percentile(nonzero, percentile_high)

    # Convert to log scale with padding for outliers
    log_min = np.log10(low_mag) - 1.0  # 1 log-unit padding below
    log_max = np.log10(high_mag) + 1.0  # 1 log-unit padding above

    return float(log_min), float(log_max)


def build_config(
    positions: list[np.ndarray],
    forces: list[np.ndarray],
    energies: np.ndarray,
) -> dict:
    """
    Build FP16-like encoding config from data.

    Args:
        positions: List of (N_atoms, 3) position arrays
        forces: List of (N_atoms, 3) force arrays
        energies: (N_molecules,) energy array

    Returns:
        Config dict with log_min/log_max per dimension
    """
    # Flatten positions and forces
    all_positions = np.vstack(positions)  # (N_total, 3)
    all_forces = np.vstack(forces)  # (N_total, 3)

    # Center positions per molecule before computing ranges
    centered_positions = []
    for pos in positions:
        centered = pos - pos.mean(axis=0, keepdims=True)
        centered_positions.append(centered)
    all_centered_positions = np.vstack(centered_positions)

    config = {
        "encoding_type": "fp16_like",
        "dimensions": {}
    }

    # Position dimensions (centered)
    print("\nComputing position ranges (centered)...")
    for i, dim_name in enumerate(["pos_x", "pos_y", "pos_z"]):
        log_min, log_max = compute_log_range(all_centered_positions[:, i])
        config["dimensions"][dim_name] = {
            "log_min": log_min,
            "log_max": log_max,
        }
        print(f"  {dim_name}: log_min={log_min:.2f}, log_max={log_max:.2f}")

    # Force dimensions
    print("\nComputing force ranges...")
    for i, dim_name in enumerate(["force_x", "force_y", "force_z"]):
        log_min, log_max = compute_log_range(all_forces[:, i])
        config["dimensions"][dim_name] = {
            "log_min": log_min,
            "log_max": log_max,
        }
        print(f"  {dim_name}: log_min={log_min:.2f}, log_max={log_max:.2f}")

    # Energy dimension (always negative in OMol25, so we take negative for magnitude)
    print("\nComputing energy range...")
    # OMol25 energies are typically large negative numbers
    energy_magnitudes = np.abs(energies)
    log_min, log_max = compute_log_range(energy_magnitudes)
    config["dimensions"]["energy"] = {
        "log_min": log_min,
        "log_max": log_max,
    }
    print(f"  energy: log_min={log_min:.2f}, log_max={log_max:.2f}")

    return config


def validate_config(config: dict, positions: list[np.ndarray], forces: list[np.ndarray], energies: np.ndarray) -> None:
    """Validate that the config covers the data well."""
    print("\nValidating config coverage...")

    all_positions = []
    for pos in positions:
        centered = pos - pos.mean(axis=0, keepdims=True)
        all_positions.append(centered)
    all_positions = np.vstack(all_positions)
    all_forces = np.vstack(forces)

    dims = config["dimensions"]

    # Check position coverage
    for i, dim_name in enumerate(["pos_x", "pos_y", "pos_z"]):
        values = all_positions[:, i]
        magnitudes = np.abs(values[values != 0])
        if len(magnitudes) > 0:
            log_vals = np.log10(magnitudes)
            in_range = (log_vals >= dims[dim_name]["log_min"]) & (log_vals <= dims[dim_name]["log_max"])
            coverage = np.mean(in_range) * 100
            print(f"  {dim_name}: {coverage:.2f}% values in range")

    # Check force coverage
    for i, dim_name in enumerate(["force_x", "force_y", "force_z"]):
        values = all_forces[:, i]
        magnitudes = np.abs(values[values != 0])
        if len(magnitudes) > 0:
            log_vals = np.log10(magnitudes)
            in_range = (log_vals >= dims[dim_name]["log_min"]) & (log_vals <= dims[dim_name]["log_max"])
            coverage = np.mean(in_range) * 100
            print(f"  {dim_name}: {coverage:.2f}% values in range")

    # Check energy coverage
    magnitudes = np.abs(energies)
    log_vals = np.log10(magnitudes)
    in_range = (log_vals >= dims["energy"]["log_min"]) & (log_vals <= dims["energy"]["log_max"])
    coverage = np.mean(in_range) * 100
    print(f"  energy: {coverage:.2f}% values in range")


def main():
    parser = argparse.ArgumentParser(description="Build FP16-like encoding config for molecular data")
    parser.add_argument("input_csv", type=str, help="Path to input CSV file")
    parser.add_argument("--output", "-o", type=str, default="fp16_config.json", help="Output JSON config file")
    parser.add_argument("--train-rows", type=int, default=None, help="Number of rows to use for training")
    parser.add_argument("--val-rows", type=int, default=None, help="Number of rows for validation")
    args = parser.parse_args()

    # Load data
    max_rows = None
    if args.train_rows is not None:
        max_rows = args.train_rows + (args.val_rows or 0)

    positions, forces, energies = load_data(args.input_csv, max_rows)

    # Split into train/val if specified
    if args.train_rows is not None and args.val_rows is not None:
        train_positions = positions[:args.train_rows]
        train_forces = forces[:args.train_rows]
        train_energies = energies[:args.train_rows]

        val_positions = positions[args.train_rows:args.train_rows + args.val_rows]
        val_forces = forces[args.train_rows:args.train_rows + args.val_rows]
        val_energies = energies[args.train_rows:args.train_rows + args.val_rows]

        print(f"\nSplit: {len(train_positions)} train, {len(val_positions)} val molecules")
    else:
        train_positions = positions
        train_forces = forces
        train_energies = energies
        val_positions = val_forces = val_energies = None

    # Build config from training data
    print("\n" + "=" * 60)
    print("Building FP16 Config from Training Data")
    print("=" * 60)

    config = build_config(train_positions, train_forces, train_energies)

    # Validate on training data
    print("\n" + "-" * 40)
    print("Training Data Coverage")
    print("-" * 40)
    validate_config(config, train_positions, train_forces, train_energies)

    # Validate on validation data if available
    if val_positions is not None:
        print("\n" + "-" * 40)
        print("Validation Data Coverage")
        print("-" * 40)
        validate_config(config, val_positions, val_forces, val_energies)

    # Save config
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nConfig saved to {output_path}")
    print("\nConfig summary:")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()

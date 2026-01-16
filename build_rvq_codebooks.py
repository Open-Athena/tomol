"""
Build Residual Vector Quantization (RVQ) codebooks for molecular data.

Supports two methods for codebook construction:
- quantile: Fast, deterministic binning using quantile positions (default)
- kmeans: K-means clustering for potentially better reconstruction

Positions use Morton (Z-order) curve to map 3D to 1D while preserving spatial locality (quantile mode).
Forces and energies use direct 1D binning on scalar values.
"""

import argparse
import ast
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
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


def preprocess_positions(positions: list[np.ndarray], scale: float | None = None) -> tuple[np.ndarray, float]:
    """
    Preprocess position vectors by centering per molecule and optionally scaling.

    Args:
        positions: List of (N_atoms, 3) arrays
        scale: If provided, use this scale factor. Otherwise compute from data.

    Returns:
        all_positions: (N_total_atoms, 3) centered position vectors
        scale: The scale factor used
    """
    centered = []
    for pos in positions:
        # Center each molecule by subtracting its mean position
        centered_pos = pos - pos.mean(axis=0, keepdims=True)
        centered.append(centered_pos)

    all_positions = np.vstack(centered)

    # Compute or apply scale factor
    if scale is None:
        # Scale so typical magnitude is ~1
        rms = np.sqrt(np.mean(all_positions ** 2))
        scale = rms if rms > 0 else 1.0

    all_positions = all_positions / scale

    print(f"Position preprocessing: {len(all_positions)} vectors, scale={scale:.4f}")
    return all_positions, scale


def preprocess_forces(forces: list[np.ndarray], scale: float | None = None) -> tuple[np.ndarray, float]:
    """
    Preprocess force vectors by optional scaling (no centering).

    Args:
        forces: List of (N_atoms, 3) arrays
        scale: If provided, use this scale factor. Otherwise compute from data.

    Returns:
        all_forces: (N_total_atoms, 3) force vectors
        scale: The scale factor used
    """
    all_forces = np.vstack(forces)

    if scale is None:
        # Scale by RMS force magnitude
        rms = np.sqrt(np.mean(all_forces ** 2))
        scale = rms if rms > 0 else 1.0

    all_forces = all_forces / scale

    print(f"Force preprocessing: {len(all_forces)} vectors, scale={scale:.4f}")
    return all_forces, scale


def preprocess_energies(energies: np.ndarray, scale: float | None = None, shift: float | None = None) -> tuple[np.ndarray, float, float]:
    """
    Preprocess energy values by shifting and scaling.

    Args:
        energies: (N_molecules,) array
        scale: If provided, use this scale factor
        shift: If provided, use this shift value

    Returns:
        energies: Preprocessed energies
        scale: The scale factor used
        shift: The shift value used
    """
    if shift is None:
        shift = energies.mean()

    energies = energies - shift

    if scale is None:
        rms = np.sqrt(np.mean(energies ** 2))
        scale = rms if rms > 0 else 1.0

    energies = energies / scale

    print(f"Energy preprocessing: {len(energies)} values, shift={shift:.4f}, scale={scale:.4f}")
    return energies, scale, shift


def _morton_code(x: np.ndarray, y: np.ndarray, z: np.ndarray, bits: int = 10) -> np.ndarray:
    """
    Compute Morton (Z-order) codes for 3D integer coordinates.

    Interleaves bits of x, y, z to create a 1D index that preserves spatial locality.
    """
    def spread_bits(v):
        # Spread bits of a 10-bit integer to every 3rd bit position
        v = v.astype(np.uint64)
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v

    return spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2)


def build_rvq_codebook_3d(
    vectors: np.ndarray,
    n_levels: int,
    codebook_size: int,
    random_state: int = 42,
    sample_size: int | None = None,
) -> list[np.ndarray]:
    """
    Build RVQ codebooks for 3D vectors using Morton curve + quantile binning.

    Maps 3D points to 1D via Morton (Z-order) curve to preserve spatial locality,
    then selects codebook entries at quantile positions along this curve.

    Args:
        vectors: (N, 3) array of vectors
        n_levels: Number of RVQ levels
        codebook_size: Number of entries per codebook (K)
        random_state: Random seed for reproducibility (used for sampling)
        sample_size: If provided, subsample vectors for faster training

    Returns:
        codebooks: List of L arrays, each of shape (K, 3)
    """
    if sample_size is not None and len(vectors) > sample_size:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(vectors), sample_size, replace=False)
        vectors = vectors[indices]

    codebooks = []
    residuals = vectors.copy()

    # Quantile indices for selecting codebook entries
    quantile_fracs = (np.arange(codebook_size) + 0.5) / codebook_size

    # Quantile fracs for K-1 entries (reserving one for zero bucket)
    quantile_fracs_km1 = (np.arange(codebook_size - 1) + 0.5) / (codebook_size - 1)

    for level in range(n_levels):
        print(f"  Level {level + 1}/{n_levels}: computing Morton-quantile codebook with K={codebook_size}...")

        # Filter out points with very small residuals (norm < 0.0001)
        residual_norms = np.linalg.norm(residuals, axis=1)
        significant_mask = residual_norms >= 0.0001
        significant_residuals = residuals[significant_mask]
        n_significant = significant_mask.sum()

        # If too few significant residuals, fall back to standard quantiles on all data
        if len(significant_residuals) < codebook_size - 1:
            print(f"    Note: Only {n_significant} points with residual >= 0.0001, using standard quantiles")
            # Normalize residuals to [0, 1] range for Morton encoding
            mins = residuals.min(axis=0)
            maxs = residuals.max(axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0
            normalized = (residuals - mins) / ranges
            discretized = (normalized * 1023).astype(np.int32).clip(0, 1023)
            morton_codes = _morton_code(discretized[:, 0], discretized[:, 1], discretized[:, 2])
            sorted_indices = np.argsort(morton_codes)
            quantile_indices = (quantile_fracs * (len(residuals) - 1)).astype(int)
            codebook = residuals[sorted_indices[quantile_indices]]
        else:
            # Reserve bucket 0 for near-zero residuals (centroid = [0,0,0])
            # Use K-1 quantile entries for significant residuals
            print(f"    Using zero bucket + {codebook_size - 1} quantile entries ({n_significant} significant points)")

            # Normalize significant residuals to [0, 1] range for Morton encoding
            mins = significant_residuals.min(axis=0)
            maxs = significant_residuals.max(axis=0)
            ranges = maxs - mins
            ranges[ranges == 0] = 1.0
            normalized = (significant_residuals - mins) / ranges

            # Discretize to 10-bit integers (0-1023)
            discretized = (normalized * 1023).astype(np.int32).clip(0, 1023)

            # Compute Morton codes and sort
            morton_codes = _morton_code(discretized[:, 0], discretized[:, 1], discretized[:, 2])
            sorted_indices = np.argsort(morton_codes)

            # Select K-1 codebook entries at quantile positions along Morton curve
            quantile_indices = (quantile_fracs_km1 * (len(significant_residuals) - 1)).astype(int)
            significant_codebook = significant_residuals[sorted_indices[quantile_indices]]

            # Prepend zero vector as bucket 0
            codebook = np.vstack([np.zeros((1, 3)), significant_codebook])

        codebooks.append(codebook)

        # Assign each vector to nearest codebook entry and compute residuals
        dists = np.linalg.norm(residuals[:, None, :] - codebook[None, :, :], axis=2)
        assignments = np.argmin(dists, axis=1)
        residuals = residuals - codebook[assignments]

        residual_rms = np.sqrt(np.mean(residuals ** 2))
        print(f"    Residual RMS: {residual_rms:.6f}")

    return codebooks


def build_rvq_codebook_3d_kmeans(
    vectors: np.ndarray,
    n_levels: int,
    codebook_size: int,
    random_state: int = 42,
    sample_size: int | None = None,
) -> list[np.ndarray]:
    """
    Build RVQ codebooks for 3D vectors using k-means clustering.

    Args:
        vectors: (N, 3) array of vectors
        n_levels: Number of RVQ levels
        codebook_size: Number of entries per codebook (K)
        random_state: Random seed for reproducibility
        sample_size: If provided, subsample vectors for faster training

    Returns:
        codebooks: List of L arrays, each of shape (K, 3)
    """
    if sample_size is not None and len(vectors) > sample_size:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(vectors), sample_size, replace=False)
        vectors = vectors[indices]

    codebooks = []
    residuals = vectors.copy()

    for level in range(n_levels):
        print(f"  Level {level + 1}/{n_levels}: computing k-means codebook with K={codebook_size}...")

        # Fit k-means on residuals
        kmeans = KMeans(
            n_clusters=codebook_size,
            random_state=random_state + level,
            n_init=10,
        )
        kmeans.fit(residuals)

        codebook = kmeans.cluster_centers_
        codebooks.append(codebook)

        # Assign each vector to nearest codebook entry and compute residuals
        dists = np.linalg.norm(residuals[:, None, :] - codebook[None, :, :], axis=2)
        assignments = np.argmin(dists, axis=1)
        residuals = residuals - codebook[assignments]

        residual_rms = np.sqrt(np.mean(residuals ** 2))
        print(f"    Residual RMS: {residual_rms:.6f}")

    return codebooks


def build_rvq_codebook_1d(
    values: np.ndarray,
    n_levels: int,
    codebook_size: int,
    random_state: int = 42,
    sample_size: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Build RVQ codebooks for 1D scalar values (force components, energies) using quantile binning.

    Uses boundary-based assignment: determines which quantile bucket a value falls into
    using boundaries, then assigns the centroid of that bucket.

    Args:
        values: (N,) array of scalar values
        n_levels: Number of RVQ levels
        codebook_size: Number of entries per codebook (K)
        random_state: Random seed for reproducibility (used for sampling)
        sample_size: If provided, subsample values for faster training

    Returns:
        codebooks: List of L arrays, each of shape (K,) - centroids for each bin
        boundaries: List of L arrays, each of shape (K-1,) - boundaries between bins
    """
    if sample_size is not None and len(values) > sample_size:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(values), sample_size, replace=False)
        values = values[indices]

    codebooks = []
    all_boundaries = []
    residuals = values.copy()

    # Percentiles for bin centers (centroids): 0.5/K, 1.5/K, ..., (K-0.5)/K
    centroid_percentiles = (np.arange(codebook_size) + 0.5) / codebook_size * 100
    # Percentiles for K-1 entries (reserving one for zero bucket)
    centroid_percentiles_km1 = (np.arange(codebook_size - 1) + 0.5) / (codebook_size - 1) * 100

    threshold = 0.0001

    for level in range(n_levels):
        print(f"  Level {level + 1}/{n_levels}: computing quantile codebook with K={codebook_size}...")

        # Filter out points with very small residuals (|residual| < threshold)
        significant_mask = np.abs(residuals) >= threshold
        significant_residuals = residuals[significant_mask]
        n_significant = significant_mask.sum()

        # If too few significant residuals, fall back to standard quantiles on all data
        if len(significant_residuals) < codebook_size - 1:
            print(f"    Note: Only {n_significant} points with |residual| >= {threshold}, using standard quantiles")
            codebook = np.percentile(residuals, centroid_percentiles)
            boundaries = (codebook[:-1] + codebook[1:]) / 2
        else:
            # Reserve one bucket for near-zero residuals (centroid = 0)
            # Use K-1 quantile entries for significant residuals
            print(f"    Using zero bucket + {codebook_size - 1} quantile entries ({n_significant} significant points)")

            # Compute K-1 quantile centroids for significant residuals
            significant_centroids = np.percentile(significant_residuals, centroid_percentiles_km1)

            # Insert zero centroid and sort
            codebook = np.sort(np.concatenate([[0.0], significant_centroids]))

            # Compute boundaries as midpoints between adjacent centroids
            boundaries = (codebook[:-1] + codebook[1:]) / 2

        codebooks.append(codebook)
        all_boundaries.append(boundaries)

        # Assign using boundaries: find which bin each value falls into
        assignments = np.searchsorted(boundaries, residuals)
        residuals = residuals - codebook[assignments]

        residual_rms = np.sqrt(np.mean(residuals ** 2))
        print(f"    Residual RMS: {residual_rms:.6f}")

    return codebooks, all_boundaries


def build_rvq_codebook_1d_kmeans(
    values: np.ndarray,
    n_levels: int,
    codebook_size: int,
    random_state: int = 42,
    sample_size: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Build RVQ codebooks for 1D scalar values using k-means clustering.

    Args:
        values: (N,) array of scalar values
        n_levels: Number of RVQ levels
        codebook_size: Number of entries per codebook (K)
        random_state: Random seed for reproducibility
        sample_size: If provided, subsample values for faster training

    Returns:
        codebooks: List of L arrays, each of shape (K,) - centroids for each bin
        boundaries: List of L arrays, each of shape (K-1,) - boundaries between bins
    """
    if sample_size is not None and len(values) > sample_size:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(values), sample_size, replace=False)
        values = values[indices]

    codebooks = []
    all_boundaries = []
    residuals = values.copy()

    for level in range(n_levels):
        print(f"  Level {level + 1}/{n_levels}: computing k-means codebook with K={codebook_size}...")

        # Fit k-means on residuals (reshape to 2D for sklearn)
        kmeans = KMeans(
            n_clusters=codebook_size,
            random_state=random_state + level,
            n_init=10,
        )
        kmeans.fit(residuals.reshape(-1, 1))

        # Get sorted centroids (1D)
        codebook = np.sort(kmeans.cluster_centers_.flatten())
        codebooks.append(codebook)

        # Compute boundaries as midpoints between adjacent centroids
        boundaries = (codebook[:-1] + codebook[1:]) / 2
        all_boundaries.append(boundaries)

        # Assign using boundaries and compute residuals
        assignments = np.searchsorted(boundaries, residuals)
        residuals = residuals - codebook[assignments]

        residual_rms = np.sqrt(np.mean(residuals ** 2))
        print(f"    Residual RMS: {residual_rms:.6f}")

    return codebooks, all_boundaries


def encode_positions(positions: np.ndarray, codebooks: list[np.ndarray]) -> np.ndarray:
    """
    Encode positions using RVQ codebooks.

    Args:
        positions: (N, 3) centered and scaled position vectors
        codebooks: List of L codebooks, each (K, 3)

    Returns:
        codes: (N, L) array of codebook indices
    """
    n_levels = len(codebooks)
    codes = np.zeros((len(positions), n_levels), dtype=np.int32)
    residuals = positions.copy()

    for level, codebook in enumerate(codebooks):
        # Find nearest centroid
        dists = np.linalg.norm(residuals[:, None, :] - codebook[None, :, :], axis=2)
        codes[:, level] = np.argmin(dists, axis=1)
        residuals = residuals - codebook[codes[:, level]]

    return codes


def encode_1d(values: np.ndarray, codebooks: list[np.ndarray], boundaries: list[np.ndarray]) -> np.ndarray:
    """
    Encode 1D values using RVQ codebooks with boundary-based assignment.

    Args:
        values: (N,) scalar values
        codebooks: List of L codebooks, each (K,) - centroids for each bin
        boundaries: List of L arrays, each (K-1,) - boundaries between bins

    Returns:
        codes: (N, L) array of codebook indices
    """
    n_levels = len(codebooks)
    codes = np.zeros((len(values), n_levels), dtype=np.int32)
    residuals = values.copy()

    for level, (codebook, level_boundaries) in enumerate(zip(codebooks, boundaries)):
        # Assign using boundaries: find which bin each value falls into
        codes[:, level] = np.searchsorted(level_boundaries, residuals)
        residuals = residuals - codebook[codes[:, level]]

    return codes


def decode_positions(codes: np.ndarray, codebooks: list[np.ndarray]) -> np.ndarray:
    """
    Decode positions from RVQ codes.

    Args:
        codes: (N, L) array of codebook indices
        codebooks: List of L codebooks, each (K, 3)

    Returns:
        positions: (N, 3) reconstructed position vectors
    """
    positions = np.zeros((len(codes), 3))
    for level, codebook in enumerate(codebooks):
        positions += codebook[codes[:, level]]
    return positions


def decode_1d(codes: np.ndarray, codebooks: list[np.ndarray]) -> np.ndarray:
    """
    Decode 1D values from RVQ codes.

    Args:
        codes: (N, L) array of codebook indices
        codebooks: List of L codebooks, each (K,)

    Returns:
        values: (N,) reconstructed scalar values
    """
    values = np.zeros(len(codes))
    for level, codebook in enumerate(codebooks):
        values += codebook[codes[:, level]]
    return values


def compute_reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """Compute reconstruction error statistics."""
    error = original - reconstructed
    abs_error = np.abs(error)
    return {
        "mae": np.mean(abs_error),
        "median_error": np.median(abs_error),
        "rmse": np.sqrt(np.mean(error ** 2)),
        "max_error": np.max(abs_error),
    }


def main():
    parser = argparse.ArgumentParser(description="Build RVQ codebooks for molecular data")
    parser.add_argument("input_csv", type=str, help="Path to input CSV file")
    parser.add_argument("--output", "-o", type=str, default="rvq_codebooks.pkl", help="Output pickle file")
    parser.add_argument("--n-levels", "-L", type=int, default=4, help="Number of RVQ levels for forces/energy")
    parser.add_argument("--pos-levels", type=int, default=8, help="Number of RVQ levels for positions (3D)")
    parser.add_argument("--codebook-size", "-K", type=int, default=256, help="Codebook size per level")
    parser.add_argument("--train-rows", type=int, default=None, help="Number of rows for training")
    parser.add_argument("--val-rows", type=int, default=None, help="Number of rows for validation")
    parser.add_argument("--sample-size", type=int, default=500000, help="Sample size for codebook training")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--method", type=str, default="quantile", choices=["quantile", "kmeans"],
                        help="Codebook construction method: 'quantile' (fast, deterministic) or 'kmeans' (slower, potentially better)")
    args = parser.parse_args()

    # Determine total rows to load
    max_rows = None
    if args.train_rows is not None:
        max_rows = args.train_rows + (args.val_rows or 0)

    # Load data
    positions, forces, energies = load_data(args.input_csv, max_rows)

    # Split into train/val
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

    positions, forces, energies = train_positions, train_forces, train_energies

    # Preprocess
    print("\nPreprocessing positions...")
    all_positions, pos_scale = preprocess_positions(positions)

    print("\nPreprocessing forces...")
    all_forces, force_scale = preprocess_forces(forces)

    print("\nPreprocessing energies...")
    all_energies, energy_scale, energy_shift = preprocess_energies(energies)

    # Build codebooks
    print(f"\nBuilding codebooks (method: {args.method})...")
    print(f"  - Position: 3D RVQ, L={args.pos_levels}, K={args.codebook_size}")
    print(f"  - Forces (x,y,z): 1D RVQ, L={args.n_levels}, K={args.codebook_size}")
    print(f"  - Energy: 1D RVQ, L={args.n_levels}, K={args.codebook_size}")

    # Select codebook building functions based on method
    if args.method == "kmeans":
        build_3d = build_rvq_codebook_3d_kmeans
        build_1d = build_rvq_codebook_1d_kmeans
    else:
        build_3d = build_rvq_codebook_3d
        build_1d = build_rvq_codebook_1d

    print("\nBuilding position codebooks...")
    pos_codebooks = build_3d(
        all_positions,
        n_levels=args.pos_levels,
        codebook_size=args.codebook_size,
        random_state=args.random_state,
        sample_size=args.sample_size,
    )

    print("\nBuilding force_x codebooks...")
    force_x_cb, force_x_bounds = build_1d(
        all_forces[:, 0].copy(),
        n_levels=args.n_levels,
        codebook_size=args.codebook_size,
        random_state=args.random_state,
        sample_size=args.sample_size,
    )

    print("\nBuilding force_y codebooks...")
    force_y_cb, force_y_bounds = build_1d(
        all_forces[:, 1].copy(),
        n_levels=args.n_levels,
        codebook_size=args.codebook_size,
        random_state=args.random_state + 100,
        sample_size=args.sample_size,
    )

    print("\nBuilding force_z codebooks...")
    force_z_cb, force_z_bounds = build_1d(
        all_forces[:, 2].copy(),
        n_levels=args.n_levels,
        codebook_size=args.codebook_size,
        random_state=args.random_state + 200,
        sample_size=args.sample_size,
    )

    print("\nBuilding energy codebooks...")
    energy_codebooks, energy_boundaries = build_1d(
        all_energies,
        n_levels=args.n_levels,
        codebook_size=args.codebook_size,
        random_state=args.random_state + 300,
        sample_size=min(args.sample_size, len(all_energies)) if args.sample_size else None,
    )

    force_codebooks = {"x": force_x_cb, "y": force_y_cb, "z": force_z_cb}
    force_boundaries = {"x": force_x_bounds, "y": force_y_bounds, "z": force_z_bounds}
    print("\nAll codebooks built successfully.")

    # Evaluate reconstruction quality
    def evaluate_reconstruction(positions_list, forces_list, energies_arr, label: str):
        """Evaluate reconstruction on a dataset."""
        print(f"\n{label} Reconstruction Quality")
        print("-" * 40)

        # Preprocess using training scales
        pos_data, _ = preprocess_positions(positions_list, scale=pos_scale)
        force_data, _ = preprocess_forces(forces_list, scale=force_scale)
        energy_data = (energies_arr - energy_shift) / energy_scale

        # Position reconstruction
        pos_codes = encode_positions(pos_data, pos_codebooks)
        pos_reconstructed = decode_positions(pos_codes, pos_codebooks)
        pos_err = compute_reconstruction_error(pos_data, pos_reconstructed)
        print(f"  Position - MAE: {pos_err['mae']:.6f}, Median: {pos_err['median_error']:.6f}, RMSE: {pos_err['rmse']:.6f}, Max: {pos_err['max_error']:.6f}")

        # Force reconstruction
        force_reconstructed = np.zeros_like(force_data)
        for i, component in enumerate(["x", "y", "z"]):
            codes = encode_1d(force_data[:, i], force_codebooks[component], force_boundaries[component])
            force_reconstructed[:, i] = decode_1d(codes, force_codebooks[component])
        force_err = compute_reconstruction_error(force_data, force_reconstructed)
        print(f"  Force    - MAE: {force_err['mae']:.6f}, Median: {force_err['median_error']:.6f}, RMSE: {force_err['rmse']:.6f}, Max: {force_err['max_error']:.6f}")

        # Energy reconstruction
        energy_codes = encode_1d(energy_data, energy_codebooks, energy_boundaries)
        energy_reconstructed = decode_1d(energy_codes, energy_codebooks)
        energy_err = compute_reconstruction_error(energy_data, energy_reconstructed)
        print(f"  Energy   - MAE: {energy_err['mae']:.6f}, Median: {energy_err['median_error']:.6f}, RMSE: {energy_err['rmse']:.6f}, Max: {energy_err['max_error']:.6f}")

        return {"position": pos_err, "force": force_err, "energy": energy_err}

    # Save codebooks before evaluation
    result = {
        "config": {
            "pos_levels": args.pos_levels,
            "n_levels": args.n_levels,
            "codebook_size": args.codebook_size,
            "random_state": args.random_state,
            "method": args.method,
        },
        "preprocessing": {
            "position_scale": pos_scale,
            "force_scale": force_scale,
            "energy_scale": energy_scale,
            "energy_shift": energy_shift,
        },
        "codebooks": {
            "position": pos_codebooks,  # List of (K, 3) arrays
            "force_x": force_codebooks["x"],  # List of (K,) arrays
            "force_y": force_codebooks["y"],
            "force_z": force_codebooks["z"],
            "energy": energy_codebooks,  # List of (K,) arrays
        },
        "boundaries": {
            "force_x": force_boundaries["x"],  # List of (K-1,) arrays
            "force_y": force_boundaries["y"],
            "force_z": force_boundaries["z"],
            "energy": energy_boundaries,  # List of (K-1,) arrays
        },
    }

    output_path = Path(args.output)
    with open(output_path, "wb") as f:
        pickle.dump(result, f)

    print(f"\nCodebooks saved to {output_path}")
    print(f"Position: {args.codebook_size} entries x {args.pos_levels} levels = {args.codebook_size * args.pos_levels} codes")
    print(f"Forces/Energy: {args.codebook_size} entries x {args.n_levels} levels = {args.codebook_size * args.n_levels} codes each")

    print("\n" + "=" * 60)
    print("Reconstruction Quality Evaluation")
    print("=" * 60)

    # Evaluate on training data
    train_errors = evaluate_reconstruction(positions, forces, energies, "TRAIN")

    # Evaluate on validation data if available
    if val_positions is not None:
        evaluate_reconstruction(val_positions, val_forces, val_energies, "VALIDATION")


if __name__ == "__main__":
    main()

"""
Serialize molecular data to token sequences using FP16-like encoding.

Each dimension has its own unique token set, eliminating positional ambiguity:
- Position X/Y/Z: [SignExp] [Mant0] [Mant1] (3 tokens each, 1024 unique per dim)
- Force X/Y/Z: [SignExp] [Mant0] [Mant1] (3 tokens each, 1024 unique per dim)
- Energy: [Exp] [Mant0] [Mant1] [Mant2] (4 tokens, 1024 unique, no sign - always negative)

Sequence format (all positions, then all forces, then energy):

[BOS]
[ATOMS] [Z=6] [Z=1] ... [ATOMS_END]
[PosX_SE:+128][PosX_M0:100][PosX_M1:50][PosY:...][PosZ:...][NL]  # atom 0 position
[PosX_SE:-130][PosX_M0:200][PosX_M1:10][PosY:...][PosZ:...][NL]  # atom 1 position
...
[FrcX_SE:+140][FrcX_M0:80][FrcX_M1:60][FrcY:...][FrcZ:...][NL]   # atom 0 force
[FrcX_SE:-135][FrcX_M0:90][FrcX_M1:40][FrcY:...][FrcZ:...][NL]   # atom 1 force
...
[Eng_E:200][Eng_M0:150][Eng_M1:75][Eng_M2:30]
[EOS]

Total vocabulary: 7292 tokens
- 25% fewer tokens per atom (3 vs 4 for position/force values)
- Same precision: 16.7M bins for signed dims, 4B bins for energy
"""

import argparse
import ast
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# Token Layout
# =============================================================================

# Special tokens (minimal set)
SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[BOS]": 1,
    "[EOS]": 2,
    "\n": 3,  # Newline token - separates atoms
    "[ATOMS]": 4,
    "[ATOMS_END]": 5,
}
NUM_SPECIAL_TOKENS = len(SPECIAL_TOKENS)  # 6

# Atomic number tokens: [Z=1] through [Z=118]
ATOM_TOKEN_OFFSET = NUM_SPECIAL_TOKENS  # 6
MAX_ATOMIC_NUMBER = 118
NUM_ATOM_TOKENS = MAX_ATOMIC_NUMBER  # 118

# Dimension token layout (compact 3-token encoding for signed, 4-token for energy)
# Each signed dimension (pos_x/y/z, force_x/y/z): SignExp(512) + M0(256) + M1(256) = 1024
# Energy (unsigned): Exp(256) + M0(256) + M1(256) + M2(256) = 1024 (4 tokens for full precision)
SIGNED_DIM_TOKENS = 512 + 256 + 256  # 1024
UNSIGNED_DIM_TOKENS = 256 + 256 + 256 + 256  # 1024

# Dimension token offsets
DIM_TOKEN_START = ATOM_TOKEN_OFFSET + NUM_ATOM_TOKENS  # 124

# Define offsets for each dimension
POS_X_START = DIM_TOKEN_START  # 124
POS_Y_START = POS_X_START + SIGNED_DIM_TOKENS  # 894
POS_Z_START = POS_Y_START + SIGNED_DIM_TOKENS  # 1664
FORCE_X_START = POS_Z_START + SIGNED_DIM_TOKENS  # 2434
FORCE_Y_START = FORCE_X_START + SIGNED_DIM_TOKENS  # 3204
FORCE_Z_START = FORCE_Y_START + SIGNED_DIM_TOKENS  # 3974
ENERGY_START = FORCE_Z_START + SIGNED_DIM_TOKENS  # 4744

# Total vocabulary size
VOCAB_SIZE = ENERGY_START + UNSIGNED_DIM_TOKENS  # 5768

# Dimension names for iteration
SIGNED_DIMENSIONS = ["pos_x", "pos_y", "pos_z", "force_x", "force_y", "force_z"]
DIM_STARTS = {
    "pos_x": POS_X_START,
    "pos_y": POS_Y_START,
    "pos_z": POS_Z_START,
    "force_x": FORCE_X_START,
    "force_y": FORCE_Y_START,
    "force_z": FORCE_Z_START,
    "energy": ENERGY_START,
}

# Short prefixes for token names
DIM_PREFIXES = {
    "pos_x": "PosX",
    "pos_y": "PosY",
    "pos_z": "PosZ",
    "force_x": "FrcX",
    "force_y": "FrcY",
    "force_z": "FrcZ",
    "energy": "Eng",
}


@dataclass
class FP16DimensionConfig:
    """Configuration for encoding a single dimension."""
    log_min: float
    log_max: float


def get_signed_token_ids(dim_start: int, sign_positive: bool, exp: int, mant0: int, mant1: int) -> list[int]:
    """Get token IDs for a signed dimension value (3 tokens: SignExp, Mant0, Mant1)."""
    # SignExp: 512 tokens (256 for positive, 256 for negative)
    sign_exp_offset = exp if sign_positive else (256 + exp)
    return [
        dim_start + sign_exp_offset,  # Combined SignExp token
        dim_start + 512 + mant0,  # Mant0 token (after 512 SignExp tokens)
        dim_start + 512 + 256 + mant1,  # Mant1 token
    ]


def get_unsigned_token_ids(dim_start: int, exp: int, mant0: int, mant1: int, mant2: int) -> list[int]:
    """Get token IDs for an unsigned dimension value (energy) - 4 tokens for full precision."""
    return [
        dim_start + exp,  # Exp token
        dim_start + 256 + mant0,  # Mant0 token
        dim_start + 256 + 256 + mant1,  # Mant1 token
        dim_start + 256 + 256 + 256 + mant2,  # Mant2 token
    ]


def encode_signed_batch(values: np.ndarray, config: FP16DimensionConfig, dim_start: int) -> np.ndarray:
    """
    Vectorized encoding of signed values to token IDs.

    Args:
        values: 1D array of values to encode
        config: Dimension configuration with log_min/log_max
        dim_start: Starting token ID for this dimension

    Returns:
        (N, 3) array of token IDs [SignExp, Mant0, Mant1]
    """
    signs_positive = values >= 0
    magnitudes = np.abs(values)

    # Vectorized log transform (clamp to avoid log10(0) warning)
    safe_magnitudes = np.maximum(magnitudes, 1e-15)
    log_vals = np.log10(safe_magnitudes)
    log_vals = np.clip(log_vals, config.log_min, config.log_max)

    # Normalize to [0, 1]
    normalized = (log_vals - config.log_min) / (config.log_max - config.log_min)

    # Quantize to bins
    total_bins = 256 * 65536  # ~16.7M bins
    bin_indices = (normalized * (total_bins - 1)).astype(np.int64)
    bin_indices = np.clip(bin_indices, 0, total_bins - 1)

    # Extract components
    exp_codes = bin_indices // 65536
    mant_totals = bin_indices % 65536
    mant0s = mant_totals // 256
    mant1s = mant_totals % 256

    # SignExp: positive uses exp directly, negative adds 256
    sign_exp_offsets = np.where(signs_positive, exp_codes, 256 + exp_codes)

    return np.column_stack([
        dim_start + sign_exp_offsets,
        dim_start + 512 + mant0s,
        dim_start + 512 + 256 + mant1s,
    ]).astype(np.int32)


def encode_unsigned_batch(values: np.ndarray, config: FP16DimensionConfig, dim_start: int) -> np.ndarray:
    """
    Vectorized encoding of unsigned values (energy) to token IDs.

    Args:
        values: 1D array of values to encode
        config: Dimension configuration with log_min/log_max
        dim_start: Starting token ID for this dimension

    Returns:
        (N, 4) array of token IDs [Exp, Mant0, Mant1, Mant2]
    """
    magnitudes = np.abs(values)

    # Vectorized log transform (clamp to avoid log10(0) warning)
    safe_magnitudes = np.maximum(magnitudes, 1e-15)
    log_vals = np.log10(safe_magnitudes)
    log_vals = np.clip(log_vals, config.log_min, config.log_max)

    # Normalize to [0, 1]
    normalized = (log_vals - config.log_min) / (config.log_max - config.log_min)

    # Quantize to bins (4B bins for energy)
    total_bins = 256 * 256 * 256 * 256
    bin_indices = (normalized * (total_bins - 1)).astype(np.int64)
    bin_indices = np.clip(bin_indices, 0, total_bins - 1)

    # Extract components
    exp_codes = bin_indices // (256 * 256 * 256)
    mant_totals = bin_indices % (256 * 256 * 256)
    mant0s = mant_totals // (256 * 256)
    mant1s = (mant_totals // 256) % 256
    mant2s = mant_totals % 256

    return np.column_stack([
        dim_start + exp_codes,
        dim_start + 256 + mant0s,
        dim_start + 512 + mant1s,
        dim_start + 768 + mant2s,
    ]).astype(np.int32)


def encode_fp16_value(value: float, config: FP16DimensionConfig, dim_start: int, include_sign: bool = True) -> list[int]:
    """
    Encode a floating-point value into FP16-like tokens for a specific dimension.

    Args:
        value: The value to encode
        config: Dimension configuration with log_min/log_max
        dim_start: Starting token ID for this dimension
        include_sign: If True, returns [signexp, mant0, mant1] (3 tokens)
                     If False, returns [exp, mant0, mant1, mant2] (4 tokens for energy)

    Returns:
        List of 3 token IDs (signed) or 4 token IDs (unsigned/energy)
    """
    # Handle sign
    sign_positive = value >= 0
    magnitude = abs(value)

    # Handle zero or very small values
    if magnitude < 1e-15:
        exp_code = 0
        mant_total = 0
        if include_sign:
            return get_signed_token_ids(dim_start, sign_positive, exp_code, 0, 0)
        else:
            return get_unsigned_token_ids(dim_start, exp_code, 0, 0, 0)

    # Map magnitude to log scale
    log_val = np.log10(magnitude)

    # Clamp to range
    log_val = np.clip(log_val, config.log_min, config.log_max)

    # Normalize to [0, 1]
    normalized = (log_val - config.log_min) / (config.log_max - config.log_min)

    if include_sign:
        # Signed values: 256 exp bins, 65536 mantissa bins (3 tokens)
        total_bins = 256 * 65536  # ~16.7M bins
        bin_index = int(normalized * (total_bins - 1))
        bin_index = np.clip(bin_index, 0, total_bins - 1)

        exp_code = bin_index // 65536
        mant_total = bin_index % 65536
        mant0 = mant_total // 256
        mant1 = mant_total % 256
        return get_signed_token_ids(dim_start, sign_positive, exp_code, mant0, mant1)
    else:
        # Unsigned (energy): 256 exp bins, 16M mantissa bins (4 tokens, ~4B total bins)
        total_bins = 256 * 256 * 256 * 256  # ~4B bins
        bin_index = int(normalized * (total_bins - 1))
        bin_index = np.clip(bin_index, 0, total_bins - 1)

        exp_code = bin_index // (256 * 256 * 256)
        mant_total = bin_index % (256 * 256 * 256)
        mant0 = mant_total // (256 * 256)
        mant1 = (mant_total // 256) % 256
        mant2 = mant_total % 256
        return get_unsigned_token_ids(dim_start, exp_code, mant0, mant1, mant2)


def decode_fp16_value(tokens: list[int], config: FP16DimensionConfig, dim_start: int, include_sign: bool = True) -> float:
    """
    Decode FP16-like tokens back to a floating-point value.

    Args:
        tokens: List of 3 token IDs (signed) or 4 token IDs (unsigned/energy)
        config: Dimension configuration with log_min/log_max
        dim_start: Starting token ID for this dimension
        include_sign: If True, expects [signexp, mant0, mant1] (3 tokens)
                     If False, expects [exp, mant0, mant1, mant2] (4 tokens)

    Returns:
        Decoded floating-point value
    """
    if include_sign:
        signexp_token, mant0_token, mant1_token = tokens
        signexp_offset = signexp_token - dim_start
        # First 256 tokens are positive, next 256 are negative
        if signexp_offset < 256:
            sign = 1.0
            exp_code = signexp_offset
        else:
            sign = -1.0
            exp_code = signexp_offset - 256
        mant0 = mant0_token - dim_start - 512
        mant1 = mant1_token - dim_start - 512 - 256
        mant_total = mant0 * 256 + mant1

        # Reconstruct bin index
        bin_index = exp_code * 65536 + mant_total
        total_bins = 256 * 65536
    else:
        exp_token, mant0_token, mant1_token, mant2_token = tokens
        sign = -1.0  # Energy is always negative
        # Cast to Python int to avoid numpy scalar overflow
        exp_code = int(exp_token - dim_start)
        mant0 = int(mant0_token - dim_start - 256)
        mant1 = int(mant1_token - dim_start - 256 - 256)
        mant2 = int(mant2_token - dim_start - 256 - 256 - 256)
        mant_total = mant0 * 256 * 256 + mant1 * 256 + mant2

        # Reconstruct bin index
        bin_index = exp_code * (256 * 256 * 256) + mant_total
        total_bins = 256 * 256 * 256 * 256

    # Handle zero
    if bin_index == 0:
        return 0.0

    # Denormalize to log scale
    normalized = bin_index / (total_bins - 1)
    log_val = normalized * (config.log_max - config.log_min) + config.log_min

    # Convert back to linear scale
    magnitude = 10.0 ** log_val

    return sign * magnitude


def parse_array(s: str) -> np.ndarray:
    """Parse a string representation of a nested list into a numpy array."""
    return np.array(ast.literal_eval(s))


def load_config(path: str) -> dict:
    """Load config from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


class MoleculeTokenizer:
    """Tokenizer for molecular data using FP16-like encoding with unique tokens per dimension."""

    def __init__(self, config_path: str):
        """
        Initialize tokenizer with config.

        Args:
            config_path: Path to JSON config file
        """
        # Check file extension
        if config_path.endswith('.pkl') or config_path.endswith('.pickle'):
            raise ValueError(
                f"Legacy RVQ codebook format detected: {config_path}\n"
                "This tokenizer now uses FP16-like encoding with JSON config.\n"
                "Please generate a new config using:\n"
                "  python build_fp16_config.py <training_data.csv> -o fp16_config.json"
            )

        config = load_config(config_path)

        if config.get("encoding_type") != "fp16_like":
            raise ValueError(f"Unknown encoding type: {config.get('encoding_type')}")

        # Store dimension configs
        dims = config["dimensions"]
        self.dim_configs = {
            "pos_x": FP16DimensionConfig(**dims["pos_x"]),
            "pos_y": FP16DimensionConfig(**dims["pos_y"]),
            "pos_z": FP16DimensionConfig(**dims["pos_z"]),
            "force_x": FP16DimensionConfig(**dims["force_x"]),
            "force_y": FP16DimensionConfig(**dims["force_y"]),
            "force_z": FP16DimensionConfig(**dims["force_z"]),
            "energy": FP16DimensionConfig(**dims["energy"]),
        }

        self.vocab_size = VOCAB_SIZE

        # Precompute token ID -> string lookup table for fast serialization
        self._token_strings: list[str] = self._build_token_lookup()

    def _build_token_lookup(self) -> list[str]:
        """Build lookup table mapping token ID -> string representation."""
        lookup = [""] * self.vocab_size

        # Special tokens
        for name, tid in SPECIAL_TOKENS.items():
            lookup[tid] = "[NL]" if name == "\n" else name

        # Atom tokens
        for z in range(1, MAX_ATOMIC_NUMBER + 1):
            lookup[ATOM_TOKEN_OFFSET + z - 1] = f"[Z={z}]"

        # Dimension tokens (signed: SignExp combined)
        for dim_name in SIGNED_DIMENSIONS:
            prefix = DIM_PREFIXES[dim_name]
            start = DIM_STARTS[dim_name]

            # SignExp tokens (512: 256 positive + 256 negative)
            for i in range(256):
                lookup[start + i] = f"[{prefix}_SE:+{i}]"
            for i in range(256):
                lookup[start + 256 + i] = f"[{prefix}_SE:-{i}]"

            # Mant0 tokens (256)
            for i in range(256):
                lookup[start + 512 + i] = f"[{prefix}_M0:{i}]"

            # Mant1 tokens (256)
            for i in range(256):
                lookup[start + 512 + 256 + i] = f"[{prefix}_M1:{i}]"

        # Energy tokens (no sign, 4 tokens for full precision)
        prefix = DIM_PREFIXES["energy"]
        start = DIM_STARTS["energy"]

        # Exp tokens (256)
        for i in range(256):
            lookup[start + i] = f"[{prefix}_E:{i}]"

        # Mant0 tokens (256)
        for i in range(256):
            lookup[start + 256 + i] = f"[{prefix}_M0:{i}]"

        # Mant1 tokens (256)
        for i in range(256):
            lookup[start + 256 + 256 + i] = f"[{prefix}_M1:{i}]"

        # Mant2 tokens (256)
        for i in range(256):
            lookup[start + 256 + 256 + 256 + i] = f"[{prefix}_M2:{i}]"

        return lookup

    def get_vocab_info(self) -> dict:
        """Return vocabulary information."""
        return {
            "vocab_size": self.vocab_size,
            "special_tokens": SPECIAL_TOKENS,
            "atom_token_offset": ATOM_TOKEN_OFFSET,
            "max_atomic_number": MAX_ATOMIC_NUMBER,
            "dim_starts": DIM_STARTS,
            "encoding_type": "fp16_like_per_dimension",
        }

    def encode_molecule(
        self,
        atomic_numbers: list[int],
        positions: np.ndarray,
        forces: np.ndarray,
        energy: float,
        shuffle_sections: bool = False,
        rng: np.random.Generator | None = None,
    ) -> list[int]:
        """
        Encode a single molecule to token sequence.

        Args:
            atomic_numbers: List of atomic numbers for each atom
            positions: (N_atoms, 3) array of positions
            forces: (N_atoms, 3) array of forces
            energy: Scalar energy value
            shuffle_sections: If True, randomly shuffle atom order (not implemented yet)
            rng: Random number generator for shuffling

        Returns:
            tokens: List of token IDs
        """
        n_atoms = len(atomic_numbers)
        NL = SPECIAL_TOKENS["\n"]

        # Center positions
        positions_centered = positions - positions.mean(axis=0, keepdims=True)

        # Batch encode all position dimensions -> (N, 3) each
        pos_x = encode_signed_batch(positions_centered[:, 0], self.dim_configs["pos_x"], POS_X_START)
        pos_y = encode_signed_batch(positions_centered[:, 1], self.dim_configs["pos_y"], POS_Y_START)
        pos_z = encode_signed_batch(positions_centered[:, 2], self.dim_configs["pos_z"], POS_Z_START)

        # Batch encode all force dimensions -> (N, 3) each
        frc_x = encode_signed_batch(forces[:, 0], self.dim_configs["force_x"], FORCE_X_START)
        frc_y = encode_signed_batch(forces[:, 1], self.dim_configs["force_y"], FORCE_Y_START)
        frc_z = encode_signed_batch(forces[:, 2], self.dim_configs["force_z"], FORCE_Z_START)

        # Build position lines: [PosX(3), PosY(3), PosZ(3), NL] per atom -> (N, 10)
        nl_col = np.full((n_atoms, 1), NL, dtype=np.int32)
        pos_lines = np.hstack([pos_x, pos_y, pos_z, nl_col])  # (N, 10)

        # Build force lines: [FrcX(3), FrcY(3), FrcZ(3), NL] per atom -> (N, 10)
        frc_lines = np.hstack([frc_x, frc_y, frc_z, nl_col])  # (N, 10)

        # Energy tokens (single value -> (4,) array)
        energy_tokens = encode_unsigned_batch(np.array([energy]), self.dim_configs["energy"], ENERGY_START)[0]

        # Pre-allocate full token array
        # Layout: BOS(1) + ATOMS(1) + atoms(n) + ATOMS_END(1) + pos_lines(n*10) + frc_lines(n*10) + energy(4) + EOS(1)
        total_len = 3 + n_atoms + n_atoms * 10 + n_atoms * 10 + 4 + 1  # = 8 + 21*n_atoms
        tokens = np.empty(total_len, dtype=np.int32)

        # Fill header
        tokens[0] = SPECIAL_TOKENS["[BOS]"]
        tokens[1] = SPECIAL_TOKENS["[ATOMS]"]

        # Atom tokens
        idx = 2
        for atomic_num in atomic_numbers:
            tokens[idx] = ATOM_TOKEN_OFFSET + atomic_num - 1
            idx += 1
        tokens[idx] = SPECIAL_TOKENS["[ATOMS_END]"]
        idx += 1

        # Position lines (flatten)
        pos_end = idx + n_atoms * 10
        tokens[idx:pos_end] = pos_lines.ravel()
        idx = pos_end

        # Force lines (flatten)
        frc_end = idx + n_atoms * 10
        tokens[idx:frc_end] = frc_lines.ravel()
        idx = frc_end

        # Energy tokens
        tokens[idx:idx + 4] = energy_tokens
        idx += 4

        # EOS
        tokens[idx] = SPECIAL_TOKENS["[EOS]"]

        return tokens.tolist()

    def tokens_to_string(self, tokens: list[int], pretty: bool = False) -> str:
        """
        Convert token IDs to human-readable string.

        Args:
            tokens: List of token IDs
            pretty: If True, render newline tokens as actual newlines for display

        Returns:
            String representation of tokens
        """
        lookup = self._token_strings
        vocab_size = self.vocab_size

        if pretty:
            parts = []
            for tok in tokens:
                if 0 <= tok < vocab_size:
                    s = lookup[tok]
                    parts.append("\n" if s == "[NL]" else s)
                else:
                    parts.append(f"[UNK:{tok}]")
            return " ".join(parts)

        return " ".join(
            lookup[tok] if 0 <= tok < vocab_size else f"[UNK:{tok}]"
            for tok in tokens
        )

    def decode_molecule(self, tokens: list[int]) -> dict:
        """
        Decode token sequence back to molecular data.

        Args:
            tokens: List of token IDs

        Returns:
            Dictionary with:
                - atomic_numbers: list of atomic numbers
                - positions: (N, 3) array (centered)
                - forces: (N, 3) array
                - energy: scalar
        """
        atomic_numbers = []
        positions = []
        forces = []
        energy = 0.0

        # State machine for parsing
        in_atoms = False
        current_line_tokens = []

        i = 0
        while i < len(tokens):
            tid = tokens[i]

            # Skip BOS/EOS
            if tid == SPECIAL_TOKENS["[BOS]"] or tid == SPECIAL_TOKENS["[EOS]"]:
                i += 1
                continue

            # Atoms section
            if tid == SPECIAL_TOKENS["[ATOMS]"]:
                in_atoms = True
                i += 1
                continue
            if tid == SPECIAL_TOKENS["[ATOMS_END]"]:
                in_atoms = False
                i += 1
                continue

            if in_atoms:
                atomic_num = tid - ATOM_TOKEN_OFFSET + 1
                if 1 <= atomic_num <= MAX_ATOMIC_NUMBER:
                    atomic_numbers.append(atomic_num)
                i += 1
                continue

            # Newline - process accumulated line tokens
            if tid == SPECIAL_TOKENS["\n"]:
                if len(current_line_tokens) >= 9:  # 3 dims * 3 tokens each
                    # Check first token to determine if position or force
                    first_tok = current_line_tokens[0]
                    if POS_X_START <= first_tok < POS_X_START + SIGNED_DIM_TOKENS:
                        # Position line
                        pos_x = decode_fp16_value(current_line_tokens[0:3], self.dim_configs["pos_x"], POS_X_START, include_sign=True)
                        pos_y = decode_fp16_value(current_line_tokens[3:6], self.dim_configs["pos_y"], POS_Y_START, include_sign=True)
                        pos_z = decode_fp16_value(current_line_tokens[6:9], self.dim_configs["pos_z"], POS_Z_START, include_sign=True)
                        positions.append([pos_x, pos_y, pos_z])
                    elif FORCE_X_START <= first_tok < FORCE_X_START + SIGNED_DIM_TOKENS:
                        # Force line
                        frc_x = decode_fp16_value(current_line_tokens[0:3], self.dim_configs["force_x"], FORCE_X_START, include_sign=True)
                        frc_y = decode_fp16_value(current_line_tokens[3:6], self.dim_configs["force_y"], FORCE_Y_START, include_sign=True)
                        frc_z = decode_fp16_value(current_line_tokens[6:9], self.dim_configs["force_z"], FORCE_Z_START, include_sign=True)
                        forces.append([frc_x, frc_y, frc_z])

                current_line_tokens = []
                i += 1
                continue

            # Check if this is an energy token (comes after all atoms)
            if ENERGY_START <= tid < ENERGY_START + UNSIGNED_DIM_TOKENS:
                # Collect 4 energy tokens
                if i + 3 < len(tokens):
                    energy_tokens = tokens[i:i+4]
                    energy = decode_fp16_value(energy_tokens, self.dim_configs["energy"], ENERGY_START, include_sign=False)
                    i += 4
                    continue

            # Otherwise, accumulate as line tokens
            current_line_tokens.append(tid)
            i += 1

        return {
            "atomic_numbers": atomic_numbers,
            "positions": np.array(positions) if positions else np.zeros((0, 3)),
            "forces": np.array(forces) if forces else np.zeros((0, 3)),
            "energy": energy,
        }

    def build_vocab(self) -> dict[str, int]:
        """
        Build vocabulary mapping token strings to IDs.

        Returns:
            Dictionary mapping token strings to token IDs
        """
        vocab = {}

        # Special tokens
        for name, tid in SPECIAL_TOKENS.items():
            if name == "\n":
                vocab["[NL]"] = tid
            else:
                vocab[name] = tid

        # Atom tokens
        for z in range(1, MAX_ATOMIC_NUMBER + 1):
            vocab[f"[Z={z}]"] = ATOM_TOKEN_OFFSET + z - 1

        # Dimension tokens (signed: combined SignExp)
        for dim_name in SIGNED_DIMENSIONS:
            prefix = DIM_PREFIXES[dim_name]
            start = DIM_STARTS[dim_name]

            # SignExp tokens (512: 256 positive + 256 negative)
            for i in range(256):
                vocab[f"[{prefix}_SE:+{i}]"] = start + i
            for i in range(256):
                vocab[f"[{prefix}_SE:-{i}]"] = start + 256 + i

            # Mantissa tokens
            for i in range(256):
                vocab[f"[{prefix}_M0:{i}]"] = start + 512 + i
                vocab[f"[{prefix}_M1:{i}]"] = start + 512 + 256 + i

        # Energy tokens (4 tokens: Exp + M0 + M1 + M2 for full precision)
        prefix = DIM_PREFIXES["energy"]
        start = DIM_STARTS["energy"]
        for i in range(256):
            vocab[f"[{prefix}_E:{i}]"] = start + i
            vocab[f"[{prefix}_M0:{i}]"] = start + 256 + i
            vocab[f"[{prefix}_M1:{i}]"] = start + 256 + 256 + i
            vocab[f"[{prefix}_M2:{i}]"] = start + 256 + 256 + 256 + i

        return vocab

    def get_hf_tokenizer(self):
        """
        Create a HuggingFace PreTrainedTokenizerFast for this vocabulary.

        Returns:
            PreTrainedTokenizerFast instance
        """
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        vocab = self.build_vocab()
        vocab["[UNK]"] = len(vocab)

        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            bos_token="[BOS]",
            eos_token="[EOS]",
        )

        return hf_tokenizer

    def save_hf_tokenizer(self, path: str):
        """Save HuggingFace tokenizer to disk."""
        hf_tokenizer = self.get_hf_tokenizer()
        hf_tokenizer.save_pretrained(path)
        print(f"Saved HuggingFace tokenizer to {path}")


def serialize_csv(
    input_csv: str,
    config_path: str,
    output_path: str,
    max_rows: int | None = None,
    show_examples: int = 0,
):
    """Serialize CSV data to token sequences."""
    tokenizer = MoleculeTokenizer(config_path)

    print("Vocabulary Info:")
    vocab_info = tokenizer.get_vocab_info()
    for key, value in vocab_info.items():
        if key != "dim_starts":
            print(f"  {key}: {value}")

    print(f"\nLoading data from {input_csv}...")
    df = pd.read_csv(input_csv, nrows=max_rows)
    print(f"Loaded {len(df)} molecules")

    all_tokens = []
    token_counts = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
        atomic_numbers = ast.literal_eval(row["atomic_numbers"])
        positions = parse_array(row["positions"])
        forces = parse_array(row["atomic_forces"])
        energy = float(row["energy"])

        tokens = tokenizer.encode_molecule(atomic_numbers, positions, forces, energy)
        all_tokens.append(tokens)
        token_counts.append(len(tokens))

        if show_examples > 0 and idx < show_examples:
            print(f"\n--- Example {idx} ({len(atomic_numbers)} atoms, {len(tokens)} tokens) ---")
            print(f"Atomic numbers: {atomic_numbers[:5]}{'...' if len(atomic_numbers) > 5 else ''}")
            print(f"Token string: {tokenizer.tokens_to_string(tokens[:30])}...")

    token_counts = np.array(token_counts)
    print(f"\nToken Statistics:")
    print(f"  Total molecules: {len(all_tokens)}")
    print(f"  Total tokens: {token_counts.sum():,}")
    print(f"  Tokens per molecule: min={token_counts.min()}, max={token_counts.max()}, "
          f"mean={token_counts.mean():.1f}, median={np.median(token_counts):.1f}")

    output = {
        "vocab_info": vocab_info,
        "tokens": all_tokens,
        "metadata": {
            "input_csv": input_csv,
            "config_path": config_path,
            "n_molecules": len(all_tokens),
        }
    }

    with open(output_path, "wb") as f:
        pickle.dump(output, f)

    print(f"\nSaved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Serialize molecular data to token sequences")
    parser.add_argument("input_csv", type=str, help="Path to input CSV file")
    parser.add_argument("--config", "-c", type=str, default="fp16_config.json",
                        help="Path to FP16 config JSON file")
    parser.add_argument("--output", "-o", type=str, default="tokenized_molecules.pkl",
                        help="Output pickle file")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Maximum rows to process")
    parser.add_argument("--show-examples", type=int, default=3,
                        help="Number of examples to print")
    args = parser.parse_args()

    serialize_csv(
        args.input_csv,
        args.config,
        args.output,
        max_rows=args.max_rows,
        show_examples=args.show_examples,
    )


if __name__ == "__main__":
    main()

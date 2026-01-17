"""
Serialize molecular data to token sequences using RVQ codebooks.

Sequence format with shuffleable sections:

[BOS]
[ATOMS]
  [Z=6] [Z=6] [Z=1] ...            # atom types (no separators)
[ATOMS_END]
[POS]
  [P0] [P1] ... [P7] \n            # position tokens for atom 0
  [P0] [P1] ... [P7] \n            # position tokens for atom 1
  ...
[POS_END]
[FORCE]
  [FX0] ... [FZ3] \n               # force tokens for atom 0
  [FX0] ... [FZ3] \n               # force tokens for atom 1
  ...
[FORCE_END]
[ENERGY]
  [E0] [E1] [E2] [E3] \n           # energy tokens (at end, like graph-free-transformer)
[ENERGY_END]
[EOS]

With shuffle_sections=True, the order of [ATOMS], [POS], [FORCE], [ENERGY]
sections is randomized. This enables flexible conditioning at inference time:
- Atoms + Energy → Positions (inverse design)
- Atoms + Positions → Energy + Forces (forward prediction)
- etc.
"""

import argparse
import ast
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm


# Special token IDs (will be at the start of vocabulary)
SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[BOS]": 1,
    "[EOS]": 2,
    "\n": 3,  # Newline token - prints as actual newline
    "[ATOMS]": 4,
    "[ATOMS_END]": 5,
    "[POS]": 6,
    "[POS_END]": 7,
    "[FORCE]": 8,
    "[FORCE_END]": 9,
    "[ENERGY]": 10,
    "[ENERGY_END]": 11,
}

# Atomic number tokens: [ATOM_1] through [ATOM_118]
# These come after special tokens
ATOM_TOKEN_OFFSET = len(SPECIAL_TOKENS)
MAX_ATOMIC_NUMBER = 118

# Codebook tokens come after atom tokens
CODEBOOK_TOKEN_OFFSET = ATOM_TOKEN_OFFSET + MAX_ATOMIC_NUMBER


def parse_array(s: str) -> np.ndarray:
    """Parse a string representation of a nested list into a numpy array."""
    return np.array(ast.literal_eval(s))


def load_codebook(path: str) -> dict:
    """Load codebook from pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


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
    n_positions = len(positions)
    codes = np.zeros((n_positions, n_levels), dtype=np.int32)
    residuals = positions.copy()

    for level, codebook in enumerate(codebooks):
        # Use squared euclidean (faster, argmin is the same)
        dists = cdist(residuals, codebook, metric="sqeuclidean")
        codes[:, level] = np.argmin(dists, axis=1)
        residuals = residuals - codebook[codes[:, level]]

    return codes


def encode_1d(values: np.ndarray, boundaries: list[np.ndarray]) -> np.ndarray:
    """
    Encode 1D values using boundary-based assignment.

    Args:
        values: (N,) scalar values
        boundaries: List of L arrays, each (K-1,) - boundaries between bins

    Returns:
        codes: (N, L) array of codebook indices
    """
    n_levels = len(boundaries)
    codes = np.zeros((len(values), n_levels), dtype=np.int32)
    residuals = values.copy()

    for level, level_boundaries in enumerate(boundaries):
        codes[:, level] = np.searchsorted(level_boundaries, residuals)
        # We need the codebook to compute residuals, but for encoding we only need boundaries
        # For now, just store the codes - residuals would need codebooks

    return codes


def encode_1d_with_residuals(
    values: np.ndarray,
    codebooks: list[np.ndarray],
    boundaries: list[np.ndarray]
) -> np.ndarray:
    """
    Encode 1D values using RVQ with residual computation.

    Args:
        values: (N,) scalar values
        codebooks: List of L codebooks, each (K,)
        boundaries: List of L arrays, each (K-1,)

    Returns:
        codes: (N, L) array of codebook indices
    """
    n_levels = len(codebooks)
    codes = np.zeros((len(values), n_levels), dtype=np.int32)
    residuals = values.copy()

    for level, (codebook, level_boundaries) in enumerate(zip(codebooks, boundaries)):
        codes[:, level] = np.searchsorted(level_boundaries, residuals)
        residuals = residuals - codebook[codes[:, level]]

    return codes


class MoleculeTokenizer:
    """Tokenizer for molecular data using RVQ codebooks."""

    def __init__(self, codebook_path: str):
        """
        Initialize tokenizer with codebook.

        Args:
            codebook_path: Path to codebook pickle file
        """
        self.codebook = load_codebook(codebook_path)
        self.config = self.codebook["config"]
        self.preprocessing = self.codebook["preprocessing"]

        # Extract codebooks and boundaries
        self.pos_codebooks = self.codebook["codebooks"]["position"]
        self.force_codebooks = {
            "x": self.codebook["codebooks"]["force_x"],
            "y": self.codebook["codebooks"]["force_y"],
            "z": self.codebook["codebooks"]["force_z"],
        }
        self.force_boundaries = {
            "x": self.codebook["boundaries"]["force_x"],
            "y": self.codebook["boundaries"]["force_y"],
            "z": self.codebook["boundaries"]["force_z"],
        }
        self.energy_codebooks = self.codebook["codebooks"]["energy"]
        self.energy_boundaries = self.codebook["boundaries"]["energy"]

        # Compute vocabulary layout
        self.K = self.config["codebook_size"]
        self.pos_levels = self.config["pos_levels"]
        self.n_levels = self.config["n_levels"]

        # Token ranges (all codebook tokens start at CODEBOOK_TOKEN_OFFSET)
        # Position: pos_levels * K tokens
        # Force X/Y/Z: n_levels * K tokens each
        # Energy: n_levels * K tokens
        self.pos_token_start = CODEBOOK_TOKEN_OFFSET
        self.force_x_token_start = self.pos_token_start + self.pos_levels * self.K
        self.force_y_token_start = self.force_x_token_start + self.n_levels * self.K
        self.force_z_token_start = self.force_y_token_start + self.n_levels * self.K
        self.energy_token_start = self.force_z_token_start + self.n_levels * self.K

        self.vocab_size = self.energy_token_start + self.n_levels * self.K

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

        # Position tokens
        for level in range(self.pos_levels):
            for code in range(self.K):
                lookup[self.pos_token_start + level * self.K + code] = f"[P{level}:{code}]"

        # Force X/Y/Z tokens
        for level in range(self.n_levels):
            for code in range(self.K):
                lookup[self.force_x_token_start + level * self.K + code] = f"[FX{level}:{code}]"
                lookup[self.force_y_token_start + level * self.K + code] = f"[FY{level}:{code}]"
                lookup[self.force_z_token_start + level * self.K + code] = f"[FZ{level}:{code}]"

        # Energy tokens
        for level in range(self.n_levels):
            for code in range(self.K):
                lookup[self.energy_token_start + level * self.K + code] = f"[E{level}:{code}]"

        return lookup

    def get_vocab_info(self) -> dict:
        """Return vocabulary information."""
        return {
            "vocab_size": self.vocab_size,
            "special_tokens": SPECIAL_TOKENS,
            "atom_token_offset": ATOM_TOKEN_OFFSET,
            "max_atomic_number": MAX_ATOMIC_NUMBER,
            "codebook_token_offset": CODEBOOK_TOKEN_OFFSET,
            "pos_token_start": self.pos_token_start,
            "force_x_token_start": self.force_x_token_start,
            "force_y_token_start": self.force_y_token_start,
            "force_z_token_start": self.force_z_token_start,
            "energy_token_start": self.energy_token_start,
            "pos_levels": self.pos_levels,
            "n_levels": self.n_levels,
            "codebook_size": self.K,
        }

    def preprocess_positions(self, positions: np.ndarray) -> np.ndarray:
        """Center and scale positions."""
        centered = positions - positions.mean(axis=0, keepdims=True)
        return centered / self.preprocessing["position_scale"]

    def preprocess_forces(self, forces: np.ndarray) -> np.ndarray:
        """Scale forces."""
        return forces / self.preprocessing["force_scale"]

    def preprocess_energy(self, energy: float) -> float:
        """Shift and scale energy."""
        return (energy - self.preprocessing["energy_shift"]) / self.preprocessing["energy_scale"]

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
            shuffle_sections: If True, randomly shuffle the order of sections
            rng: Random number generator for shuffling

        Returns:
            tokens: List of token IDs
        """
        # Preprocess
        pos_scaled = self.preprocess_positions(positions)
        forces_scaled = self.preprocess_forces(forces)
        energy_scaled = self.preprocess_energy(energy)

        # Encode all data
        pos_codes = encode_positions(pos_scaled, self.pos_codebooks)
        fx_codes = encode_1d_with_residuals(forces_scaled[:, 0], self.force_codebooks["x"], self.force_boundaries["x"])
        fy_codes = encode_1d_with_residuals(forces_scaled[:, 1], self.force_codebooks["y"], self.force_boundaries["y"])
        fz_codes = encode_1d_with_residuals(forces_scaled[:, 2], self.force_codebooks["z"], self.force_boundaries["z"])
        energy_codes = encode_1d_with_residuals(np.array([energy_scaled]), self.energy_codebooks, self.energy_boundaries)[0]

        # Build each section
        NL = SPECIAL_TOKENS["\n"]

        # [ATOMS] section: atom types (no newlines needed - single token per atom)
        atoms_section = [SPECIAL_TOKENS["[ATOMS]"]]
        for atomic_num in atomic_numbers:
            atoms_section.append(ATOM_TOKEN_OFFSET + atomic_num - 1)
        atoms_section.append(SPECIAL_TOKENS["[ATOMS_END]"])

        # [POS] section: position codes with newline after each atom's position
        pos_section = [SPECIAL_TOKENS["[POS]"]]
        for i in range(len(atomic_numbers)):
            for level, code in enumerate(pos_codes[i]):
                pos_section.append(self.pos_token_start + level * self.K + code)
            pos_section.append(NL)
        pos_section.append(SPECIAL_TOKENS["[POS_END]"])

        # [FORCE] section: force codes with newline after each atom's forces
        force_section = [SPECIAL_TOKENS["[FORCE]"]]
        for i in range(len(atomic_numbers)):
            for level, code in enumerate(fx_codes[i]):
                force_section.append(self.force_x_token_start + level * self.K + code)
            for level, code in enumerate(fy_codes[i]):
                force_section.append(self.force_y_token_start + level * self.K + code)
            for level, code in enumerate(fz_codes[i]):
                force_section.append(self.force_z_token_start + level * self.K + code)
            force_section.append(NL)
        force_section.append(SPECIAL_TOKENS["[FORCE_END]"])

        # [ENERGY] section (single value with newline)
        energy_section = [SPECIAL_TOKENS["[ENERGY]"]]
        for level, code in enumerate(energy_codes):
            energy_section.append(self.energy_token_start + level * self.K + code)
        energy_section.append(NL)
        energy_section.append(SPECIAL_TOKENS["[ENERGY_END]"])

        # Combine sections (order: atoms, pos, force, energy - energy at end like graph-free-transformer)
        sections = [atoms_section, pos_section, force_section, energy_section]

        if shuffle_sections:
            if rng is None:
                rng = np.random.default_rng()
            rng.shuffle(sections)

        # Build final token sequence
        tokens = [SPECIAL_TOKENS["[BOS]"]]
        for section in sections:
            tokens.extend(section)
        tokens.append(SPECIAL_TOKENS["[EOS]"])

        return tokens

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
            # Slower path: replace [NL] with actual newlines
            parts = []
            for tok in tokens:
                if 0 <= tok < vocab_size:
                    s = lookup[tok]
                    parts.append("\n" if s == "[NL]" else s)
                else:
                    parts.append(f"[UNK:{tok}]")
            return " ".join(parts)

        # Fast path: direct lookup
        return " ".join(
            lookup[tok] if 0 <= tok < vocab_size else f"[UNK:{tok}]"
            for tok in tokens
        )

    def build_vocab(self) -> dict[str, int]:
        """
        Build vocabulary mapping token strings to IDs.

        Returns:
            Dictionary mapping token strings to token IDs
        """
        vocab = {}

        # Special tokens (use [NL] as string representation for newline token)
        for name, tid in SPECIAL_TOKENS.items():
            if name == "\n":
                vocab["[NL]"] = tid  # String representation for tokenizer
            else:
                vocab[name] = tid

        # Atom tokens [Z=1] through [Z=118]
        for z in range(1, MAX_ATOMIC_NUMBER + 1):
            vocab[f"[Z={z}]"] = ATOM_TOKEN_OFFSET + z - 1

        # Position codebook tokens [P0:0] through [P{pos_levels-1}:{K-1}]
        for level in range(self.pos_levels):
            for code in range(self.K):
                tok_id = self.pos_token_start + level * self.K + code
                vocab[f"[P{level}:{code}]"] = tok_id

        # Force X codebook tokens
        for level in range(self.n_levels):
            for code in range(self.K):
                tok_id = self.force_x_token_start + level * self.K + code
                vocab[f"[FX{level}:{code}]"] = tok_id

        # Force Y codebook tokens
        for level in range(self.n_levels):
            for code in range(self.K):
                tok_id = self.force_y_token_start + level * self.K + code
                vocab[f"[FY{level}:{code}]"] = tok_id

        # Force Z codebook tokens
        for level in range(self.n_levels):
            for code in range(self.K):
                tok_id = self.force_z_token_start + level * self.K + code
                vocab[f"[FZ{level}:{code}]"] = tok_id

        # Energy codebook tokens
        for level in range(self.n_levels):
            for code in range(self.K):
                tok_id = self.energy_token_start + level * self.K + code
                vocab[f"[E{level}:{code}]"] = tok_id

        return vocab

    def get_hf_tokenizer(self):
        """
        Create a HuggingFace PreTrainedTokenizerFast for this vocabulary.

        Returns:
            PreTrainedTokenizerFast instance
        """
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        # Build vocabulary
        vocab = self.build_vocab()

        # Add [UNK] to vocab (required for WordLevel)
        vocab["[UNK]"] = len(vocab)

        # Create base tokenizer with WordLevel model
        tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))

        # Set up pre-tokenizer to split only on whitespace (not punctuation)
        # Our tokens are space-separated and contain brackets/colons
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()

        # Wrap in HuggingFace PreTrainedTokenizerFast
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            bos_token="[BOS]",
            eos_token="[EOS]",
        )

        return hf_tokenizer

    def save_hf_tokenizer(self, path: str):
        """
        Save HuggingFace tokenizer to disk.

        Args:
            path: Directory path to save tokenizer
        """
        hf_tokenizer = self.get_hf_tokenizer()
        hf_tokenizer.save_pretrained(path)
        print(f"Saved HuggingFace tokenizer to {path}")


def serialize_csv(
    input_csv: str,
    codebook_path: str,
    output_path: str,
    max_rows: int | None = None,
    show_examples: int = 0,
):
    """
    Serialize CSV data to token sequences.

    Args:
        input_csv: Path to input CSV
        codebook_path: Path to codebook pickle
        output_path: Path to output file
        max_rows: Maximum rows to process
        show_examples: Number of examples to print
    """
    tokenizer = MoleculeTokenizer(codebook_path)

    print("Vocabulary Info:")
    vocab_info = tokenizer.get_vocab_info()
    for key, value in vocab_info.items():
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
            print(f"Token string: {tokenizer.tokens_to_string(tokens[:50])}{'...' if len(tokens) > 50 else ''}")

    # Statistics
    token_counts = np.array(token_counts)
    print(f"\nToken Statistics:")
    print(f"  Total molecules: {len(all_tokens)}")
    print(f"  Total tokens: {token_counts.sum():,}")
    print(f"  Tokens per molecule: min={token_counts.min()}, max={token_counts.max()}, "
          f"mean={token_counts.mean():.1f}, median={np.median(token_counts):.1f}")

    # Save
    output = {
        "vocab_info": vocab_info,
        "tokens": all_tokens,
        "metadata": {
            "input_csv": input_csv,
            "codebook_path": codebook_path,
            "n_molecules": len(all_tokens),
        }
    }

    with open(output_path, "wb") as f:
        pickle.dump(output, f)

    print(f"\nSaved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Serialize molecular data to token sequences")
    parser.add_argument("input_csv", type=str, help="Path to input CSV file")
    parser.add_argument("--codebook", "-c", type=str, default="codebook_mol_1m.pkl",
                        help="Path to codebook pickle file")
    parser.add_argument("--output", "-o", type=str, default="tokenized_molecules.pkl",
                        help="Output pickle file")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Maximum rows to process")
    parser.add_argument("--show-examples", type=int, default=3,
                        help="Number of examples to print")
    args = parser.parse_args()

    serialize_csv(
        args.input_csv,
        args.codebook,
        args.output,
        max_rows=args.max_rows,
        show_examples=args.show_examples,
    )


if __name__ == "__main__":
    main()

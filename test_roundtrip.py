"""Test round-trip encoding/decoding through HuggingFace tokenizer."""

import ast
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from serialize_molecules import MoleculeTokenizer


def test_roundtrip():
    """Test round-trip through HF tokenizer."""
    print("=" * 60)
    print("Round-trip test: omol25 → serialize → HF tokenize/detokenize → deserialize")
    print("=" * 60)

    # 1. Load one row from omol25 sample
    print("\n1. Loading one row from omol25_train_sample_1k.csv...")
    df = pd.read_csv("omol25_train_sample_1k.csv", nrows=1)
    row = df.iloc[0]

    atomic_numbers = ast.literal_eval(row["atomic_numbers"])
    positions = np.array(ast.literal_eval(row["positions"]))
    forces = np.array(ast.literal_eval(row["atomic_forces"]))
    energy = float(row["energy"])

    print(f"   Atoms: {len(atomic_numbers)}, Energy: {energy:.4f}")

    # 2. Create tokenizer and encode
    print("\n2. Encoding with MoleculeTokenizer...")
    tokenizer = MoleculeTokenizer("codebook_mol_1m.pkl")
    tokens = tokenizer.encode_molecule(atomic_numbers, positions, forces, energy)
    token_string = tokenizer.tokens_to_string(tokens)
    print(f"   Tokens: {len(tokens)}, String length: {len(token_string)}")

    # 3. Round-trip through HF tokenizer
    print("\n3. HF tokenizer round-trip...")
    hf_tokenizer = AutoTokenizer.from_pretrained("WillHeld/marin-tomol")
    hf_ids = hf_tokenizer(token_string, add_special_tokens=False)["input_ids"]
    decoded_string = hf_tokenizer.decode(hf_ids, skip_special_tokens=False)
    print(f"   HF IDs match original: {hf_ids == tokens}")
    print(f"   String preserved: {decoded_string == token_string}")

    # 4. Decode back to molecular data
    print("\n4. Decoding tokens back to molecular data...")
    decoded = tokenizer.decode_molecule(hf_ids)

    # 5. Compute reconstruction errors
    print("\n5. Reconstruction errors:")

    # Atomic numbers (should be exact)
    atoms_match = decoded["atomic_numbers"] == atomic_numbers
    print(f"   Atomic numbers match: {atoms_match}")

    # Positions (center both for fair comparison since we lose absolute position)
    pos_orig = positions - positions.mean(axis=0)
    pos_dec = decoded["positions"] - decoded["positions"].mean(axis=0)
    pos_rmse = np.sqrt(np.mean((pos_orig - pos_dec) ** 2))
    pos_max = np.max(np.abs(pos_orig - pos_dec))
    print(f"   Position RMSE: {pos_rmse:.6f} Å, Max: {pos_max:.6f} Å")

    # Forces
    force_rmse = np.sqrt(np.mean((forces - decoded["forces"]) ** 2))
    force_max = np.max(np.abs(forces - decoded["forces"]))
    print(f"   Force RMSE: {force_rmse:.6f} eV/Å, Max: {force_max:.6f} eV/Å")

    # Energy
    energy_err = abs(energy - decoded["energy"])
    energy_rel = energy_err / abs(energy) * 100
    print(f"   Energy error: {energy_err:.6f} eV ({energy_rel:.4f}%)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_roundtrip()

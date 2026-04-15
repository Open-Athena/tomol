"""Test round-trip encoding/decoding through HuggingFace tokenizer."""

import ast
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from serialize_molecules import MoleculeTokenizer


def test_roundtrip(config_path: str = "fp16_config.json"):
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
    print(f"\n2. Encoding with MoleculeTokenizer (config: {config_path})...")
    tokenizer = MoleculeTokenizer(config_path)
    tokens = tokenizer.encode_molecule(atomic_numbers, positions, forces, energy)
    token_string = tokenizer.tokens_to_string(tokens)
    print(f"   Tokens: {len(tokens)}, String length: {len(token_string)}")
    print(f"   Vocab size: {tokenizer.vocab_size}")

    # 3. Round-trip through HF tokenizer
    print("\n3. HF tokenizer round-trip...")
    hf_tokenizer = tokenizer.get_hf_tokenizer()
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
    pos_mae = np.mean(np.abs(pos_orig - pos_dec))
    print(f"   Position MAE:  {pos_mae:.6f} Å ({pos_mae * 1000:.4f} mÅ)")
    print(f"   Position RMSE: {pos_rmse:.6f} Å, Max: {pos_max:.6f} Å")

    # Forces
    force_rmse = np.sqrt(np.mean((forces - decoded["forces"]) ** 2))
    force_max = np.max(np.abs(forces - decoded["forces"]))
    force_mae = np.mean(np.abs(forces - decoded["forces"]))
    print(f"   Force MAE:  {force_mae:.6f} eV/Å ({force_mae * 1000:.4f} meV/Å)")
    print(f"   Force RMSE: {force_rmse:.6f} eV/Å, Max: {force_max:.6f} eV/Å")

    # Energy
    energy_err = abs(energy - decoded["energy"])
    energy_rel = energy_err / abs(energy) * 100 if energy != 0 else 0
    print(f"   Energy error: {energy_err:.6f} eV ({energy_err * 1000:.4f} meV, {energy_rel:.6f}%)")

    # 6. Precision checks (based on plan targets)
    print("\n6. Precision checks vs targets:")
    pos_target = 0.001  # 0.001 Å = 1 mÅ
    force_target = 0.001  # 0.001 eV/Å = 1 meV/Å
    energy_target = 0.0001  # 0.1 meV

    pos_ok = pos_mae < pos_target
    force_ok = force_mae < force_target
    energy_ok = energy_err < energy_target

    print(f"   Position MAE < {pos_target * 1000:.1f} mÅ: {'✓ PASS' if pos_ok else '✗ FAIL'} ({pos_mae * 1000:.4f} mÅ)")
    print(f"   Force MAE < {force_target * 1000:.1f} meV/Å: {'✓ PASS' if force_ok else '✗ FAIL'} ({force_mae * 1000:.4f} meV/Å)")
    print(f"   Energy error < {energy_target * 1000:.1f} meV: {'✓ PASS' if energy_ok else '✗ FAIL'} ({energy_err * 1000:.4f} meV)")

    print("\n" + "=" * 60)

    return {
        "atoms_match": atoms_match,
        "position_mae": pos_mae,
        "position_rmse": pos_rmse,
        "force_mae": force_mae,
        "force_rmse": force_rmse,
        "energy_error": energy_err,
    }


def test_multiple_molecules(config_path: str = "fp16_config.json", n_molecules: int = 10):
    """Test round-trip on multiple molecules."""
    print("=" * 60)
    print(f"Testing round-trip on {n_molecules} molecules")
    print("=" * 60)

    df = pd.read_csv("omol25_train_sample_1k.csv", nrows=n_molecules)
    tokenizer = MoleculeTokenizer(config_path)

    pos_errors = []
    force_errors = []
    energy_errors = []
    all_atoms_match = True

    for idx, row in df.iterrows():
        atomic_numbers = ast.literal_eval(row["atomic_numbers"])
        positions = np.array(ast.literal_eval(row["positions"]))
        forces = np.array(ast.literal_eval(row["atomic_forces"]))
        energy = float(row["energy"])

        # Encode and decode
        tokens = tokenizer.encode_molecule(atomic_numbers, positions, forces, energy)
        decoded = tokenizer.decode_molecule(tokens)

        # Check atoms
        if decoded["atomic_numbers"] != atomic_numbers:
            all_atoms_match = False

        # Compute errors (center positions)
        pos_orig = positions - positions.mean(axis=0)
        pos_dec = decoded["positions"] - decoded["positions"].mean(axis=0)
        pos_errors.append(np.mean(np.abs(pos_orig - pos_dec)))
        force_errors.append(np.mean(np.abs(forces - decoded["forces"])))
        energy_errors.append(abs(energy - decoded["energy"]))

    # Summary statistics
    print(f"\nResults over {n_molecules} molecules:")
    print(f"  All atoms match: {all_atoms_match}")
    print(f"  Position MAE:  {np.mean(pos_errors)*1000:.4f} mÅ (max: {np.max(pos_errors)*1000:.4f} mÅ)")
    print(f"  Force MAE:     {np.mean(force_errors)*1000:.4f} meV/Å (max: {np.max(force_errors)*1000:.4f} meV/Å)")
    print(f"  Energy error:  {np.mean(energy_errors)*1000:.4f} meV (max: {np.max(energy_errors)*1000:.4f} meV)")


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "fp16_config.json"

    test_roundtrip(config_path)
    print()
    test_multiple_molecules(config_path)

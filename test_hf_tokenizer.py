"""Test that the HuggingFace tokenizer at WillHeld/marin-tomol matches local implementation."""

import ast
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from serialize_molecules import MoleculeTokenizer, VOCAB_SIZE


def test_hf_tokenizer():
    """Test that HF tokenizer from hub matches local implementation."""
    print("=" * 60)
    print("Testing HuggingFace tokenizer: WillHeld/marin-tomol")
    print("=" * 60)

    # 1. Load HF tokenizer from hub
    print("\n1. Loading tokenizer from HuggingFace Hub...")
    hf_tokenizer = AutoTokenizer.from_pretrained("WillHeld/marin-tomol")
    print(f"   Vocab size (HF): {hf_tokenizer.vocab_size}")
    print(f"   Special tokens: pad={hf_tokenizer.pad_token}, bos={hf_tokenizer.bos_token}, eos={hf_tokenizer.eos_token}")

    # 2. Load local tokenizer
    print("\n2. Loading local MoleculeTokenizer...")
    local_tokenizer = MoleculeTokenizer("fp16_config.json")
    local_hf = local_tokenizer.get_hf_tokenizer()
    print(f"   Vocab size (local): {local_tokenizer.vocab_size}")
    print(f"   Expected vocab size: {VOCAB_SIZE}")

    # 3. Compare vocabularies
    print("\n3. Comparing vocabularies...")
    hf_vocab = hf_tokenizer.get_vocab()
    local_vocab = local_hf.get_vocab()

    # Check sizes (HF might have extra [UNK] token)
    print(f"   HF vocab entries: {len(hf_vocab)}")
    print(f"   Local vocab entries: {len(local_vocab)}")

    # Compare common tokens
    common_tokens = set(hf_vocab.keys()) & set(local_vocab.keys())
    mismatches = []
    for token in common_tokens:
        if hf_vocab[token] != local_vocab[token]:
            mismatches.append((token, hf_vocab[token], local_vocab[token]))

    if mismatches:
        print(f"   MISMATCH: {len(mismatches)} tokens have different IDs!")
        for token, hf_id, local_id in mismatches[:10]:
            print(f"      {token}: HF={hf_id}, local={local_id}")
    else:
        print(f"   All {len(common_tokens)} common tokens match!")

    # Check for tokens only in one vocab
    hf_only = set(hf_vocab.keys()) - set(local_vocab.keys())
    local_only = set(local_vocab.keys()) - set(hf_vocab.keys())
    if hf_only:
        print(f"   Tokens only in HF: {list(hf_only)[:5]}...")
    if local_only:
        print(f"   Tokens only in local: {list(local_only)[:5]}...")

    # 4. Test encoding/decoding with sample data
    print("\n4. Testing encode/decode with molecule data...")
    try:
        df = pd.read_csv("omol25_train_sample_1k.csv", nrows=5)
    except FileNotFoundError:
        print("   Skipping molecule test (no sample data available)")
        return

    all_passed = True
    for idx, row in df.iterrows():
        atomic_numbers = ast.literal_eval(row["atomic_numbers"])
        positions = np.array(ast.literal_eval(row["positions"]))
        forces = np.array(ast.literal_eval(row["atomic_forces"]))
        energy = float(row["energy"])

        # Encode with local tokenizer
        tokens = local_tokenizer.encode_molecule(atomic_numbers, positions, forces, energy)
        token_string = local_tokenizer.tokens_to_string(tokens)

        # Tokenize through HF tokenizer from hub
        hf_ids = hf_tokenizer(token_string, add_special_tokens=False)["input_ids"]

        # Check if IDs match
        ids_match = hf_ids == tokens

        # Decode back through both tokenizers
        hf_decoded = hf_tokenizer.decode(hf_ids, skip_special_tokens=False)
        local_decoded = local_hf.decode(tokens, skip_special_tokens=False)
        strings_match = hf_decoded == local_decoded

        status = "PASS" if ids_match and strings_match else "FAIL"
        if not (ids_match and strings_match):
            all_passed = False

        print(f"   Molecule {idx}: {len(atomic_numbers)} atoms, {len(tokens)} tokens - {status}")
        if not ids_match:
            print(f"      Token ID mismatch at positions: {[i for i, (a, b) in enumerate(zip(hf_ids, tokens)) if a != b][:5]}")
        if not strings_match:
            print(f"      String mismatch!")

    # 5. Test roundtrip through HF tokenizer
    print("\n5. Testing roundtrip decode accuracy...")
    row = df.iloc[0]
    atomic_numbers = ast.literal_eval(row["atomic_numbers"])
    positions = np.array(ast.literal_eval(row["positions"]))
    forces = np.array(ast.literal_eval(row["atomic_forces"]))
    energy = float(row["energy"])

    tokens = local_tokenizer.encode_molecule(atomic_numbers, positions, forces, energy)
    token_string = local_tokenizer.tokens_to_string(tokens)

    # Round-trip through HF tokenizer
    hf_ids = hf_tokenizer(token_string, add_special_tokens=False)["input_ids"]
    decoded = local_tokenizer.decode_molecule(hf_ids)

    # Check reconstruction
    atoms_match = decoded["atomic_numbers"] == atomic_numbers

    pos_orig = positions - positions.mean(axis=0)
    pos_dec = decoded["positions"] - decoded["positions"].mean(axis=0)
    pos_mae = np.mean(np.abs(pos_orig - pos_dec))

    force_mae = np.mean(np.abs(forces - decoded["forces"]))
    energy_err = abs(energy - decoded["energy"])

    print(f"   Atomic numbers match: {atoms_match}")
    print(f"   Position MAE: {pos_mae * 1000:.4f} mA")
    print(f"   Force MAE: {force_mae * 1000:.4f} meV/A")
    print(f"   Energy error: {energy_err * 1000:.4f} meV")

    # 6. Final summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    vocab_ok = len(mismatches) == 0
    print(f"  Vocabulary match: {'PASS' if vocab_ok else 'FAIL'}")
    print(f"  Tokenization match: {'PASS' if all_passed else 'FAIL'}")
    print(f"  Reconstruction accuracy: {'PASS' if atoms_match and pos_mae < 0.001 else 'FAIL'}")

    overall = vocab_ok and all_passed and atoms_match
    print(f"\n  OVERALL: {'PASS' if overall else 'FAIL'}")

    return overall


if __name__ == "__main__":
    test_hf_tokenizer()

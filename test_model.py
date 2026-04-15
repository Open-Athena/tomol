#!/usr/bin/env python3
"""Test WillHeld/ToMol-marin-1B on a few datapoints from the sample CSV."""

import ast
import time
import numpy as np
import pandas as pd
import torch
from ase import Atoms
from transformers import AutoModelForCausalLM

from serialize_molecules import MoleculeTokenizer


def batched_inference(
    model,
    mol_tokenizer,
    hf_tokenizer,
    samples: list[dict],
    device: str = "cuda",
    max_new_tokens: int = 2048,
) -> list[dict]:
    """
    Run batched inference on multiple molecules.

    Args:
        model: The loaded causal LM
        mol_tokenizer: MoleculeTokenizer instance
        hf_tokenizer: HuggingFace tokenizer
        samples: List of dicts with 'atomic_numbers' and 'positions'
        device: Device to run on
        max_new_tokens: Max tokens to generate

    Returns:
        List of dicts with 'energy' and 'forces' predictions
    """
    vocab_info = mol_tokenizer.get_vocab_info()
    eos_id = vocab_info["special_tokens"]["[EOS]"]
    force_id = vocab_info["special_tokens"]["[FORCE]"]
    pad_id = vocab_info["special_tokens"]["[PAD]"]

    # Build input sequences for each sample
    all_input_tokens = []
    for sample in samples:
        full_tokens = mol_tokenizer.encode_molecule(
            atomic_numbers=sample["atomic_numbers"],
            positions=sample["positions"],
            forces=np.zeros_like(sample["positions"]),
            energy=0.0,
            shuffle_sections=False,
        )
        # Truncate at [FORCE] to prompt for generation
        force_idx = full_tokens.index(force_id)
        input_tokens = full_tokens[:force_idx + 1]
        all_input_tokens.append(input_tokens)

    # Left-pad to same length
    max_len = max(len(t) for t in all_input_tokens)
    padded_inputs = []
    attention_masks = []

    for tokens in all_input_tokens:
        pad_len = max_len - len(tokens)
        padded = [pad_id] * pad_len + tokens
        mask = [0] * pad_len + [1] * len(tokens)
        padded_inputs.append(padded)
        attention_masks.append(mask)

    input_ids = torch.tensor(padded_inputs, device=device)
    attention_mask = torch.tensor(attention_masks, device=device)

    # Generate in batch
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )

    # Parse each output
    results = []
    for i, (output, sample) in enumerate(zip(outputs, samples)):
        output_tokens = output.cpu().numpy().tolist()
        # Remove padding from the start
        while output_tokens and output_tokens[0] == pad_id:
            output_tokens.pop(0)

        n_atoms = len(sample["atomic_numbers"])
        try:
            decoded = mol_tokenizer.decode_molecule(output_tokens)
            forces = decoded.get("forces")
            energy = decoded.get("energy")

            if forces is None or len(forces) != n_atoms:
                forces = np.zeros((n_atoms, 3))
            if energy is None:
                energy = 0.0

            results.append({
                "energy": float(energy),
                "forces": np.array(forces),
                "valid": True,
            })
        except Exception as e:
            results.append({
                "energy": 0.0,
                "forces": np.zeros((n_atoms, 3)),
                "valid": False,
                "error": str(e),
            })

    return results


def test_model_on_samples(
    model_name: str = "WillHeld/ToMol-marin-1B",
    config_path: str = "fp16_config.json",
    csv_path: str = "omol25_train_sample_1k.csv",
    n_samples: int = 10,
    max_atoms: int = 100,  # Allow larger molecules
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Test the model on a few sample molecules."""
    print("=" * 70)
    print(f"Testing model: {model_name}")
    print(f"Device: {device}")
    print("=" * 70)

    # Load tokenizer
    print("\n1. Loading MoleculeTokenizer...")
    mol_tokenizer = MoleculeTokenizer(config_path)
    vocab_info = mol_tokenizer.get_vocab_info()
    hf_tokenizer = mol_tokenizer.get_hf_tokenizer()
    print(f"   Vocab size: {vocab_info['vocab_size']}")

    # Load model
    print(f"\n2. Loading model from {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(f"   Model loaded: {model.config.model_type}")

    # Get special token IDs
    eos_id = vocab_info["special_tokens"]["[EOS]"]
    force_id = vocab_info["special_tokens"]["[FORCE]"]
    pos_end_id = vocab_info["special_tokens"]["[POS_END]"]

    # Load sample data - filter to small molecules
    print(f"\n3. Loading samples with <= {max_atoms} atoms from {csv_path}...")
    df = pd.read_csv(csv_path)
    # Filter to small molecules
    df["n_atoms"] = df["atomic_numbers"].apply(lambda x: len(ast.literal_eval(x)))
    df = df[df["n_atoms"] <= max_atoms].head(n_samples)
    print(f"   Found {len(df)} molecules with <= {max_atoms} atoms")

    results = []
    for idx, row in df.iterrows():
        print(f"\n{'='*70}")
        print(f"Sample {idx + 1}/{n_samples}")
        print(f"{'='*70}")

        # Parse molecule data
        atomic_numbers = ast.literal_eval(row["atomic_numbers"])
        positions = np.array(ast.literal_eval(row["positions"]))
        forces_true = np.array(ast.literal_eval(row["atomic_forces"]))
        energy_true = float(row["energy"])

        n_atoms = len(atomic_numbers)
        print(f"  Atoms: {n_atoms}")
        print(f"  Atomic numbers: {atomic_numbers[:5]}{'...' if n_atoms > 5 else ''}")
        print(f"  Target energy: {energy_true:.4f} eV")
        print(f"  Target forces shape: {forces_true.shape}")

        # Encode full molecule (with dummy forces/energy) to get input tokens
        full_tokens = mol_tokenizer.encode_molecule(
            atomic_numbers=atomic_numbers,
            positions=positions,
            forces=np.zeros_like(positions),  # dummy
            energy=0.0,  # dummy
            shuffle_sections=False,
        )

        # Truncate at [FORCE] to prompt model to generate forces + energy
        try:
            force_idx = full_tokens.index(force_id)
            input_tokens = full_tokens[:force_idx + 1]
        except ValueError:
            pos_end_idx = full_tokens.index(pos_end_id)
            input_tokens = full_tokens[:pos_end_idx + 1] + [force_id]

        print(f"  Input tokens: {len(input_tokens)}")
        print(f"  Input prompt (first 100 chars): {mol_tokenizer.tokens_to_string(input_tokens[:30])}...")

        # Generate completion
        input_ids = torch.tensor([input_tokens], device=device)

        print("\n  Generating prediction...")
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=hf_tokenizer.pad_token_id or 0,
                eos_token_id=eos_id,
            )

        # Decode output
        output_tokens = outputs[0].cpu().numpy().tolist()
        print(f"  Generated {len(output_tokens) - len(input_tokens)} new tokens")

        # Parse predictions - check validity
        generated_tokens = output_tokens[len(input_tokens):]
        generated_str = mol_tokenizer.tokens_to_string(generated_tokens)

        # Check structural validity
        has_force_end = "[FORCE_END]" in generated_str
        has_energy = "[ENERGY]" in generated_str
        has_energy_end = "[ENERGY_END]" in generated_str
        has_eos = "[EOS]" in generated_str

        print(f"\n  Structural validity:")
        print(f"    Has [FORCE_END]: {has_force_end}")
        print(f"    Has [ENERGY]:    {has_energy}")
        print(f"    Has [ENERGY_END]: {has_energy_end}")
        print(f"    Has [EOS]:       {has_eos}")

        is_valid_structure = has_force_end and has_energy and has_energy_end and has_eos

        try:
            decoded = mol_tokenizer.decode_molecule(output_tokens)
            forces_pred = decoded.get("forces")
            energy_pred = decoded.get("energy")
            atoms_decoded = decoded.get("atomic_numbers", [])

            # Check semantic validity
            atoms_match = len(atoms_decoded) == n_atoms
            forces_shape_ok = forces_pred is not None and len(forces_pred) == n_atoms
            energy_ok = energy_pred is not None

            print(f"\n  Semantic validity:")
            print(f"    Atoms count match: {atoms_match} (decoded {len(atoms_decoded)}, expected {n_atoms})")
            print(f"    Forces shape OK:   {forces_shape_ok} (got {forces_pred.shape if forces_pred is not None else None})")
            print(f"    Energy decoded:    {energy_ok}")

            is_valid_semantic = atoms_match and forces_shape_ok and energy_ok

            if not forces_shape_ok:
                print(f"  WARNING: Forces shape mismatch!")
                forces_pred = np.zeros((n_atoms, 3))

            if not energy_ok:
                print("  WARNING: Energy not decoded!")
                energy_pred = 0.0

            # Compute errors
            energy_error = abs(energy_pred - energy_true)
            force_mae = np.mean(np.abs(forces_pred - forces_true))
            force_rmse = np.sqrt(np.mean((forces_pred - forces_true) ** 2))

            print(f"\n  Results:")
            print(f"    Predicted energy: {energy_pred:.4f} eV")
            print(f"    Energy error:     {energy_error:.4f} eV ({energy_error * 1000:.2f} meV)")
            print(f"    Force MAE:        {force_mae:.6f} eV/A ({force_mae * 1000:.2f} meV/A)")
            print(f"    Force RMSE:       {force_rmse:.6f} eV/A ({force_rmse * 1000:.2f} meV/A)")

            results.append({
                "n_atoms": n_atoms,
                "valid_structure": is_valid_structure,
                "valid_semantic": is_valid_semantic,
                "energy_true": energy_true,
                "energy_pred": energy_pred,
                "energy_error_eV": energy_error,
                "force_mae_eV_A": force_mae,
                "force_rmse_eV_A": force_rmse,
            })

        except Exception as e:
            print(f"\n  ERROR decoding output: {e}")
            print(f"  Generated tokens (first 200 chars): {generated_str[:200]}...")
            results.append({
                "n_atoms": n_atoms,
                "valid_structure": is_valid_structure,
                "valid_semantic": False,
                "error": str(e),
            })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    n_valid_structure = sum(1 for r in results if r.get("valid_structure", False))
    n_valid_semantic = sum(1 for r in results if r.get("valid_semantic", False))
    valid_results = [r for r in results if "error" not in r and r.get("valid_semantic", False)]

    print(f"\n  Validity:")
    print(f"    Valid structure (all sections): {n_valid_structure}/{len(results)} ({100*n_valid_structure/len(results):.1f}%)")
    print(f"    Valid semantic (correct shapes): {n_valid_semantic}/{len(results)} ({100*n_valid_semantic/len(results):.1f}%)")

    if valid_results:
        avg_energy_err = np.mean([r["energy_error_eV"] for r in valid_results])
        avg_force_mae = np.mean([r["force_mae_eV_A"] for r in valid_results])
        print(f"\n  Accuracy (on valid predictions):")
        print(f"    Average energy error: {avg_energy_err:.4f} eV ({avg_energy_err * 1000:.2f} meV)")
        print(f"    Average force MAE:    {avg_force_mae:.6f} eV/A ({avg_force_mae * 1000:.2f} meV/A)")
    else:
        print("\n  No valid predictions to compute accuracy!")

    return results


def batched_inference_bucketed(
    model,
    mol_tokenizer,
    hf_tokenizer,
    samples: list[dict],
    device: str = "cuda",
    max_new_tokens: int = 2048,
    bucket_size: int = 8,
) -> list[dict]:
    """
    Run batched inference with bucketing by input length.
    Processes similar-length inputs together to minimize padding.
    """
    vocab_info = mol_tokenizer.get_vocab_info()
    eos_id = vocab_info["special_tokens"]["[EOS]"]
    force_id = vocab_info["special_tokens"]["[FORCE]"]
    pad_id = vocab_info["special_tokens"]["[PAD]"]

    # Build input sequences and track original indices
    indexed_inputs = []
    for i, sample in enumerate(samples):
        full_tokens = mol_tokenizer.encode_molecule(
            atomic_numbers=sample["atomic_numbers"],
            positions=sample["positions"],
            forces=np.zeros_like(sample["positions"]),
            energy=0.0,
            shuffle_sections=False,
        )
        force_idx = full_tokens.index(force_id)
        input_tokens = full_tokens[:force_idx + 1]
        indexed_inputs.append((i, input_tokens, sample))

    # Sort by input length for efficient bucketing
    indexed_inputs.sort(key=lambda x: len(x[1]))

    # Process in buckets
    all_results = [None] * len(samples)

    for bucket_start in range(0, len(indexed_inputs), bucket_size):
        bucket = indexed_inputs[bucket_start:bucket_start + bucket_size]

        # Left-pad within this bucket
        max_len = max(len(t[1]) for t in bucket)
        padded_inputs = []
        attention_masks = []
        original_indices = []

        for orig_idx, tokens, sample in bucket:
            pad_len = max_len - len(tokens)
            padded = [pad_id] * pad_len + tokens
            mask = [0] * pad_len + [1] * len(tokens)
            padded_inputs.append(padded)
            attention_masks.append(mask)
            original_indices.append((orig_idx, sample))

        input_ids = torch.tensor(padded_inputs, device=device)
        attention_mask = torch.tensor(attention_masks, device=device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )

        # Parse outputs
        for output, (orig_idx, sample) in zip(outputs, original_indices):
            output_tokens = output.cpu().numpy().tolist()
            while output_tokens and output_tokens[0] == pad_id:
                output_tokens.pop(0)

            n_atoms = len(sample["atomic_numbers"])
            try:
                decoded = mol_tokenizer.decode_molecule(output_tokens)
                forces = decoded.get("forces")
                energy = decoded.get("energy")

                if forces is None or len(forces) != n_atoms:
                    forces = np.zeros((n_atoms, 3))
                if energy is None:
                    energy = 0.0

                all_results[orig_idx] = {
                    "energy": float(energy),
                    "forces": np.array(forces),
                    "valid": True,
                }
            except Exception as e:
                all_results[orig_idx] = {
                    "energy": 0.0,
                    "forces": np.zeros((n_atoms, 3)),
                    "valid": False,
                    "error": str(e),
                }

    return all_results


def test_batched_inference(
    model_name: str = "WillHeld/ToMol-marin-1B",
    config_path: str = "fp16_config.json",
    csv_path: str = "omol25_train_sample_1k.csv",
    n_samples: int = 20,
    batch_size: int = 8,
    max_atoms: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Compare sequential vs batched inference."""
    print("=" * 70)
    print("BATCHED INFERENCE TEST")
    print(f"Model: {model_name}")
    print(f"Batch size: {batch_size}, Samples: {n_samples}")
    print("=" * 70)

    # Load tokenizer and model
    mol_tokenizer = MoleculeTokenizer(config_path)
    hf_tokenizer = mol_tokenizer.get_hf_tokenizer()
    vocab_info = mol_tokenizer.get_vocab_info()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # Load samples
    df = pd.read_csv(csv_path)
    df["n_atoms"] = df["atomic_numbers"].apply(lambda x: len(ast.literal_eval(x)))
    df = df[df["n_atoms"] <= max_atoms].head(n_samples)

    samples = []
    for _, row in df.iterrows():
        samples.append({
            "atomic_numbers": ast.literal_eval(row["atomic_numbers"]),
            "positions": np.array(ast.literal_eval(row["positions"])),
            "forces_true": np.array(ast.literal_eval(row["atomic_forces"])),
            "energy_true": float(row["energy"]),
        })

    print(f"Loaded {len(samples)} samples (max {max_atoms} atoms)")

    # --- Sequential inference ---
    print(f"\n{'='*70}")
    print("Sequential inference...")
    eos_id = vocab_info["special_tokens"]["[EOS]"]
    force_id = vocab_info["special_tokens"]["[FORCE]"]

    start_seq = time.time()
    seq_results = []
    for sample in samples:
        full_tokens = mol_tokenizer.encode_molecule(
            atomic_numbers=sample["atomic_numbers"],
            positions=sample["positions"],
            forces=np.zeros_like(sample["positions"]),
            energy=0.0,
            shuffle_sections=False,
        )
        force_idx = full_tokens.index(force_id)
        input_tokens = full_tokens[:force_idx + 1]

        input_ids = torch.tensor([input_tokens], device=device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=0,
                eos_token_id=eos_id,
            )
        output_tokens = outputs[0].cpu().numpy().tolist()
        decoded = mol_tokenizer.decode_molecule(output_tokens)
        seq_results.append({
            "energy": decoded.get("energy", 0.0),
            "forces": decoded.get("forces", np.zeros((len(sample["atomic_numbers"]), 3))),
        })
    time_seq = time.time() - start_seq
    print(f"  Time: {time_seq:.2f}s ({time_seq/len(samples)*1000:.1f}ms per sample)")

    # --- Batched inference (naive) ---
    print(f"\n{'='*70}")
    print(f"Batched inference - naive (batch_size={batch_size})...")

    start_batch = time.time()
    batch_results = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        results = batched_inference(
            model, mol_tokenizer, hf_tokenizer, batch, device
        )
        batch_results.extend(results)
    time_batch = time.time() - start_batch
    print(f"  Time: {time_batch:.2f}s ({time_batch/len(samples)*1000:.1f}ms per sample)")

    # --- Batched inference (bucketed by length) ---
    print(f"\n{'='*70}")
    print(f"Batched inference - bucketed (batch_size={batch_size})...")

    start_bucketed = time.time()
    bucketed_results = batched_inference_bucketed(
        model, mol_tokenizer, hf_tokenizer, samples, device, bucket_size=batch_size
    )
    time_bucketed = time.time() - start_bucketed
    print(f"  Time: {time_bucketed:.2f}s ({time_bucketed/len(samples)*1000:.1f}ms per sample)")

    # --- Compare results ---
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Speedup: {time_seq/time_batch:.2f}x")

    # Check that results match
    energy_diffs = []
    force_diffs = []
    for seq, bat in zip(seq_results, batch_results):
        energy_diffs.append(abs(seq["energy"] - bat["energy"]))
        force_diffs.append(np.max(np.abs(seq["forces"] - bat["forces"])))

    print(f"  Max energy diff (seq vs batch): {max(energy_diffs):.6f} eV")
    print(f"  Max force diff (seq vs batch):  {max(force_diffs):.6f} eV/A")

    # Compute accuracy
    energy_errors = []
    force_maes = []
    for sample, result in zip(samples, batch_results):
        energy_errors.append(abs(result["energy"] - sample["energy_true"]))
        force_maes.append(np.mean(np.abs(result["forces"] - sample["forces_true"])))

    print(f"\n  Accuracy (batched):")
    print(f"    Mean energy error: {np.mean(energy_errors)*1000:.2f} meV")
    print(f"    Mean force MAE:    {np.mean(force_maes)*1000:.2f} meV/A")

    return {
        "time_sequential": time_seq,
        "time_batched": time_batch,
        "speedup": time_seq / time_batch,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        test_batched_inference()
    else:
        test_model_on_samples()

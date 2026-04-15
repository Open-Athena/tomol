#!/usr/bin/env python3
"""Test WillHeld/ToMol-marin-1B with vLLM for efficient batched inference."""

import ast
import time
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams

from serialize_molecules import MoleculeTokenizer


def test_vllm_inference(
    model_name: str = "WillHeld/ToMol-marin-1B",
    config_path: str = "fp16_config.json",
    csv_path: str = "omol25_train_sample_1k.csv",
    n_samples: int = 20,
    max_atoms: int = 50,
):
    """Test vLLM batched inference."""
    print("=" * 70)
    print("vLLM INFERENCE TEST")
    print(f"Model: {model_name}")
    print(f"Samples: {n_samples}")
    print("=" * 70)

    # Load tokenizer
    print("\n1. Loading MoleculeTokenizer...")
    mol_tokenizer = MoleculeTokenizer(config_path)
    vocab_info = mol_tokenizer.get_vocab_info()
    print(f"   Vocab size: {vocab_info['vocab_size']}")

    # Load vLLM model
    print(f"\n2. Loading model with vLLM...")
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    print("   Model loaded!")

    # Sampling params - greedy decoding
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=0,  # greedy
        stop_token_ids=[vocab_info["special_tokens"]["[EOS]"]],
    )

    # Load samples
    print(f"\n3. Loading samples from {csv_path}...")
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
    print(f"   Loaded {len(samples)} samples")

    # Build prompts
    print("\n4. Building input prompts...")
    force_id = vocab_info["special_tokens"]["[FORCE]"]

    prompts = []
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
        prompt_str = mol_tokenizer.tokens_to_string(input_tokens)
        prompts.append(prompt_str)

    print(f"   Built {len(prompts)} prompts")
    print(f"   Example prompt (first 100 chars): {prompts[0][:100]}...")

    # Run inference
    print("\n5. Running vLLM inference...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start_time
    print(f"   Time: {elapsed:.2f}s ({elapsed/len(samples)*1000:.1f}ms per sample)")

    # Parse results
    print("\n6. Parsing outputs...")
    results = []
    n_valid = 0

    for i, (output, sample) in enumerate(zip(outputs, samples)):
        generated_text = output.outputs[0].text
        full_text = prompts[i] + " " + generated_text

        # Convert back to tokens
        hf_tokenizer = mol_tokenizer.get_hf_tokenizer()
        token_ids = hf_tokenizer.encode(full_text, add_special_tokens=False)

        n_atoms = len(sample["atomic_numbers"])

        try:
            decoded = mol_tokenizer.decode_molecule(token_ids)
            forces = decoded.get("forces")
            energy = decoded.get("energy")

            valid_forces = forces is not None and len(forces) == n_atoms
            valid_energy = energy is not None

            if not valid_forces:
                forces = np.zeros((n_atoms, 3))
            if not valid_energy:
                energy = 0.0

            is_valid = valid_forces and valid_energy
            if is_valid:
                n_valid += 1

            # Compute errors
            energy_error = abs(energy - sample["energy_true"])
            force_mae = np.mean(np.abs(forces - sample["forces_true"]))

            results.append({
                "n_atoms": n_atoms,
                "valid": is_valid,
                "energy_pred": energy,
                "energy_true": sample["energy_true"],
                "energy_error": energy_error,
                "force_mae": force_mae,
            })

        except Exception as e:
            results.append({
                "n_atoms": n_atoms,
                "valid": False,
                "error": str(e),
            })

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    print(f"\n  Validity: {n_valid}/{len(results)} ({100*n_valid/len(results):.1f}%)")
    print(f"  Throughput: {len(samples)/elapsed:.2f} samples/sec")

    valid_results = [r for r in results if r.get("valid", False)]
    if valid_results:
        avg_energy_err = np.mean([r["energy_error"] for r in valid_results])
        avg_force_mae = np.mean([r["force_mae"] for r in valid_results])
        print(f"\n  Accuracy (on valid predictions):")
        print(f"    Mean energy error: {avg_energy_err*1000:.2f} meV")
        print(f"    Mean force MAE:    {avg_force_mae*1000:.2f} meV/Å")

    # Print per-sample results
    print(f"\n  Per-sample results:")
    print(f"  {'#':<4} {'Atoms':<6} {'Valid':<6} {'E err (meV)':<12} {'F MAE (meV/Å)':<14}")
    print(f"  {'-'*4} {'-'*6} {'-'*6} {'-'*12} {'-'*14}")
    for i, r in enumerate(results):
        if r.get("valid"):
            print(f"  {i:<4} {r['n_atoms']:<6} {'✓':<6} {r['energy_error']*1000:<12.2f} {r['force_mae']*1000:<14.2f}")
        else:
            print(f"  {i:<4} {r['n_atoms']:<6} {'✗':<6} {'N/A':<12} {'N/A':<14}")

    return results


if __name__ == "__main__":
    test_vllm_inference()

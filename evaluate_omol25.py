#!/usr/bin/env python3
"""
Evaluate a Qwen3 model trained on moltok tokenization against OMol25 benchmarks.

This script supports:
1. S2EF (Structure to Energy and Forces) evaluation on validation/test splits
2. The 7 specialized chemistry evaluation tasks from the leaderboard

Note: This model does NOT use charge/spin conditioning. For IE/EA and spin_gap
tasks, predictions are based on geometry alone.

Data setup:
    # Download OMol25 validation data (choose one):
    # Full validation (20GB, 2.76M structures):
    curl -L -o val.tar.gz "https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/val.tar.gz"
    # Neutral validation (119MB, 27k structures - faster for testing):
    curl -L -o neutral_val.tar.gz "https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/neutral_val.tar.gz"
    tar -xzf neutral_val.tar.gz -C ./omol_data/

Usage:
    # S2EF evaluation with vLLM (fast, recommended)
    python evaluate_omol25.py \
        --model WillHeld/ToMol-marin-1B \
        --config fp16_config.json \
        --use-vllm \
        --data-path ./omol_data/neutral_val \
        --output predictions_val.npz

    # S2EF evaluation with HuggingFace (slower)
    python evaluate_omol25.py \
        --model WillHeld/ToMol-marin-1B \
        --config fp16_config.json \
        --data-path ./omol_data/neutral_val \
        --output predictions_val.npz

    # Run all evaluations
    python evaluate_omol25.py \
        --model WillHeld/ToMol-marin-1B \
        --config fp16_config.json \
        --use-vllm \
        --run-evals \
        --eval-output-dir eval_results
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from tqdm import tqdm

from serialize_molecules import MoleculeTokenizer

# Delay torch import to avoid CUDA init before vLLM
torch = None

def _get_torch():
    global torch
    if torch is None:
        import torch as _torch
        torch = _torch
    return torch

# Optional imports for specialized evals
FAIRCHEM_AVAILABLE = False
FairChemEvaluator = None
OMOL_EVALS_AVAILABLE = False

def _init_optional_imports():
    global FAIRCHEM_AVAILABLE, FairChemEvaluator, OMOL_EVALS_AVAILABLE
    global conformers, distance_scaling, ie_ea, ligand_pocket, ligand_strain, protonation, spin_gap

    try:
        from fairchem.core.modules.evaluator import Evaluator as _FairChemEvaluator
        FairChemEvaluator = _FairChemEvaluator
        FAIRCHEM_AVAILABLE = True
    except ImportError:
        pass

    try:
        from fairchem.data.omol.evals import (
            conformers as _conformers,
            distance_scaling as _distance_scaling,
            ie_ea as _ie_ea,
            ligand_pocket as _ligand_pocket,
            ligand_strain as _ligand_strain,
            protonation as _protonation,
            spin_gap as _spin_gap,
        )
        conformers = _conformers
        distance_scaling = _distance_scaling
        ie_ea = _ie_ea
        ligand_pocket = _ligand_pocket
        ligand_strain = _ligand_strain
        protonation = _protonation
        spin_gap = _spin_gap
        OMOL_EVALS_AVAILABLE = True
    except ImportError:
        pass


# =============================================================================
# ASE Calculator for Qwen3 Model (no charge/spin conditioning)
# =============================================================================


class Qwen3MolCalculator(Calculator):
    """
    ASE-compatible calculator for Qwen3 models trained on moltok tokenization.

    This calculator predicts energy and forces given atomic numbers and positions.
    It does NOT use charge or spin information - predictions are geometry-only.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model_name_or_path: str,
        config_path: str,
        device: str = "cuda",
        max_new_tokens: int = 512,
        dtype: str = "bfloat16",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device = device
        self.max_new_tokens = max_new_tokens

        torch = _get_torch()
        from transformers import AutoModelForCausalLM

        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        # Load moltok tokenizer with codebook
        self.mol_tokenizer = MoleculeTokenizer(config_path)
        self.vocab_info = self.mol_tokenizer.get_vocab_info()

        # Load HuggingFace tokenizer
        self.hf_tokenizer = self.mol_tokenizer.get_hf_tokenizer()

        # Load Qwen3 model
        print(f"Loading model from {model_name_or_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()

        # Get special token IDs
        self.eos_id = self.vocab_info["special_tokens"]["[EOS]"]
        self.force_id = self.vocab_info["special_tokens"]["[FORCE]"]
        self.pos_end_id = self.vocab_info["special_tokens"]["[POS_END]"]

    def calculate(
        self,
        atoms: Atoms = None,
        properties: list[str] = None,
        system_changes: list = all_changes,
    ):
        """Calculate energy and forces for the given atoms."""
        super().calculate(atoms, properties, system_changes)

        atomic_numbers = atoms.get_atomic_numbers()
        positions = atoms.get_positions()

        # Build input prompt: [BOS] [ATOMS]...[ATOMS_END] [POS]...[POS_END] [FORCE]
        input_tokens = self._build_input_prompt(atomic_numbers, positions)

        # Generate completion
        torch = _get_torch()
        input_ids = torch.tensor([input_tokens], device=self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.hf_tokenizer.pad_token_id or 0,
                eos_token_id=self.eos_id,
            )

        # Parse output tokens
        full_tokens = outputs[0].cpu().numpy().tolist()
        result = self._parse_output(full_tokens, len(atomic_numbers))

        self.results = {
            "energy": result["energy"],
            "forces": result["forces"],
        }

    def _build_input_prompt(
        self,
        atomic_numbers: np.ndarray,
        positions: np.ndarray,
    ) -> list[int]:
        """Build input token sequence: atoms + positions, prompting for forces."""
        # Encode full molecule with dummy forces/energy
        full_tokens = self.mol_tokenizer.encode_molecule(
            atomic_numbers=atomic_numbers.tolist(),
            positions=positions,
            forces=np.zeros_like(positions),
            energy=0.0,
            shuffle_sections=False,
        )

        # Truncate at [FORCE] to prompt model to generate forces + energy
        try:
            force_idx = full_tokens.index(self.force_id)
            return full_tokens[: force_idx + 1]
        except ValueError:
            try:
                pos_end_idx = full_tokens.index(self.pos_end_id)
                return full_tokens[: pos_end_idx + 1] + [self.force_id]
            except ValueError:
                raise ValueError("Could not find position section in tokens")

    def _parse_output(self, tokens: list[int], n_atoms: int) -> dict:
        """Parse model output to extract forces and energy."""
        try:
            decoded = self.mol_tokenizer.decode_molecule(tokens)
            forces = decoded.get("forces")
            energy = decoded.get("energy")

            if forces is None or len(forces) != n_atoms:
                forces = np.zeros((n_atoms, 3))
            if energy is None:
                energy = 0.0

            return {"energy": float(energy), "forces": np.array(forces)}
        except Exception as e:
            print(f"Warning: Failed to parse output: {e}")
            return {"energy": 0.0, "forces": np.zeros((n_atoms, 3))}


# =============================================================================
# vLLM-based Calculator (fast batched inference)
# =============================================================================


class VLLMMolCalculator(Calculator):
    """
    ASE-compatible calculator using vLLM for fast batched inference.

    This calculator supports both single-molecule and batched prediction.
    For best performance, use predict_batch() directly instead of ASE interface.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        model_name_or_path: str,
        config_path: str,
        max_new_tokens: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_new_tokens = max_new_tokens

        # Load moltok tokenizer with codebook
        self.mol_tokenizer = MoleculeTokenizer(config_path)
        self.vocab_info = self.mol_tokenizer.get_vocab_info()
        self.hf_tokenizer = self.mol_tokenizer.get_hf_tokenizer()

        # Get special token IDs
        self.eos_id = self.vocab_info["special_tokens"]["[EOS]"]
        self.force_id = self.vocab_info["special_tokens"]["[FORCE]"]

        # Load vLLM
        from vllm import LLM, SamplingParams

        print(f"Loading model with vLLM from {model_name_or_path}...")
        self.llm = LLM(
            model=model_name_or_path,
            dtype="bfloat16",
            trust_remote_code=True,
        )

        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0,  # greedy
            stop_token_ids=[self.eos_id],
        )
        print("vLLM model loaded!")

    def _build_prompt(self, atomic_numbers: np.ndarray, positions: np.ndarray) -> str:
        """Build input prompt string for a molecule."""
        full_tokens = self.mol_tokenizer.encode_molecule(
            atomic_numbers=atomic_numbers.tolist() if hasattr(atomic_numbers, 'tolist') else list(atomic_numbers),
            positions=positions,
            forces=np.zeros_like(positions),
            energy=0.0,
            shuffle_sections=False,
        )
        force_idx = full_tokens.index(self.force_id)
        input_tokens = full_tokens[:force_idx + 1]
        return self.mol_tokenizer.tokens_to_string(input_tokens)

    def _parse_output(self, prompt: str, generated_text: str, n_atoms: int) -> dict:
        """Parse vLLM output to extract forces and energy."""
        full_text = prompt + " " + generated_text
        token_ids = self.hf_tokenizer.encode(full_text, add_special_tokens=False)

        try:
            decoded = self.mol_tokenizer.decode_molecule(token_ids)
            forces = decoded.get("forces")
            energy = decoded.get("energy")

            if forces is None or len(forces) != n_atoms:
                forces = np.zeros((n_atoms, 3))
            if energy is None:
                energy = 0.0

            return {"energy": float(energy), "forces": np.array(forces)}
        except Exception as e:
            return {"energy": 0.0, "forces": np.zeros((n_atoms, 3))}

    def calculate(
        self,
        atoms: Atoms = None,
        properties: list[str] = None,
        system_changes: list = all_changes,
    ):
        """Calculate energy and forces for a single molecule (ASE interface)."""
        super().calculate(atoms, properties, system_changes)

        atomic_numbers = atoms.get_atomic_numbers()
        positions = atoms.get_positions()
        n_atoms = len(atomic_numbers)

        prompt = self._build_prompt(atomic_numbers, positions)
        outputs = self.llm.generate([prompt], self.sampling_params)
        generated_text = outputs[0].outputs[0].text

        result = self._parse_output(prompt, generated_text, n_atoms)
        self.results = result

    def predict_batch(
        self,
        samples: list[dict],
        show_progress: bool = True,
    ) -> list[dict]:
        """
        Predict energy and forces for a batch of molecules.

        Args:
            samples: List of dicts with 'atomic_numbers' and 'positions'
            show_progress: Show progress bar

        Returns:
            List of dicts with 'energy' and 'forces'
        """
        # Build all prompts
        prompts = []
        n_atoms_list = []
        for sample in samples:
            atomic_numbers = sample["atomic_numbers"]
            positions = sample["positions"]
            n_atoms_list.append(len(atomic_numbers))
            prompts.append(self._build_prompt(atomic_numbers, positions))

        # Run batched inference
        if show_progress:
            print(f"Running vLLM inference on {len(prompts)} samples...")
        outputs = self.llm.generate(prompts, self.sampling_params)

        # Parse results
        results = []
        for i, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            result = self._parse_output(prompts[i], generated_text, n_atoms_list[i])
            results.append(result)

        return results


# =============================================================================
# Metrics Computation (using FAIRChem Evaluator)
# =============================================================================


def compute_s2ef_metrics(
    pred_energies: np.ndarray,
    pred_forces: list[np.ndarray],
    target_energies: np.ndarray,
    target_forces: list[np.ndarray],
    natoms: np.ndarray,
) -> dict:
    """
    Compute S2EF metrics.

    Uses FAIRChem's Evaluator if available, otherwise computes basic metrics.
    Reports energy in meV/atom and forces in meV/Å.
    """
    all_pred_forces = np.concatenate(pred_forces, axis=0)
    all_target_forces = np.concatenate(target_forces, axis=0)

    if FAIRCHEM_AVAILABLE and FairChemEvaluator is not None:
        torch = _get_torch()
        evaluator = FairChemEvaluator(task="s2ef")

        prediction = {
            "energy": torch.tensor(pred_energies, dtype=torch.float32),
            "forces": torch.tensor(all_pred_forces, dtype=torch.float32),
            "natoms": torch.tensor(natoms, dtype=torch.long),
        }
        target = {
            "energy": torch.tensor(target_energies, dtype=torch.float32),
            "forces": torch.tensor(all_target_forces, dtype=torch.float32),
            "natoms": torch.tensor(natoms, dtype=torch.long),
        }

        metrics_raw = evaluator.eval(prediction, target)

        # Convert to meV and meV/Å
        metrics = {
            "energy_mae_meV": float(metrics_raw.get("energy_mae", {}).get("metric", 0)) * 1000,
            "force_mae_meV_A": float(metrics_raw.get("forces_mae", {}).get("metric", 0)) * 1000,
            "forcesx_mae_meV_A": float(metrics_raw.get("forcesx_mae", {}).get("metric", 0)) * 1000,
            "forcesy_mae_meV_A": float(metrics_raw.get("forcesy_mae", {}).get("metric", 0)) * 1000,
            "forcesz_mae_meV_A": float(metrics_raw.get("forcesz_mae", {}).get("metric", 0)) * 1000,
            "force_cosine": float(metrics_raw.get("cosine_similarity", {}).get("metric", 0)),
            "force_magnitude_error_meV_A": float(
                metrics_raw.get("magnitude_error", {}).get("metric", 0)
            ) * 1000,
            "efwt": float(metrics_raw.get("energy_forces_within_threshold", {}).get("metric", 0)),
            "n_structures": len(pred_energies),
            "n_atoms": len(all_pred_forces),
        }
    else:
        # Basic metrics without FAIRChem
        energy_mae = np.mean(np.abs(pred_energies - target_energies))
        force_mae = np.mean(np.abs(all_pred_forces - all_target_forces))

        metrics = {
            "energy_mae_meV": float(energy_mae) * 1000,
            "force_mae_meV_A": float(force_mae) * 1000,
            "n_structures": len(pred_energies),
            "n_atoms": len(all_pred_forces),
        }

    return metrics


def print_metrics_table(metrics: dict) -> None:
    """Print metrics in leaderboard format (meV, meV/Å)."""
    print("\n" + "=" * 60)
    print("S2EF EVALUATION METRICS (OMol25 Leaderboard Format)")
    print("=" * 60)

    print("\nEnergy Metrics:")
    print(f"  {'MAE':<25} {metrics['energy_mae_meV']:>12.2f} meV/atom")

    print("\nForce Metrics:")
    print(f"  {'MAE':<25} {metrics['force_mae_meV_A']:>12.2f} meV/Å")

    # Additional metrics only available with FAIRChem evaluator
    if 'forcesx_mae_meV_A' in metrics:
        print(f"  {'MAE (x-component)':<25} {metrics['forcesx_mae_meV_A']:>12.2f} meV/Å")
        print(f"  {'MAE (y-component)':<25} {metrics['forcesy_mae_meV_A']:>12.2f} meV/Å")
        print(f"  {'MAE (z-component)':<25} {metrics['forcesz_mae_meV_A']:>12.2f} meV/Å")
        print(f"  {'Magnitude error':<25} {metrics['force_magnitude_error_meV_A']:>12.2f} meV/Å")
        print(f"  {'Cosine similarity':<25} {metrics['force_cosine']:>12.4f}")

        print("\nThreshold Metrics:")
        print(f"  {'EFwT (E<20meV, F<30meV/Å)':<25} {metrics['efwt']*100:>11.2f}%")

    print("\nDataset Stats:")
    print(f"  {'Structures':<25} {metrics['n_structures']:>12,}")
    print(f"  {'Total atoms':<25} {metrics['n_atoms']:>12,}")
    print("=" * 60)


# =============================================================================
# Dataset Loading
# =============================================================================


class OMol25Dataset:
    """Dataset for loading OMol25 validation/test structures.

    Downloads data from: https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/
    See: https://huggingface.co/facebook/OMol25/blob/main/DATASET.md

    Usage:
        # First download and extract the data:
        # curl -L -o val.tar.gz "https://dl.fbaipublicfiles.com/opencatalystproject/data/omol/250514/val.tar.gz"
        # tar -xzf val.tar.gz

        dataset = OMol25Dataset(data_path="./val")
    """

    def __init__(
        self,
        data_path: str,
        max_samples: Optional[int] = None,
        load_labels: bool = True,
    ):
        self.data_path = data_path
        self.samples = []
        self.has_labels = load_labels

        print(f"Loading OMol25 data from {data_path}...")

        from fairchem.core.datasets import AseDBDataset

        ase_dataset = AseDBDataset({"src": data_path})
        n_samples = len(ase_dataset)

        if max_samples:
            n_samples = min(n_samples, max_samples)

        print(f"Loading {n_samples} samples...")

        for i in tqdm(range(n_samples), desc="Loading"):
            atoms = ase_dataset.get_atoms(i)

            item = {
                "id": atoms.info.get("source", str(i)),
                "atomic_numbers": atoms.get_atomic_numbers(),
                "positions": atoms.get_positions(),
                "natoms": len(atoms),
            }

            if self.has_labels:
                item["energy_target"] = atoms.get_potential_energy()
                item["forces_target"] = atoms.get_forces()

            self.samples.append(item)

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)


# =============================================================================
# S2EF Evaluation
# =============================================================================


def evaluate_s2ef(
    calculator: Union[Qwen3MolCalculator, "VLLMMolCalculator"],
    dataset: OMol25Dataset,
    output_path: str,
    batch_size: int = 32,
) -> dict:
    """Run S2EF evaluation and save predictions in leaderboard format."""
    print(f"\nRunning S2EF evaluation on {len(dataset)} structures...")
    if dataset.has_labels:
        print("Ground truth labels available - will compute running metrics.")

    samples_list = list(dataset.samples)

    # Check if using vLLM (continuous batching)
    if hasattr(calculator, 'llm'):
        return _evaluate_s2ef_vllm(calculator, samples_list, dataset.has_labels, output_path)
    else:
        return _evaluate_s2ef_sequential(calculator, samples_list, dataset.has_labels, output_path)


def _evaluate_s2ef_vllm(
    calculator: "VLLMMolCalculator",
    samples: list[dict],
    has_labels: bool,
    output_path: str,
) -> dict:
    """Run S2EF with vLLM continuous batching and running metrics."""
    import time

    # Build all prompts upfront
    print("Building prompts...")
    prompts = []
    n_atoms_list = []
    for sample in tqdm(samples, desc="Building prompts"):
        prompt = calculator._build_prompt(sample["atomic_numbers"], sample["positions"])
        prompts.append(prompt)
        n_atoms_list.append(len(sample["atomic_numbers"]))

    # Submit all to vLLM at once - it handles continuous batching internally
    print(f"\nSubmitting {len(prompts)} prompts to vLLM (continuous batching)...")
    start_time = time.time()
    outputs = calculator.llm.generate(prompts, calculator.sampling_params)
    elapsed = time.time() - start_time
    print(f"Inference complete in {elapsed:.1f}s ({len(prompts)/elapsed:.1f} samples/sec)")

    # Parse results with running metrics
    print("\nParsing outputs and computing metrics...")
    all_ids = []
    all_energies = []
    all_forces = []
    all_natoms = []
    all_target_energies = []
    all_target_forces = []

    # Running metrics
    energy_errors = []
    force_errors = []
    last_report = 0
    report_interval = max(1, len(samples) // 20)  # Report ~20 times

    for i, (output, sample) in enumerate(zip(outputs, samples)):
        generated_text = output.outputs[0].text
        result = calculator._parse_output(prompts[i], generated_text, n_atoms_list[i])

        all_ids.append(sample["id"])
        all_energies.append(result["energy"])
        all_forces.append(result["forces"])
        all_natoms.append(sample["natoms"])

        if has_labels:
            target_energy = sample["energy_target"]
            target_forces = sample["forces_target"]
            all_target_energies.append(target_energy)
            all_target_forces.append(target_forces)

            # Track errors for running metrics
            energy_errors.append(abs(result["energy"] - target_energy))
            force_errors.append(np.mean(np.abs(result["forces"] - target_forces)))

            # Report running metrics periodically
            if (i + 1) - last_report >= report_interval or i == len(samples) - 1:
                energy_mae = np.mean(energy_errors) * 1000  # meV
                force_mae = np.mean(force_errors) * 1000    # meV/Å
                pct = 100 * (i + 1) / len(samples)
                print(f"  [{i+1:>6}/{len(samples)}] ({pct:5.1f}%) | Energy MAE: {energy_mae:8.2f} meV | Force MAE: {force_mae:8.2f} meV/Å")
                last_report = i + 1

    # Save predictions
    all_forces_concat = np.concatenate(all_forces, axis=0)
    np.savez_compressed(
        output_path,
        ids=np.array(all_ids),
        energy=np.array(all_energies),
        forces=all_forces_concat,
        natoms=np.array(all_natoms),
    )
    print(f"\nPredictions saved to {output_path}")

    result = {
        "n_structures": len(all_ids),
        "n_force_vectors": len(all_forces_concat),
        "output_path": output_path,
    }

    # Final metrics
    if has_labels and all_target_energies:
        metrics = compute_s2ef_metrics(
            pred_energies=np.array(all_energies),
            pred_forces=all_forces,
            target_energies=np.array(all_target_energies),
            target_forces=all_target_forces,
            natoms=np.array(all_natoms),
        )
        print_metrics_table(metrics)
        result["metrics"] = metrics

        metrics_path = output_path.replace(".npz", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {metrics_path}")

    return result


def _evaluate_s2ef_sequential(
    calculator: Qwen3MolCalculator,
    samples: list[dict],
    has_labels: bool,
    output_path: str,
) -> dict:
    """Run S2EF with sequential inference (HuggingFace backend)."""
    all_ids = []
    all_energies = []
    all_forces = []
    all_natoms = []
    all_target_energies = []
    all_target_forces = []

    energy_errors = []
    force_errors = []
    last_report = 0
    report_interval = max(1, len(samples) // 20)

    print("Using sequential inference...")
    for i, sample in enumerate(tqdm(samples, desc="Predicting")):
        atoms = Atoms(
            numbers=sample["atomic_numbers"],
            positions=sample["positions"],
        )
        atoms.calc = calculator

        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()

        all_ids.append(sample["id"])
        all_energies.append(energy)
        all_forces.append(forces)
        all_natoms.append(sample["natoms"])

        if has_labels:
            target_energy = sample["energy_target"]
            target_forces = sample["forces_target"]
            all_target_energies.append(target_energy)
            all_target_forces.append(target_forces)

            energy_errors.append(abs(energy - target_energy))
            force_errors.append(np.mean(np.abs(forces - target_forces)))

            if (i + 1) - last_report >= report_interval or i == len(samples) - 1:
                energy_mae = np.mean(energy_errors) * 1000
                force_mae = np.mean(force_errors) * 1000
                pct = 100 * (i + 1) / len(samples)
                print(f"  [{i+1:>6}/{len(samples)}] ({pct:5.1f}%) | Energy MAE: {energy_mae:8.2f} meV | Force MAE: {force_mae:8.2f} meV/Å")
                last_report = i + 1

    # Save predictions
    all_forces_concat = np.concatenate(all_forces, axis=0)
    np.savez_compressed(
        output_path,
        ids=np.array(all_ids),
        energy=np.array(all_energies),
        forces=all_forces_concat,
        natoms=np.array(all_natoms),
    )
    print(f"\nPredictions saved to {output_path}")

    result = {
        "n_structures": len(all_ids),
        "n_force_vectors": len(all_forces_concat),
        "output_path": output_path,
    }

    if has_labels and all_target_energies:
        metrics = compute_s2ef_metrics(
            pred_energies=np.array(all_energies),
            pred_forces=all_forces,
            target_energies=np.array(all_target_energies),
            target_forces=all_target_forces,
            natoms=np.array(all_natoms),
        )
        print_metrics_table(metrics)
        result["metrics"] = metrics

        metrics_path = output_path.replace(".npz", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {metrics_path}")

    return result


# =============================================================================
# Specialized Evaluation Tasks
# =============================================================================


def run_conformers_eval(calculator: Calculator, output_dir: Path) -> dict:
    """
    Conformers evaluation: identify lowest energy conformer.

    For each molecule, predict energies for all conformers and check if
    the lowest predicted energy matches the lowest DFT energy.
    """
    print("Loading conformers evaluation data...")
    eval_data = conformers.load_eval_data()

    correct = 0
    total = 0
    all_results = []

    for mol_id, conf_list in tqdm(eval_data.items(), desc="Conformers"):
        pred_energies = []
        target_energies = []

        for conf in conf_list:
            atoms = conf["atoms"]
            atoms.calc = calculator
            pred_e = atoms.get_potential_energy()
            target_e = conf["energy"]

            pred_energies.append(pred_e)
            target_energies.append(target_e)

        # Check if predicted lowest matches target lowest
        pred_lowest = np.argmin(pred_energies)
        target_lowest = np.argmin(target_energies)

        if pred_lowest == target_lowest:
            correct += 1
        total += 1

        all_results.append({
            "mol_id": mol_id,
            "pred_lowest_idx": int(pred_lowest),
            "target_lowest_idx": int(target_lowest),
            "correct": pred_lowest == target_lowest,
        })

    accuracy = correct / total if total > 0 else 0.0

    result = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "details": all_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "conformers_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Conformers accuracy: {accuracy*100:.2f}% ({correct}/{total})")
    return result


def run_distance_scaling_eval(calculator: Calculator, output_dir: Path) -> dict:
    """
    Distance scaling evaluation: predict energy vs intermolecular distance.

    Tests how well the model captures non-bonded interactions.
    """
    print("Loading distance scaling evaluation data...")
    eval_data = distance_scaling.load_eval_data()

    all_results = []
    total_mae = 0.0
    count = 0

    for system_id, distance_data in tqdm(eval_data.items(), desc="Distance scaling"):
        pred_energies = []
        target_energies = []
        distances = []

        for point in distance_data:
            atoms = point["atoms"]
            atoms.calc = calculator
            pred_e = atoms.get_potential_energy()

            pred_energies.append(pred_e)
            target_energies.append(point["energy"])
            distances.append(point["distance"])

        # Compute MAE for this system
        mae = np.mean(np.abs(np.array(pred_energies) - np.array(target_energies)))
        total_mae += mae
        count += 1

        all_results.append({
            "system_id": system_id,
            "distances": distances,
            "pred_energies": pred_energies,
            "target_energies": target_energies,
            "mae_eV": float(mae),
        })

    avg_mae = total_mae / count if count > 0 else 0.0

    result = {
        "mae_eV": avg_mae,
        "mae_meV": avg_mae * 1000,
        "n_systems": count,
        "details": all_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "distance_scaling_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Distance scaling MAE: {avg_mae*1000:.2f} meV")
    return result


def run_spin_gap_eval(calculator: Calculator, output_dir: Path) -> dict:
    """
    Spin gap evaluation: predict singlet-triplet energy gaps.

    Note: This model doesn't use spin conditioning, so predictions are
    based purely on geometry differences between spin states.
    """
    print("Loading spin gap evaluation data...")
    print("WARNING: Model has no spin conditioning - using geometry only")
    eval_data = spin_gap.load_eval_data()

    all_results = []
    pred_gaps = []
    target_gaps = []

    for mol_id, spin_data in tqdm(eval_data.items(), desc="Spin gap"):
        singlet_atoms = spin_data["singlet"]["atoms"]
        triplet_atoms = spin_data["triplet"]["atoms"]

        singlet_atoms.calc = calculator
        triplet_atoms.calc = calculator

        pred_singlet = singlet_atoms.get_potential_energy()
        pred_triplet = triplet_atoms.get_potential_energy()
        pred_gap = pred_triplet - pred_singlet

        target_gap = spin_data["triplet"]["energy"] - spin_data["singlet"]["energy"]

        pred_gaps.append(pred_gap)
        target_gaps.append(target_gap)

        all_results.append({
            "mol_id": mol_id,
            "pred_gap_eV": float(pred_gap),
            "target_gap_eV": float(target_gap),
            "error_eV": float(pred_gap - target_gap),
        })

    pred_gaps = np.array(pred_gaps)
    target_gaps = np.array(target_gaps)
    mae = np.mean(np.abs(pred_gaps - target_gaps))

    result = {
        "mae_eV": float(mae),
        "mae_meV": float(mae * 1000),
        "n_molecules": len(all_results),
        "note": "Model has no spin conditioning - geometry-only prediction",
        "details": all_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "spin_gap_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Spin gap MAE: {mae*1000:.2f} meV (geometry-only, no spin conditioning)")
    return result


def run_ie_ea_eval(calculator: Calculator, output_dir: Path) -> dict:
    """
    IE/EA evaluation: predict ionization energies and electron affinities.

    Note: This model doesn't use charge conditioning, so predictions are
    based purely on geometry differences between charge states.
    """
    print("Loading IE/EA evaluation data...")
    print("WARNING: Model has no charge conditioning - using geometry only")
    eval_data = ie_ea.load_eval_data()

    ie_results = []
    ea_results = []
    ie_errors = []
    ea_errors = []

    for mol_id, charge_data in tqdm(eval_data.items(), desc="IE/EA"):
        neutral_atoms = charge_data["neutral"]["atoms"]
        neutral_atoms.calc = calculator
        pred_neutral = neutral_atoms.get_potential_energy()
        target_neutral = charge_data["neutral"]["energy"]

        # Ionization energy (neutral -> cation)
        if "cation" in charge_data:
            cation_atoms = charge_data["cation"]["atoms"]
            cation_atoms.calc = calculator
            pred_cation = cation_atoms.get_potential_energy()

            pred_ie = pred_cation - pred_neutral
            target_ie = charge_data["cation"]["energy"] - target_neutral

            ie_results.append({
                "mol_id": mol_id,
                "pred_ie_eV": float(pred_ie),
                "target_ie_eV": float(target_ie),
                "error_eV": float(pred_ie - target_ie),
            })
            ie_errors.append(abs(pred_ie - target_ie))

        # Electron affinity (neutral -> anion)
        if "anion" in charge_data:
            anion_atoms = charge_data["anion"]["atoms"]
            anion_atoms.calc = calculator
            pred_anion = anion_atoms.get_potential_energy()

            pred_ea = pred_neutral - pred_anion
            target_ea = target_neutral - charge_data["anion"]["energy"]

            ea_results.append({
                "mol_id": mol_id,
                "pred_ea_eV": float(pred_ea),
                "target_ea_eV": float(target_ea),
                "error_eV": float(pred_ea - target_ea),
            })
            ea_errors.append(abs(pred_ea - target_ea))

    ie_mae = np.mean(ie_errors) if ie_errors else 0.0
    ea_mae = np.mean(ea_errors) if ea_errors else 0.0

    result = {
        "ie_mae_eV": float(ie_mae),
        "ie_mae_meV": float(ie_mae * 1000),
        "ea_mae_eV": float(ea_mae),
        "ea_mae_meV": float(ea_mae * 1000),
        "n_ie": len(ie_results),
        "n_ea": len(ea_results),
        "note": "Model has no charge conditioning - geometry-only prediction",
        "ie_details": ie_results,
        "ea_details": ea_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "ie_ea_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"IE MAE: {ie_mae*1000:.2f} meV (geometry-only)")
    print(f"EA MAE: {ea_mae*1000:.2f} meV (geometry-only)")
    return result


def run_protonation_eval(calculator: Calculator, output_dir: Path) -> dict:
    """
    Protonation evaluation: predict pKa-related energy differences.

    Compares protonated vs deprotonated forms of molecules.
    """
    print("Loading protonation evaluation data...")
    eval_data = protonation.load_eval_data()

    all_results = []
    errors = []

    for mol_id, prot_data in tqdm(eval_data.items(), desc="Protonation"):
        protonated_atoms = prot_data["protonated"]["atoms"]
        deprotonated_atoms = prot_data["deprotonated"]["atoms"]

        protonated_atoms.calc = calculator
        deprotonated_atoms.calc = calculator

        pred_prot = protonated_atoms.get_potential_energy()
        pred_deprot = deprotonated_atoms.get_potential_energy()
        pred_delta = pred_prot - pred_deprot

        target_delta = prot_data["protonated"]["energy"] - prot_data["deprotonated"]["energy"]

        all_results.append({
            "mol_id": mol_id,
            "pred_delta_eV": float(pred_delta),
            "target_delta_eV": float(target_delta),
            "error_eV": float(pred_delta - target_delta),
        })
        errors.append(abs(pred_delta - target_delta))

    mae = np.mean(errors) if errors else 0.0

    result = {
        "mae_eV": float(mae),
        "mae_meV": float(mae * 1000),
        "n_molecules": len(all_results),
        "details": all_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "protonation_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Protonation MAE: {mae*1000:.2f} meV")
    return result


def run_ligand_strain_eval(calculator: Calculator, output_dir: Path) -> dict:
    """
    Ligand strain evaluation: compare bound vs relaxed ligand energies.
    """
    print("Loading ligand strain evaluation data...")
    eval_data = ligand_strain.load_eval_data()

    all_results = []
    errors = []

    for lig_id, strain_data in tqdm(eval_data.items(), desc="Ligand strain"):
        bound_atoms = strain_data["bound"]["atoms"]
        relaxed_atoms = strain_data["relaxed"]["atoms"]

        bound_atoms.calc = calculator
        relaxed_atoms.calc = calculator

        pred_bound = bound_atoms.get_potential_energy()
        pred_relaxed = relaxed_atoms.get_potential_energy()
        pred_strain = pred_bound - pred_relaxed

        target_strain = strain_data["bound"]["energy"] - strain_data["relaxed"]["energy"]

        all_results.append({
            "ligand_id": lig_id,
            "pred_strain_eV": float(pred_strain),
            "target_strain_eV": float(target_strain),
            "error_eV": float(pred_strain - target_strain),
        })
        errors.append(abs(pred_strain - target_strain))

    mae = np.mean(errors) if errors else 0.0

    result = {
        "mae_eV": float(mae),
        "mae_meV": float(mae * 1000),
        "n_ligands": len(all_results),
        "details": all_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "ligand_strain_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Ligand strain MAE: {mae*1000:.2f} meV")
    return result


def run_ligand_pocket_eval(calculator: Calculator, output_dir: Path) -> dict:
    """
    Ligand pocket evaluation: protein-ligand interaction energies.
    """
    print("Loading ligand pocket evaluation data...")
    eval_data = ligand_pocket.load_eval_data()

    all_results = []
    errors = []

    for complex_id, pocket_data in tqdm(eval_data.items(), desc="Ligand pocket"):
        complex_atoms = pocket_data["complex"]["atoms"]
        complex_atoms.calc = calculator
        pred_energy = complex_atoms.get_potential_energy()
        target_energy = pocket_data["complex"]["energy"]

        error = abs(pred_energy - target_energy)
        errors.append(error)

        all_results.append({
            "complex_id": complex_id,
            "pred_energy_eV": float(pred_energy),
            "target_energy_eV": float(target_energy),
            "error_eV": float(error),
        })

    mae = np.mean(errors) if errors else 0.0

    result = {
        "mae_eV": float(mae),
        "mae_meV": float(mae * 1000),
        "n_complexes": len(all_results),
        "details": all_results,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "ligand_pocket_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"Ligand pocket MAE: {mae*1000:.2f} meV")
    return result


def run_specialized_evals(
    calculator: Union[Qwen3MolCalculator, "VLLMMolCalculator"],
    tasks: list[str],
    output_dir: str,
) -> dict:
    """Run the specialized chemistry evaluation tasks."""
    if not OMOL_EVALS_AVAILABLE:
        print("WARNING: fairchem.data.omol.evals not available, skipping specialized evals")
        print("Install with: pip install fairchem-data-omol")
        return {"error": "fairchem.data.omol.evals not available"}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_runners = {
        "conformers": run_conformers_eval,
        "distance_scaling": run_distance_scaling_eval,
        "spin_gap": run_spin_gap_eval,
        "ie_ea": run_ie_ea_eval,
        "protonation": run_protonation_eval,
        "ligand_strain": run_ligand_strain_eval,
        "ligand_pocket": run_ligand_pocket_eval,
    }

    results = {}

    for task in tasks:
        if task not in task_runners:
            print(f"Unknown task: {task}. Available: {list(task_runners.keys())}")
            continue

        print(f"\n{'='*60}")
        print(f"Running {task} evaluation...")
        print(f"{'='*60}")

        try:
            task_result = task_runners[task](calculator, output_dir / task)
            results[task] = task_result
        except Exception as e:
            print(f"  Error running {task}: {e}")
            results[task] = {"error": str(e)}

    # Save summary
    summary_path = output_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    for task, res in results.items():
        if "error" in res:
            print(f"  {task}: ERROR - {res['error']}")
        elif "accuracy" in res:
            print(f"  {task}: {res['accuracy']*100:.2f}% accuracy")
        elif "mae_meV" in res:
            print(f"  {task}: {res['mae_meV']:.2f} meV MAE")
        elif "ie_mae_meV" in res:
            print(f"  {task}: IE={res['ie_mae_meV']:.2f} meV, EA={res['ea_mae_meV']:.2f} meV")

    print(f"\nFull results saved to {summary_path}")
    return results


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3 model on OMol25 benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="fp16_config.json",
        help="Path to FP16 tokenizer config JSON file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (for HuggingFace backend)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Use vLLM for fast batched inference (recommended)",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default="./hf_cache",
        help="Directory for HuggingFace cache (default: ./hf_cache)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for vLLM inference",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./omol_data/neutral_val",
        help="Path to OMol25 data directory (aselmdb files)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples for S2EF (for testing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.npz",
        help="Output path for S2EF predictions",
    )
    parser.add_argument(
        "--run-evals",
        action="store_true",
        help="Run specialized chemistry evaluations",
    )
    parser.add_argument(
        "--eval-tasks",
        type=str,
        nargs="+",
        default=[
            "conformers",
            "distance_scaling",
            "protonation",
            "ligand_strain",
            "ligand_pocket",
            "ie_ea",
            "spin_gap",
        ],
        help="Which specialized tasks to run",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=str,
        default="eval_results",
        help="Output directory for specialized evaluations",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max new tokens to generate per molecule",
    )
    parser.add_argument(
        "--skip-s2ef",
        action="store_true",
        help="Skip S2EF evaluation (only run specialized evals)",
    )

    args = parser.parse_args()

    # Set HuggingFace cache directory
    cache_dir = os.path.abspath(args.hf_cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_CACHE"] = cache_dir

    print("=" * 60)
    print("OMol25 Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Backend: {'vLLM (fast)' if args.use_vllm else 'HuggingFace'}")
    print(f"HF Cache: {cache_dir}")
    print(f"NOTE: This model has NO charge/spin conditioning")
    print("=" * 60)

    # Create calculator
    if args.use_vllm:
        calculator = VLLMMolCalculator(
            model_name_or_path=args.model,
            config_path=args.config,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        calculator = Qwen3MolCalculator(
            model_name_or_path=args.model,
            config_path=args.config,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            dtype=args.dtype,
        )

    results = {}

    # S2EF evaluation
    if not args.skip_s2ef:
        print(f"\n{'='*60}")
        print(f"S2EF Evaluation ({args.data_path})")
        print(f"{'='*60}")

        dataset = OMol25Dataset(
            data_path=args.data_path,
            max_samples=args.max_samples,
        )

        s2ef_results = evaluate_s2ef(
            calculator=calculator,
            dataset=dataset,
            output_path=args.output,
            batch_size=args.batch_size,
        )
        results["s2ef"] = s2ef_results

    # Specialized evaluations
    if args.run_evals:
        eval_results = run_specialized_evals(
            calculator=calculator,
            tasks=args.eval_tasks,
            output_dir=args.eval_output_dir,
        )
        results["specialized"] = eval_results

    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}")

    if "s2ef" in results:
        print(f"\nS2EF predictions: {results['s2ef']['output_path']}")

    if "specialized" in results:
        print(f"\nSpecialized eval results: {args.eval_output_dir}/")

    return results


if __name__ == "__main__":
    main()

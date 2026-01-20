#!/usr/bin/env python3
"""
Evaluate a Qwen3 model trained on moltok tokenization against OMol25 benchmarks.

This script supports:
1. S2EF (Structure to Energy and Forces) evaluation on validation/test splits
2. The 7 specialized chemistry evaluation tasks from the leaderboard

Note: This model does NOT use charge/spin conditioning. For IE/EA and spin_gap
tasks, predictions are based on geometry alone.

Usage:
    # S2EF evaluation on validation set
    python evaluate_omol25.py \
        --model WillHeld/qwen3-omol \
        --codebook codebook_mol_1m.pkl \
        --split val \
        --output predictions_val.npz

    # Run all evaluations
    python evaluate_omol25.py \
        --model WillHeld/qwen3-omol \
        --codebook codebook_mol_1m.pkl \
        --run-evals \
        --eval-output-dir eval_results
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from datasets import load_dataset
from fairchem.core.modules.evaluator import Evaluator as FairChemEvaluator
from fairchem.data.omol.evals import (
    conformers,
    distance_scaling,
    ie_ea,
    ligand_pocket,
    ligand_strain,
    protonation,
    spin_gap,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from serialize_molecules import MoleculeTokenizer


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
        codebook_path: str,
        device: str = "cuda",
        max_new_tokens: int = 512,
        dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device = device
        self.max_new_tokens = max_new_tokens

        # Load moltok tokenizer with codebook
        self.mol_tokenizer = MoleculeTokenizer(codebook_path)
        self.vocab_info = self.mol_tokenizer.get_vocab_info()

        # Load HuggingFace tokenizer
        self.hf_tokenizer = self.mol_tokenizer.get_hf_tokenizer()

        # Load Qwen3 model
        print(f"Loading model from {model_name_or_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
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
    Compute S2EF metrics using FAIRChem's official Evaluator.

    Reports energy in meV/atom and forces in meV/Å.
    """
    evaluator = FairChemEvaluator(task="s2ef")

    all_pred_forces = np.concatenate(pred_forces, axis=0)
    all_target_forces = np.concatenate(target_forces, axis=0)

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
    """Dataset for loading OMol25 validation/test structures."""

    def __init__(
        self,
        split: str = "val",
        max_samples: Optional[int] = None,
        load_labels: bool = True,
    ):
        self.split = split
        self.samples = []
        self.has_labels = load_labels and split == "val"

        print(f"Loading OMol25 {split} split from HuggingFace...")

        split_name = "validation" if split == "val" else "test"
        ds = load_dataset("facebook/OMol25", split=split_name, streaming=True)

        for i, sample in enumerate(tqdm(ds, desc=f"Loading {split}")):
            if max_samples and i >= max_samples:
                break

            item = {
                "id": sample.get("id", str(i)),
                "atomic_numbers": np.array(sample["atomic_numbers"]),
                "positions": np.array(sample["positions"]),
                "natoms": len(sample["atomic_numbers"]),
            }

            if self.has_labels:
                item["energy_target"] = sample.get("energy", sample.get("total_energy"))
                item["forces_target"] = np.array(sample.get("forces", []))

            self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)


# =============================================================================
# S2EF Evaluation
# =============================================================================


def evaluate_s2ef(
    calculator: Qwen3MolCalculator,
    dataset: OMol25Dataset,
    output_path: str,
) -> dict:
    """Run S2EF evaluation and save predictions in leaderboard format."""
    all_ids = []
    all_energies = []
    all_forces = []
    all_natoms = []
    all_target_energies = []
    all_target_forces = []

    print(f"\nRunning S2EF evaluation on {len(dataset)} structures...")
    if dataset.has_labels:
        print("Ground truth labels available - will compute metrics.")

    for sample in tqdm(dataset, desc="Predicting"):
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

        if dataset.has_labels:
            all_target_energies.append(sample["energy_target"])
            all_target_forces.append(sample["forces_target"])

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

    # Compute metrics if labels available
    if dataset.has_labels and all_target_energies:
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
    calculator: Qwen3MolCalculator,
    tasks: list[str],
    output_dir: str,
) -> dict:
    """Run the specialized chemistry evaluation tasks."""
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
        "--codebook",
        type=str,
        default="codebook_mol_1m.pkl",
        help="Path to moltok codebook pickle file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split for S2EF evaluation",
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
        default=512,
        help="Max new tokens to generate per molecule",
    )
    parser.add_argument(
        "--skip-s2ef",
        action="store_true",
        help="Skip S2EF evaluation (only run specialized evals)",
    )

    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    print("=" * 60)
    print("OMol25 Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"NOTE: This model has NO charge/spin conditioning")
    print("=" * 60)

    calculator = Qwen3MolCalculator(
        model_name_or_path=args.model,
        codebook_path=args.codebook,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        dtype=dtype_map[args.dtype],
    )

    results = {}

    # S2EF evaluation
    if not args.skip_s2ef:
        print(f"\n{'='*60}")
        print(f"S2EF Evaluation ({args.split} split)")
        print(f"{'='*60}")

        dataset = OMol25Dataset(
            split=args.split,
            max_samples=args.max_samples,
        )

        s2ef_results = evaluate_s2ef(
            calculator=calculator,
            dataset=dataset,
            output_path=args.output,
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

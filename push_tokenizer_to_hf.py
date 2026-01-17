"""Push MoleculeTokenizer to HuggingFace Hub."""

import argparse
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from serialize_molecules import MoleculeTokenizer


def main():
    parser = argparse.ArgumentParser(description="Push tokenizer to HuggingFace Hub")
    parser.add_argument("--repo-id", "-r", default="WillHeld/marin-tomol",
                        help="HuggingFace repo ID")
    parser.add_argument("--codebook", "-c", default="codebook_mol_1m.pkl",
                        help="Path to codebook pickle file")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.codebook}...")
    tokenizer = MoleculeTokenizer(args.codebook)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer_path = Path(tmpdir) / "tokenizer"
        tokenizer.save_hf_tokenizer(str(tokenizer_path))

        print(f"Pushing to {args.repo_id}...")
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=str(tokenizer_path),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Add tokenizer",
        )

    print(f"Done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()

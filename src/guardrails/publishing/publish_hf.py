from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--org", default="governanceai")
    parser.add_argument(
        "--dataset-repo",
        default="governance-ai-guardrail-es-dataset",
    )
    parser.add_argument(
        "--model-repo",
        default="governance-ai-guardrail-qwen25-1_5b-v3-corrective",
    )
    parser.add_argument(
        "--dataset-folder",
        default=str(ROOT / "hf_publish" / "dataset-governance-ai-guardrail-es"),
    )
    parser.add_argument(
        "--model-folder",
        default=str(ROOT / "hf_publish" / "model-governance-ai-guardrail-qwen25-1_5b-v3-corrective"),
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    dataset_repo_id = f"{args.org}/{args.dataset_repo}"
    model_repo_id = f"{args.org}/{args.model_repo}"

    print(f"Creating or reusing dataset repo: {dataset_repo_id}")
    api.create_repo(repo_id=dataset_repo_id, repo_type="dataset", private=args.private, exist_ok=True)

    print(f"Creating or reusing model repo: {model_repo_id}")
    api.create_repo(repo_id=model_repo_id, repo_type="model", private=args.private, exist_ok=True)

    print(f"Uploading dataset folder: {args.dataset_folder}")
    api.upload_large_folder(
        repo_id=dataset_repo_id,
        repo_type="dataset",
        folder_path=args.dataset_folder,
    )

    print(f"Uploading model folder: {args.model_folder}")
    api.upload_large_folder(
        repo_id=model_repo_id,
        repo_type="model",
        folder_path=args.model_folder,
    )

    print("Done.")


if __name__ == "__main__":
    main()

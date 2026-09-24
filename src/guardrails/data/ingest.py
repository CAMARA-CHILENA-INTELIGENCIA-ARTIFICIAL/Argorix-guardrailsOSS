from __future__ import annotations

import json
import argparse
from pathlib import Path

from datasets import load_dataset


DEFAULT_DATASET_ID = "J1N2/mix-prompt-injection-dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Descarga un dataset de Hugging Face a parquet local.")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--token", default=None, help="Token HF opcional; por defecto usa el login local.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / args.dataset_id.replace("/", "__")
    output_dir.mkdir(parents=True, exist_ok=True)

    load_kwargs = {}
    if args.token:
        load_kwargs["token"] = args.token
    dataset = load_dataset(args.dataset_id, **load_kwargs)

    summary = {
        "dataset_id": args.dataset_id,
        "output_dir": str(output_dir),
        "splits": {},
    }

    for split_name, split in dataset.items():
        output_path = output_dir / f"{split_name}.parquet"
        split.to_parquet(output_path)

        summary["splits"][split_name] = {
            "rows": split.num_rows,
            "columns": split.column_names,
            "file": output_path.name,
        }

        print(f"{split_name}: {split.num_rows} filas -> {output_path}")
        print(split.to_pandas().head())

    info_path = output_dir / "dataset_info.json"
    info_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Resumen -> {info_path}")


if __name__ == "__main__":
    main()

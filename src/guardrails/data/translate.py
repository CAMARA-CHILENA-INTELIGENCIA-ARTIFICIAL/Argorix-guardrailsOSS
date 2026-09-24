from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.utils import logging as hf_logging


DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"

#: Cuántos textos se truncaron al traducir. Se registra en dataset_info.json:
#: el fragmento conservado puede no justificar ya la etiqueta heredada.
TRUNCATION_STATS: dict[str, int] = {"total": 0, "truncated": 0}
TARGET_LANG = "spa_Latn"
hf_logging.set_verbosity_error()

LANGUAGE_TO_NLLB = {
    "ar": "arb_Arab",
    "ce": "rus_Cyrl",
    "cs": "ces_Latn",
    "de": "deu_Latn",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "hi": "hin_Deva",
    "hu": "hun_Latn",
    "id": "ind_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ka": "kat_Geor",
    "ko": "kor_Hang",
    "ms": "zsm_Latn",
    "ru": "rus_Cyrl",
    "sr": "srp_Cyrl",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "zh": "zho_Hans",
}


@dataclass(frozen=True)
class DatasetJob:
    name: str
    input_path: Path
    output_path: Path
    text_columns: tuple[str, ...]
    default_source_lang: str
    language_column: str | None = None


def stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def target_lang_id(tokenizer) -> int:
    if hasattr(tokenizer, "lang_code_to_id"):
        return tokenizer.lang_code_to_id[TARGET_LANG]
    return tokenizer.convert_tokens_to_ids(TARGET_LANG)


def load_translator(model_name: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()
    return tokenizer, model


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".arrow":
        return Dataset.from_file(str(path)).to_pandas()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Formato no soportado: {path}")


def sample_group(group: pd.DataFrame, limit: int, seed_key: str) -> pd.DataFrame:
    if limit <= 0 or len(group) <= limit:
        return group
    return group.sample(n=limit, random_state=stable_seed(seed_key))


def guardrails_primary_label(row: pd.Series) -> str:
    if bool(row.get("SEXUAL", 0)):
        return "SEXUAL"
    if bool(row.get("VIOLENCE", 0)):
        return "VIOLENCE"
    if bool(row.get("HATE SPEECH", 0)):
        return "HATE"
    if bool(row.get("POLITICS", 0)):
        return "POLITICS"
    if bool(row.get("HARMFULNESS", 0)):
        return "HARMFUL"
    return "SAFE"


def necent_primary_label(row: pd.Series) -> str:
    category = str(row.get("category", "")).lower()
    source = str(row.get("source", "")).lower()
    attack = str(row.get("attack_technique", "")).lower()
    if bool(row.get("prompt_adversarial", False)) or "injection" in source or "injection" in attack:
        return "PROMPT_INJECTION"
    if "jailbreak" in source or "jailbreak" in attack:
        return "JAILBREAK"
    if "sexual" in category:
        return "SEXUAL"
    if "violence" in category or "violent" in category:
        return "VIOLENCE"
    if "hate" in category or "harassment" in category or "discrimination" in category:
        return "HATE"
    if "politic" in category:
        return "POLITICS"
    if bool(row.get("is_dangerous", False)) or bool(row.get("prompt_harmful", False)):
        return "HARMFUL"
    return "SAFE"


def bogdanminko_primary_label(row: pd.Series) -> str:
    label = str(row.get("type", "")).strip().lower().replace("-", "_")
    if "jailbreak" in label:
        return "JAILBREAK"
    if "inject" in label:
        return "PROMPT_INJECTION"
    return "SAFE"


def select_for_finetune(
    df: pd.DataFrame,
    job: DatasetJob,
    max_rows_per_label: int,
    max_safe_rows: int,
) -> pd.DataFrame:
    if max_rows_per_label <= 0:
        return df

    selected: list[pd.DataFrame] = []

    if job.name.startswith("guardrails/"):
        labels = df.apply(guardrails_primary_label, axis=1)
        for label_name, group in df.groupby(labels, sort=False):
            limit = max_safe_rows if label_name == "SAFE" else max_rows_per_label
            selected.append(sample_group(group, limit, f"{job.name}:{label_name}"))
    elif job.name.startswith("Necent/"):
        labels = df.apply(necent_primary_label, axis=1)
        for label_name, group in df.groupby(labels, sort=False):
            limit = max_safe_rows if label_name == "SAFE" else max_rows_per_label
            selected.append(sample_group(group, limit, f"{job.name}:{label_name}"))
    elif job.name.startswith("J1N2/") and "label" in df.columns:
        for label_value, group in df.groupby(df["label"].astype(bool), sort=False):
            label_name = "PROMPT_INJECTION" if label_value else "SAFE"
            limit = max_rows_per_label if label_value else max_safe_rows
            selected.append(sample_group(group, limit, f"{job.name}:{label_name}"))
    elif job.name.startswith("bogdanminko/"):
        labels = df.apply(bogdanminko_primary_label, axis=1)
        for label_name, group in df.groupby(labels, sort=False):
            limit = max_safe_rows if label_name == "SAFE" else max_rows_per_label
            selected.append(sample_group(group, limit, f"{job.name}:{label_name}"))
    else:
        selected.append(sample_group(df, max_rows_per_label, job.name))

    return pd.concat(selected).sort_index().copy()


def discover_jobs(data_dir: Path, output_dir: Path) -> list[DatasetJob]:
    jobs: list[DatasetJob] = []

    guardrails_output_dir = output_dir / "huyhoangdinhcong__guardrails-dataset-full"
    for input_path in sorted(data_dir.glob("guardrails-dataset-full-*.arrow")):
        split_name = input_path.stem.replace("guardrails-dataset-full-", "")
        jobs.append(
            DatasetJob(
                name=f"guardrails/{split_name}",
                input_path=input_path,
                output_path=guardrails_output_dir / f"{split_name}.parquet",
                text_columns=("text",),
                default_source_lang="vie_Latn",
            )
        )

    necent_path = data_dir / "Necent__llm-jailbreak-prompt-injection-dataset" / "train.parquet"
    if necent_path.exists():
        jobs.append(
            DatasetJob(
                name="Necent/llm-jailbreak-prompt-injection-dataset/train",
                input_path=necent_path,
                output_path=output_dir
                / "Necent__llm-jailbreak-prompt-injection-dataset"
                / "train.parquet",
                text_columns=("prompt", "response"),
                default_source_lang="eng_Latn",
                language_column="language",
            )
        )

    j1n2_path = data_dir / "J1N2__mix-prompt-injection-dataset" / "train.parquet"
    if j1n2_path.exists():
        jobs.append(
            DatasetJob(
                name="J1N2/mix-prompt-injection-dataset/train",
                input_path=j1n2_path,
                output_path=output_dir / "J1N2__mix-prompt-injection-dataset" / "train.parquet",
                text_columns=("prompt", "base_prompt"),
                default_source_lang="eng_Latn",
            )
        )

    bogdanminko_path = (
        data_dir / "bogdanminko__Catch_the_prompt_injection_or_jailbreak_or_benign" / "train.parquet"
    )
    if bogdanminko_path.exists():
        jobs.append(
            DatasetJob(
                name="bogdanminko/Catch_the_prompt_injection_or_jailbreak_or_benign/train",
                input_path=bogdanminko_path,
                output_path=output_dir
                / "bogdanminko__Catch_the_prompt_injection_or_jailbreak_or_benign"
                / "train.parquet",
                text_columns=("prompt",),
                default_source_lang="eng_Latn",
            )
        )

    return jobs


def source_lang_for_row(row: pd.Series, job: DatasetJob) -> str:
    if job.language_column is None:
        return job.default_source_lang

    raw_lang = str(row.get(job.language_column, "")).strip().lower()
    if not raw_lang:
        return job.default_source_lang
    return LANGUAGE_TO_NLLB.get(raw_lang, job.default_source_lang)


def translate_texts(
    texts: list[str],
    src_lang: str,
    tokenizer,
    model,
    device: torch.device,
    max_input_tokens: int,
    max_new_tokens: int,
) -> list[str]:
    if src_lang == TARGET_LANG:
        return texts

    if hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = src_lang

    sin_truncar = tokenizer(texts, padding=False, truncation=False)["input_ids"]
    truncados = sum(1 for ids in sin_truncar if len(ids) > max_input_tokens)
    if truncados:
        # El texto se trunca pero la etiqueta viene del original completo: puede
        # dejar de estar justificada por el fragmento que sobrevive.
        TRUNCATION_STATS["truncated"] += truncados
    TRUNCATION_STATS["total"] += len(texts)

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_tokens,
    ).to(device)

    with torch.inference_mode():
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=target_lang_id(tokenizer),
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
        )

    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)


def flush_checkpoint(
    df: pd.DataFrame,
    translated_columns: dict[str, list[str]],
    row_count: int,
    checkpoint_path: Path,
) -> None:
    partial = df.iloc[:row_count].copy()
    for column, translated_values in translated_columns.items():
        partial[f"{column}_es"] = translated_values[:row_count]
    partial.to_parquet(checkpoint_path, index=False)


def load_checkpoint(
    checkpoint_path: Path,
    text_columns: tuple[str, ...],
) -> tuple[int, dict[str, list[str]]]:
    if not checkpoint_path.exists():
        return 0, {column: [] for column in text_columns}

    partial = pd.read_parquet(checkpoint_path)
    translated_columns = {
        column: partial.get(f"{column}_es", pd.Series(dtype=str)).fillna("").astype(str).tolist()
        for column in text_columns
    }
    row_count = min((len(values) for values in translated_columns.values()), default=0)
    return row_count, translated_columns


def translate_job(
    job: DatasetJob,
    tokenizer,
    model,
    model_name: str,
    device: torch.device,
    batch_size: int,
    checkpoint_every: int,
    max_input_tokens: int,
    max_new_tokens: int,
    limit: int | None,
    overwrite: bool,
    profile: str,
    max_rows_per_label: int,
    max_safe_rows: int,
) -> None:
    checkpoint_path = job.output_path.with_suffix(".partial.parquet")

    df = read_table(job.input_path)
    if limit is not None:
        df = df.head(limit).copy()
    elif profile == "finetune":
        before = len(df)
        df = select_for_finetune(df, job, max_rows_per_label, max_safe_rows)
        print(f"Seleccion {job.name}: {before} -> {len(df)} filas para fine-tuning.")

    if job.output_path.exists() and not overwrite and not checkpoint_path.exists():
        try:
            parquet_file = pq.ParquetFile(job.output_path)
            output_rows = parquet_file.metadata.num_rows
            existing_columns = set(parquet_file.schema.names)
        except Exception:
            output_rows = -1
            existing_columns = set()
        expected_columns = {f"{column}_es" for column in job.text_columns}
        if output_rows == len(df) and expected_columns.issubset(existing_columns):
            print(f"Saltando {job.name}: ya existe completo {job.output_path}")
            return
        print(
            f"Regenerando {job.name}: salida incompleta o antigua "
            f"({output_rows}/{len(df)} filas)."
        )

    missing_columns = [column for column in job.text_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{job.input_path} no tiene columnas requeridas: {missing_columns}")

    job.output_path.parent.mkdir(parents=True, exist_ok=True)

    start, translated_columns = (0, {column: [] for column in job.text_columns})
    if not overwrite:
        start, translated_columns = load_checkpoint(checkpoint_path, job.text_columns)
        if start:
            print(f"Reanudando {job.name} desde fila {start}.")

    print(f"Traduciendo {job.name}: {len(df)} filas -> {job.output_path}")

    progress = tqdm(range(start, len(df), batch_size), desc=job.name)
    for start_idx in progress:
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]

        for column in job.text_columns:
            translated_batch = [""] * len(batch_df)
            pending_by_lang: dict[str, list[tuple[int, str]]] = {}

            for local_index, (_, row) in enumerate(batch_df.iterrows()):
                text = row.get(column, "")
                if pd.isna(text) or str(text).strip() == "":
                    translated_batch[local_index] = ""
                    continue

                src_lang = source_lang_for_row(row, job)
                pending_by_lang.setdefault(src_lang, []).append((local_index, str(text)))

            for src_lang, indexed_texts in pending_by_lang.items():
                local_indexes = [item[0] for item in indexed_texts]
                texts = [item[1] for item in indexed_texts]
                translations = translate_texts(
                    texts,
                    src_lang,
                    tokenizer,
                    model,
                    device,
                    max_input_tokens=max_input_tokens,
                    max_new_tokens=max_new_tokens,
                )
                for local_index, translated_text in zip(local_indexes, translations, strict=True):
                    translated_batch[local_index] = translated_text

            translated_columns[column].extend(translated_batch)

        translated_rows = len(next(iter(translated_columns.values())))
        if checkpoint_every > 0 and translated_rows % checkpoint_every < batch_size:
            flush_checkpoint(df, translated_columns, translated_rows, checkpoint_path)

    result = df.copy()
    for column, translated_values in translated_columns.items():
        result[f"{column}_es"] = translated_values
    result.to_parquet(job.output_path, index=False)

    if TRUNCATION_STATS["truncated"]:
        print(
            f"Aviso: {TRUNCATION_STATS['truncated']} de {TRUNCATION_STATS['total']} textos se "
            f"truncaron a {max_input_tokens} tokens al traducir; la etiqueta procede del "
            f"texto completo."
        )

    metadata_path = job.output_path.parent / "dataset_info.json"
    metadata = {
        "source": str(job.input_path),
        "output": str(job.output_path),
        "rows": len(result),
        "source_columns": list(df.columns),
        "translated_columns": [f"{column}_es" for column in job.text_columns],
        "target_language": TARGET_LANG,
        "model": model_name,
        "max_input_tokens": max_input_tokens,
        "truncation": dict(TRUNCATION_STATS),
        "fallback_to_original": int((result[f"{job.text_columns[0]}_es"] == "").sum())
        if f"{job.text_columns[0]}_es" in result
        else 0,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    if checkpoint_path.exists():
        checkpoint_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traduce a espanol los datasets locales bajo data/."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_es"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--profile", choices=("full", "finetune"), default="full")
    parser.add_argument("--max-rows-per-label", type=int, default=60000)
    parser.add_argument("--max-safe-rows", type=int, default=60000)
    parser.add_argument(
        "--jobs",
        nargs="*",
        default=None,
        help="Filtra por texto contenido en el nombre del job, por ejemplo: train hate_val Necent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = discover_jobs(args.data_dir, args.output_dir)
    if args.jobs:
        wanted = [item.lower() for item in args.jobs]
        jobs = [job for job in jobs if any(item in job.name.lower() for item in wanted)]

    if not jobs:
        raise FileNotFoundError(f"No encontre datasets traducibles en {args.data_dir}")

    device = pick_device()
    print(f"Usando device: {device}")
    print(f"Modelo: {args.model}")
    tokenizer, model = load_translator(args.model, device)

    for job in jobs:
        translate_job(
            job=job,
            tokenizer=tokenizer,
            model=model,
            model_name=args.model,
            device=device,
            batch_size=args.batch_size,
            checkpoint_every=args.checkpoint_every,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
            overwrite=args.overwrite,
            profile=args.profile,
            max_rows_per_label=args.max_rows_per_label,
            max_safe_rows=args.max_safe_rows,
        )


if __name__ == "__main__":
    main()

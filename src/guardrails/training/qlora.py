"""Fine-tuning QLoRA del guardrail.

El campo ``sft_text`` contiene instrucción, texto de usuario y JSON objetivo.
La pérdida se calcula **sólo sobre el completado** —el JSON—, no sobre toda la
secuencia: entrenar sobre el prompt gasta la mayor parte de la señal en enseñar
al modelo a reproducir la entrada, que no es lo que se le pide en inferencia.

Se guarda el mejor punto de control según la pérdida de validación, no el
último paso, y se registran los identificadores de las filas efectivamente
consumidas, para poder reconstruir después qué vio el modelo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

from guardrails import prompting

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class SaveStateCallback(TrainerCallback):
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def on_train_begin(self, args, state, control, **kwargs):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train_state.json").write_text(
            json.dumps({"status": "running", "global_step": 0}, indent=2),
            encoding="utf-8",
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        payload = {
            "status": "running",
            "global_step": state.global_step,
            "epoch": state.epoch,
            "logs": logs or {},
        }
        (self.output_dir / "train_state.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def on_train_end(self, args, state, control, **kwargs):
        (self.output_dir / "train_state.json").write_text(
            json.dumps(
                {
                    "status": "finished",
                    "global_step": state.global_step,
                    "epoch": state.epoch,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def consumed_ids(dataset) -> list[str]:
    """Identificadores de las filas que el entrenamiento consumió."""
    if "id" not in dataset.column_names:
        return []
    return list(dataset["id"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tuning QLoRA del guardrail sobre pares prompt-completion."
    )
    parser.add_argument("--dataset", type=Path, default=Path("data_finetune/guardrail_es.parquet"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("models/guardrail-qwen25-1_5b-qlora"))
    parser.add_argument("--adapter-init", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Dispositivo de entrenamiento. `auto` usa CUDA si está disponible.",
    )
    parser.add_argument(
        "--no-quantization",
        dest="quantize",
        action="store_false",
        help="Carga el modelo base sin cuantizar. Obligatorio fuera de CUDA.",
    )
    parser.set_defaults(quantize=True)
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        metavar="RUTA|auto",
        help="Reanuda desde un punto de control. `auto` toma el más reciente de --output-dir.",
    )
    parser.add_argument(
        "--no-best-model",
        dest="load_best_model",
        action="store_false",
        help="Conserva el último punto de control en vez del de menor pérdida de validación.",
    )
    parser.set_defaults(load_best_model=True)
    return parser


def resolve_device(preferencia: str) -> str:
    """Dispositivo efectivo a partir de la preferencia indicada."""
    if preferencia == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if preferencia == "cuda":
        raise RuntimeError("Se pidió CUDA y torch.cuda no está disponible.")
    return "cpu"


def latest_checkpoint(output_dir: Path) -> Path | None:
    """Punto de control más reciente dentro de un directorio de salida."""
    puntos = sorted(
        (d for d in output_dir.glob("checkpoint-*") if d.is_dir()),
        key=lambda d: int(d.name.rsplit("-", 1)[-1]),
    )
    return puntos[-1] if puntos else None


def resolve_resume(valor: str | None, output_dir: Path) -> str | bool | None:
    """Argumento que espera ``Trainer.train`` para reanudar."""
    if valor is None:
        return None
    if valor != "auto":
        return valor
    punto = latest_checkpoint(output_dir)
    if punto is None:
        print(f"No hay puntos de control en {output_dir}; se entrena desde cero.")
        return None
    print(f"Reanudando desde {punto}.")
    return str(punto)


def _ensure_prompt_completion(dataset):
    """Garantiza las columnas ``prompt`` y ``completion``.

    TRL sólo calcula la pérdida sobre el completado si el dataset viene en ese
    formato; con un único campo de texto lo trata como *language modeling* y
    rechaza la opción con ValueError. Un parquet que sólo traiga ``sft_text``
    se convierte al vuelo.
    """
    if {"prompt", "completion"} <= set(dataset.column_names):
        return dataset
    if "sft_text" not in dataset.column_names:
        raise ValueError("el dataset no tiene columnas prompt/completion ni sft_text del que derivarlas")

    def dividir(row):
        prompt, completion = prompting.split_sft_text(row["sft_text"])
        return {"prompt": prompt, "completion": completion}

    return dataset.map(dividir)


def load_sft_splits(dataset_path: Path, max_train: int | None, max_eval: int | None, seed: int):
    """Particiones de entrenamiento y validación, ya barajadas y recortadas."""
    dataset = load_dataset("parquet", data_files=str(dataset_path), split="train")
    dataset = _ensure_prompt_completion(dataset)
    keep_columns = {"prompt", "completion", "split", "primary_label", "decision", "id", "group_id"}
    remove_columns = [column for column in dataset.column_names if column not in keep_columns]
    dataset = dataset.remove_columns(remove_columns)

    train_ds = dataset.filter(lambda row: row["split"] == "train")
    eval_ds = dataset.filter(lambda row: row["split"] == "validation")

    train_ds = train_ds.shuffle(seed=seed)
    eval_ds = eval_ds.shuffle(seed=seed)

    if max_train is not None:
        train_ds = train_ds.select(range(min(max_train, len(train_ds))))
    if max_eval is not None:
        eval_ds = eval_ds.select(range(min(max_eval, len(eval_ds))))

    return train_ds, eval_ds


def print_dataset_stats(train_ds: Dataset, eval_ds: Dataset) -> None:
    print(f"Train rows: {len(train_ds):,}")
    print(f"Eval rows: {len(eval_ds):,}")
    print("Train labels:")
    labels = {}
    for label in train_ds["primary_label"]:
        labels[label] = labels.get(label, 0) + 1
    print(json.dumps(dict(sorted(labels.items())), indent=2, ensure_ascii=False))


def main_with_args(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    quantize = args.quantize and device == "cuda"
    if args.quantize and not quantize:
        print(f"Cuantización desactivada: requiere CUDA y el dispositivo es {device}.")
    # `paged_adamw_8bit` proviene de bitsandbytes y sólo está disponible con CUDA.
    optimizador = "paged_adamw_8bit" if quantize else "adamw_torch"

    train_ds, eval_ds = load_sft_splits(
        args.dataset,
        max_train=args.max_train_samples,
        max_eval=args.max_eval_samples,
        seed=args.seed,
    )
    print_dataset_stats(train_ds, eval_ds)

    tokenizer_source = args.adapter_init if args.adapter_init and args.adapter_init.exists() else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
    carga: dict[str, Any] = {"attn_implementation": "sdpa"}
    if quantize:
        carga["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        carga["device_map"] = "auto"
    else:
        carga["dtype"] = torch.float32 if device == "cpu" else compute_dtype

    model = AutoModelForCausalLM.from_pretrained(args.model, **carga)
    model.config.use_cache = False
    if quantize:
        model = prepare_model_for_kbit_training(model)

    peft_config = None
    if args.adapter_init is not None:
        model = PeftModel.from_pretrained(model, args.adapter_init, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim=optimizador,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=args.bf16 and device == "cuda",
        fp16=(args.fp16 or not args.bf16) and device == "cuda",
        use_cpu=device == "cpu",
        gradient_checkpointing=True,
        report_to="none",
        # Sin dataset_text_field: el dataset viene en formato prompt-completion,
        # que es el único con el que TRL admite completion_only_loss.
        packing=False,
        seed=args.seed,
        # La pérdida se calcula sólo sobre el JSON de salida. Sin esto, la mayor
        # parte de la señal enseña a reproducir el prompt.
        completion_only_loss=True,
        # Se conserva el punto de control con menor pérdida de validación, no el
        # del último paso.
        load_best_model_at_end=args.load_best_model,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_strategy="steps",
    )

    # El colador de TRL decide el formato por las claves del primer ejemplo: si
    # ve cualquier otra columna de texto, vuelve a tratarlo como language
    # modeling. Se le pasan sólo las dos que necesita.
    columnas = ["prompt", "completion"]
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds.select_columns(columnas),
        eval_dataset=eval_ds.select_columns(columnas),
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[SaveStateCallback(args.output_dir)],
    )

    trainer.train(resume_from_checkpoint=resolve_resume(args.resume_from_checkpoint, args.output_dir))
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    # Qué filas vio realmente el modelo. Con --max-steps el entrenamiento corta
    # antes de completar una época, así que el dataset no basta para saberlo.
    ids_train = consumed_ids(train_ds)
    ids_eval = consumed_ids(eval_ds)
    if ids_train:
        (args.output_dir / "consumed_ids.json").write_text(
            json.dumps({"train": ids_train, "validation": ids_eval}, ensure_ascii=False),
            encoding="utf-8",
        )

    metadata = {
        "base_model": args.model,
        "adapter_init": str(args.adapter_init) if args.adapter_init else None,
        "dataset": str(args.dataset),
        "max_length": args.max_length,
        "train_rows": len(train_ds),
        "eval_rows": len(eval_ds),
        "seed": args.seed,
        "max_steps": args.max_steps,
        "effective_batch_size": args.train_batch_size * args.gradient_accumulation_steps,
        "device": device,
        "quantized": quantize,
        "optim": optimizador,
        "completion_only_loss": True,
        "load_best_model_at_end": args.load_best_model,
        "consumed_ids_file": "consumed_ids.json" if ids_train else None,
        "type": "qlora_adapter",
    }
    (args.output_dir / "guardrail_training.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Adapter guardado en {args.output_dir}")


def main() -> None:
    """Punto de entrada de la línea de órdenes."""
    main_with_args()


if __name__ == "__main__":
    main()

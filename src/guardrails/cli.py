"""Punto de entrada único y portable del proyecto.

El mismo comando funciona en Windows y en Linux::

    uv run guardrails doctor
    uv run guardrails data build --output data_finetune/guardrail_es.parquet
    uv run guardrails eval golden-set
    uv run guardrails train qlora --bf16 --max-steps 3000

Cada subcomando delega en el ``main()`` del módulo correspondiente y le pasa
los argumentos restantes sin tocarlos, de modo que las banderas propias de cada
etapa se documentan con ``--help`` en ese nivel.

Las importaciones son perezosas a propósito: ``guardrails doctor`` y
``guardrails eval golden-set`` no requieren torch.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from collections.abc import Sequence

# subcomando -> (módulo, descripción)
_COMMANDS: dict[str, tuple[str, str]] = {
    "data ingest": ("guardrails.data.ingest", "Descarga un dataset de Hugging Face a parquet local"),
    "data translate": ("guardrails.data.translate", "Traduce los datasets locales al español (NLLB-200)"),
    "data build": (
        "guardrails.data.build_dataset",
        "Consolida los datasets traducidos en un parquet de fine-tuning",
    ),
    "data corrective": (
        "guardrails.data.corrective",
        "Construye el dataset correctivo a partir de predicciones del golden set",
    ),
    "train qlora": ("guardrails.training.qlora", "Fine-tuning QLoRA sobre el campo sft_text"),
    "eval golden-set": (
        "guardrails.evaluation.golden_set",
        "Genera eval_golden_es.csv de forma determinista",
    ),
    "eval run": ("guardrails.evaluation.run_eval", "Evalúa un adaptador y escribe predicciones y métricas"),
    "publish hf": ("guardrails.publishing.publish_hf", "Publica dataset y modelo en Hugging Face"),
}


def _run_module(module_name: str, argv: Sequence[str], prog: str) -> int:
    module = importlib.import_module(module_name)
    main_fn = getattr(module, "main", None)
    if main_fn is None:  # pragma: no cover - defensivo
        print(f"El módulo {module_name} no expone main()", file=sys.stderr)
        return 2
    saved = sys.argv
    try:
        sys.argv = [prog, *argv]
        main_fn()
    finally:
        sys.argv = saved
    return 0


def _doctor() -> int:
    """Informe del entorno: plataforma, Python, GPU y dependencias opcionales."""
    from guardrails import paths

    print(f"plataforma        : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"python            : {sys.version.split()[0]} ({sys.executable})")
    print(f"raíz del proyecto : {paths.PROJECT_ROOT}")
    print(f"raíz vista en WSL : {paths.wsl_project_root()}")

    try:
        import torch
    except ImportError:
        print("torch             : NO INSTALADO  (instala el extra: uv sync --extra train)")
        print("cuda              : no evaluable sin torch")
    else:
        print(f"torch             : {torch.__version__}")
        available = torch.cuda.is_available()
        print(f"cuda disponible   : {available}")
        if available:
            print(f"cuda (build)      : {torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total = props.total_memory / (1024**3)
                print(f"  gpu[{i}]          : {props.name}  ({total:.1f} GiB)")
        else:
            print(
                "  aviso           : el entrenamiento QLoRA aborta sin CUDA "
                "(guardrails/training/qlora.py, comprobación al inicio de main())"
            )

    for name in ("transformers", "peft", "trl", "datasets", "bitsandbytes", "pandas", "pyarrow"):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            print(f"{name:<18}: NO INSTALADO")
        else:
            print(f"{name:<18}: {getattr(mod, '__version__', 'desconocida')}")
    return 0


def _serve(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="guardrails serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)
    try:
        import uvicorn
    except ImportError:
        print("uvicorn no está instalado. Ejecuta: uv sync --extra serve", file=sys.stderr)
        return 1
    uvicorn.run("guardrails.serving.backend:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="guardrails",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="group", metavar="GRUPO")

    for group in ("data", "train", "eval", "publish"):
        group_parser = sub.add_parser(group, help=f"comandos de {group}")
        group_sub = group_parser.add_subparsers(dest="command", metavar="COMANDO")
        for key, (_, help_text) in _COMMANDS.items():
            head, _, tail = key.partition(" ")
            if head == group:
                group_sub.add_parser(tail, help=help_text, add_help=False)

    sub.add_parser("serve", help="levanta la consola de gobernanza (FastAPI)", add_help=False)
    sub.add_parser("doctor", help="informe del entorno: plataforma, GPU y dependencias")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()

    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0

    group, rest = argv[0], argv[1:]

    if group == "doctor":
        return _doctor()
    if group == "serve":
        return _serve(rest)

    if not rest:
        parser.parse_args([group, "--help"])
        return 2

    key = f"{group} {rest[0]}"
    entry = _COMMANDS.get(key)
    if entry is None:
        parser.error(f"subcomando desconocido: {key!r}")
    module_name, _ = entry
    return _run_module(module_name, rest[1:], prog=f"guardrails {key}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

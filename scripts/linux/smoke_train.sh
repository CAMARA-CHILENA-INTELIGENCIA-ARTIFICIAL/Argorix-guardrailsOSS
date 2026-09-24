#!/usr/bin/env bash
# Lanzador Linux. Equivalente exacto de scripts/windows/<mismo-nombre>.ps1.
# Los parámetros aplicados están registrados en configs/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

command -v uv >/dev/null 2>&1 || { echo "uv no está instalado: https://docs.astral.sh/uv/" >&2; exit 1; }

uv sync --extra train

uv run guardrails train qlora \
  --dataset data_finetune/guardrail_es.parquet \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir models/guardrail-qwen25-1_5b-qlora-smoke \
  --max-train-samples 2000 \
  --max-eval-samples 200 \
  --max-steps 20 \
  --eval-steps 10 \
  --save-steps 10 \
  --logging-steps 2 \
  --train-batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --max-length 512 \
  --bf16

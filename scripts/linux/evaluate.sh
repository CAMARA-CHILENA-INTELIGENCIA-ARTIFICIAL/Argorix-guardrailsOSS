#!/usr/bin/env bash
# Evaluación de extremo a extremo. Equivalente exacto de scripts/windows/evaluate.ps1.
#
# Genera el conjunto de medición, evalúa el modelo aislado y la ruta completa de
# producción, y deja las métricas con su bloque run_metadata.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

command -v uv >/dev/null 2>&1 || { echo "uv no está instalado: https://docs.astral.sh/uv/" >&2; exit 1; }

ADAPTER="${1:-models/guardrail-qwen25-1_5b-qlora-v3-corrective}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="reports/${STAMP}"
mkdir -p "$OUT"

echo "== 1/3 conjunto de medición =="
uv run guardrails eval golden-set --output "$OUT/holdout.csv"

echo "== 2/3 modelo aislado =="
uv run guardrails eval run \
  --adapter "$ADAPTER" \
  --input-csv "$OUT/holdout.csv" \
  --load-4bit \
  --quiet \
  --output-jsonl "$OUT/model_predictions.jsonl" \
  --metrics-output "$OUT/model_metrics.json"

echo "== 3/3 ruta completa de producción =="
uv run guardrails eval run \
  --adapter "$ADAPTER" \
  --input-csv "$OUT/holdout.csv" \
  --load-4bit \
  --production-path \
  --quiet \
  --output-jsonl "$OUT/production_predictions.jsonl" \
  --metrics-output "$OUT/production_metrics.json"

echo
echo "Resultados en $OUT"
echo "  model_metrics.json       modelo aislado"
echo "  production_metrics.json  ruta completa; nunca bloquea menos que la anterior"

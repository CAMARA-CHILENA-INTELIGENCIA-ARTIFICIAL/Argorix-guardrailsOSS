#!/usr/bin/env bash
# Prepara el entorno de entrenamiento (requiere GPU NVIDIA con CUDA).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
command -v uv >/dev/null 2>&1 || { echo "uv no está instalado: https://docs.astral.sh/uv/" >&2; exit 1; }

uv sync --extra train --group dev
uv run guardrails doctor

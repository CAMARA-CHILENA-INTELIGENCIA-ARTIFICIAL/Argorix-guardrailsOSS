#!/usr/bin/env bash
# Lanzador Linux. Equivalente exacto de scripts/windows/<mismo-nombre>.ps1.
# Los parámetros aplicados están registrados en configs/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

command -v uv >/dev/null 2>&1 || { echo "uv no está instalado: https://docs.astral.sh/uv/" >&2; exit 1; }

uv sync --extra serve
uv run guardrails serve --host 127.0.0.1 --port 8000

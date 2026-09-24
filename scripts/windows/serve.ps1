# Lanzador Windows. Equivalente exacto de scripts/linux/<mismo-nombre>.sh.
# Los parámetros aplicados están registrados en configs/.
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv no está instalado: https://docs.astral.sh/uv/"
    exit 1
}

uv sync --extra serve
uv run guardrails serve --host 127.0.0.1 --port 8000

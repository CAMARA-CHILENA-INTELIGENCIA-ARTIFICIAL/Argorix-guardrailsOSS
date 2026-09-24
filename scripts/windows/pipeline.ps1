# Lanzador Windows. Equivalente exacto de scripts/linux/<mismo-nombre>.sh.
# Los parámetros aplicados están registrados en configs/.
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv no está instalado: https://docs.astral.sh/uv/"
    exit 1
}

uv run guardrails data translate `
  --profile finetune `
  --max-rows-per-label 60000 `
  --max-safe-rows 60000 `
  --batch-size 64 `
  --checkpoint-every 500 `
  --max-input-tokens 256 `
  --max-new-tokens 256

uv run guardrails data build `
  --output data_finetune/guardrail_es.parquet `
  --max-variants-per-row 2 `
  --max-train-per-label 120000 `
  --max-eval-per-label 10000 `
  --max-train-per-source-label 60000

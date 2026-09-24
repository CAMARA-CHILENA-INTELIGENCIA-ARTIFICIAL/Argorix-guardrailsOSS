# Lanzador Windows. Equivalente exacto de scripts/linux/<mismo-nombre>.sh.
# Los parámetros aplicados están registrados en configs/.
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv no está instalado: https://docs.astral.sh/uv/"
    exit 1
}

uv run guardrails train qlora `
  --dataset data_finetune/guardrail_es.parquet `
  --model Qwen/Qwen2.5-1.5B-Instruct `
  --output-dir models/guardrail-qwen25-1_5b-qlora-v1 `
  --max-eval-samples 2000 `
  --epochs 1 `
  --max-steps 3000 `
  --eval-steps 250 `
  --save-steps 250 `
  --logging-steps 20 `
  --train-batch-size 2 `
  --eval-batch-size 2 `
  --gradient-accumulation-steps 16 `
  --max-length 512 `
  --bf16

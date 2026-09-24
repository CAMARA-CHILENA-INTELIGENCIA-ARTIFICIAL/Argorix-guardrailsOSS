# Lanzador Windows. Equivalente exacto de scripts/linux/<mismo-nombre>.sh.
# Los parámetros aplicados están registrados en configs/.
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv no está instalado: https://docs.astral.sh/uv/"
    exit 1
}

# Espera a que termine la traducción del cuarto origen y construye el dataset v2.
$translated = Join-Path $Root "data_es\bogdanminko__Catch_the_prompt_injection_or_jailbreak_or_benign\train.parquet"
$partial    = Join-Path $Root "data_es\bogdanminko__Catch_the_prompt_injection_or_jailbreak_or_benign\train.partial.parquet"
$log        = Join-Path $Root "logs\v2_prepare_after_translate.log"

New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

while ((-not (Test-Path $translated)) -or (Test-Path $partial)) {
    Start-Sleep -Seconds 60
}

uv run guardrails data build `
    --output data_finetune/guardrail_es_v2.parquet `
    --max-variants-per-row 2 `
    --max-train-per-label 160000 `
    --max-eval-per-label 12000 `
    --max-train-per-source-label 90000 `
    *> $log

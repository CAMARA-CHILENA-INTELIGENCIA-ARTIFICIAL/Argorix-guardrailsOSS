# Evaluación de extremo a extremo. Equivalente exacto de scripts/linux/evaluate.sh.
#
# Genera el conjunto de medición, evalúa el modelo aislado y la ruta completa de
# producción, y deja las métricas con su bloque run_metadata.
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv no está instalado: https://docs.astral.sh/uv/"
    exit 1
}

$Adapter = if ($args.Count -ge 1) { $args[0] } else { "models/guardrail-qwen25-1_5b-qlora-v3-corrective" }
$Stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$Out = "reports/$Stamp"
New-Item -ItemType Directory -Force -Path $Out | Out-Null

Write-Host "== 1/3 conjunto de medición =="
uv run guardrails eval golden-set --output "$Out/holdout.csv"

Write-Host "== 2/3 modelo aislado =="
uv run guardrails eval run `
  --adapter $Adapter `
  --input-csv "$Out/holdout.csv" `
  --load-4bit `
  --quiet `
  --output-jsonl "$Out/model_predictions.jsonl" `
  --metrics-output "$Out/model_metrics.json"

Write-Host "== 3/3 ruta completa de producción =="
uv run guardrails eval run `
  --adapter $Adapter `
  --input-csv "$Out/holdout.csv" `
  --load-4bit `
  --production-path `
  --quiet `
  --output-jsonl "$Out/production_predictions.jsonl" `
  --metrics-output "$Out/production_metrics.json"

Write-Host ""
Write-Host "Resultados en $Out"
Write-Host "  model_metrics.json       modelo aislado"
Write-Host "  production_metrics.json  ruta completa; nunca bloquea menos que la anterior"

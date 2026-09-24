# Prepara el entorno de entrenamiento (requiere GPU NVIDIA con CUDA).
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv no está instalado: https://docs.astral.sh/uv/"
    exit 1
}

uv sync --extra train --group dev
uv run guardrails doctor

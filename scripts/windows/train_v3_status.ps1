$ErrorActionPreference = "SilentlyContinue"

$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$out = Join-Path $project "models\guardrail-qwen25-1_5b-qlora-v3-corrective"
$state = Join-Path $out "train_state.json"
$stdout = Join-Path $project "logs\train_v3_stdout.log"
$stderr = Join-Path $project "logs\train_v3_stderr.log"

Write-Host "== PROCESOS =="
Get-Process python,powershell,wsl -ErrorAction SilentlyContinue |
  Select-Object Id, CPU, StartTime, Path |
  Format-Table

Write-Host "`n== GPU =="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits

Write-Host "`n== TRAIN STATE =="
if (Test-Path $state) { Get-Content $state } else { Write-Host "Aun no existe train_state.json." }

Write-Host "`n== STDOUT =="
if (Test-Path $stdout) { Get-Content $stdout -Tail 30 }

Write-Host "`n== STDERR / PROGRESS =="
if (Test-Path $stderr) { Get-Content $stderr -Tail 40 }

Write-Host "`n== OUTPUT =="
if (Test-Path $out) {
  Get-ChildItem $out | Sort-Object LastWriteTime -Descending | Select-Object Name,Length,LastWriteTime | Format-Table
} else {
  Write-Host "No existe output todavia: $out"
}

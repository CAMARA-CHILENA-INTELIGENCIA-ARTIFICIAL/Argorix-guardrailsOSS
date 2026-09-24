$ErrorActionPreference = "SilentlyContinue"

$PidFile = "logs\full_pipeline.pid"
$Stdout = "logs\full_pipeline.log"
$Stderr = "logs\full_pipeline.err.log"

if (Test-Path $PidFile) {
  $PipelinePid = [int](Get-Content $PidFile)
  $Process = Get-Process -Id $PipelinePid
  if ($Process) {
    Write-Output "RUNNING PID=$PipelinePid CPU=$($Process.CPU)"
  } else {
    Write-Output "NOT RUNNING PID=$PipelinePid"
  }
} else {
  Write-Output "NO PID FILE"
}

Write-Output ""
Write-Output "== STDOUT =="
if (Test-Path $Stdout) {
  Get-Content $Stdout -Tail 25
}

Write-Output ""
Write-Output "== STDERR / PROGRESS =="
if (Test-Path $Stderr) {
  Get-Content $Stderr -Tail 25
}

Write-Output ""
Write-Output "== LATEST OUTPUTS =="
if (Test-Path "data_es") {
  Get-ChildItem -Path "data_es" -Recurse -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 15 FullName,Length,LastWriteTime |
    Format-Table -AutoSize
}

Write-Output ""
Write-Output "== FINAL DATASET =="
if (Test-Path "data_finetune\guardrail_es.parquet") {
  Get-ChildItem "data_finetune\guardrail_es.parquet", "data_finetune\guardrail_es.summary.json" |
    Select-Object FullName,Length,LastWriteTime |
    Format-Table -AutoSize
} else {
  Write-Output "Final parquet not created yet."
}

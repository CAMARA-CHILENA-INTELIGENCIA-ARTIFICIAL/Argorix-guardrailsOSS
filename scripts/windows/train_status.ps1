$ErrorActionPreference = "SilentlyContinue"

$PidFile = "logs\wsl_full_train.pid"
$Stdout = "logs\wsl_full_train.log"
$Stderr = "logs\wsl_full_train.err.log"
$State = "models\guardrail-qwen25-1_5b-qlora-v1\train_state.json"

if (Test-Path $PidFile) {
  $TrainPid = [int](Get-Content $PidFile)
  $Process = Get-Process -Id $TrainPid
  if ($Process) {
    Write-Output "RUNNING PID=$TrainPid CPU=$($Process.CPU)"
  } else {
    Write-Output "NOT RUNNING PID=$TrainPid"
  }
} else {
  Write-Output "NO PID FILE"
}

Write-Output ""
Write-Output "== TRAIN STATE =="
if (Test-Path $State) {
  Get-Content $State
} else {
  Write-Output "No train_state.json yet."
}

Write-Output ""
Write-Output "== STDOUT =="
if (Test-Path $Stdout) {
  Get-Content $Stdout -Tail 40
}

Write-Output ""
Write-Output "== STDERR / PROGRESS =="
if (Test-Path $Stderr) {
  Get-Content $Stderr -Tail 40
}

Write-Output ""
Write-Output "== OUTPUT =="
if (Test-Path "models\guardrail-qwen25-1_5b-qlora-v1") {
  Get-ChildItem "models\guardrail-qwen25-1_5b-qlora-v1" -Force |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 15 Name,Length,LastWriteTime |
    Format-Table -AutoSize
} else {
  Write-Output "Full model output not created yet."
}

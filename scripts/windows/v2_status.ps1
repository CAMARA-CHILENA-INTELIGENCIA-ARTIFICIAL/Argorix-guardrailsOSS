$ErrorActionPreference = "SilentlyContinue"

$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$stdout = Join-Path $project "logs\translate_v2_bogdan_stdout.log"
$stderr = Join-Path $project "logs\translate_v2_bogdan_stderr.log"
$partial = Join-Path $project "data_es\bogdanminko__Catch_the_prompt_injection_or_jailbreak_or_benign\train.partial.parquet"
$final = Join-Path $project "data_es\bogdanminko__Catch_the_prompt_injection_or_jailbreak_or_benign\train.parquet"
$v2 = Join-Path $project "data_finetune\guardrail_es_v2.parquet"
$summary = Join-Path $project "data_finetune\guardrail_es_v2.summary.json"

Write-Host "== PROCESOS PYTHON =="
Get-Process python | Select-Object Id, CPU, StartTime, Path | Format-Table

Write-Host "`n== GPU =="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits

Write-Host "`n== TRADUCCION V2 =="
foreach ($path in @($partial, $final)) {
    if (Test-Path $path) {
        Get-Item $path | Select-Object FullName, Length, LastWriteTime | Format-List
    } else {
        Write-Host "No existe: $path"
    }
}

Write-Host "`n== STDERR / PROGRESO =="
if (Test-Path $stderr) {
    Get-Content $stderr -Tail 30
}

Write-Host "`n== STDOUT =="
if (Test-Path $stdout) {
    Get-Content $stdout -Tail 30
}

Write-Host "`n== DATASET V2 =="
if (Test-Path $v2) {
    Get-Item $v2 | Select-Object FullName, Length, LastWriteTime | Format-List
} else {
    Write-Host "guardrail_es_v2.parquet aun no existe."
}

if (Test-Path $summary) {
    Get-Content $summary -Tail 80
}

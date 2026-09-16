<#
Windows launcher: runs main.py with the given config on all visible GPUs.
1 GPU  -> python main.py ...            N GPUs -> python -m csf.launch --nproc N (gloo backend, file rendezvous;
torchrun's TCP rendezvous is unreliable on Windows builds of PyTorch)
Extra arguments are forwarded to main.py (e.g. --stage train_qwen --force train_qwen --set train.qwen.lr=1e-4).

Usage: powershell -ExecutionPolicy Bypass -File scripts\launch.ps1 -Config configs\test.yaml [main.py args]
#>
param(
    [Parameter(Mandatory = $true)][string]$Config,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$env:HF_HUB_DISABLE_XET = "1"
$env:USE_LIBUV = "0"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONFAULTHANDLER = "1"

$gpus = [int](& $py -c "import torch; print(torch.cuda.device_count())")
Write-Host "==> Config $Config | GPUs detected: $gpus"
if ($gpus -gt 1) {
    & $py -m csf.launch --nproc $gpus main.py --config $Config @Rest
} else {
    & $py main.py --config $Config @Rest
}
exit $LASTEXITCODE

<#
Windows environment setup for Chrono-Spectral Forensics.

Creates .venv, installs a CUDA build of PyTorch (default cu128 = required for RTX 50xx / Blackwell,
also fine for RTX 30xx/40xx with driver >= 570), installs requirements.txt and runs csf.env_check.

Usage (PowerShell, from the repo root):
    powershell -ExecutionPolicy Bypass -File setup_env.ps1
    powershell -ExecutionPolicy Bypass -File setup_env.ps1 -Cuda cu126 -Python "py -3.11"
    powershell -ExecutionPolicy Bypass -File setup_env.ps1 -Cuda cpu        # CPU-only smoke tests
#>
param(
    [string]$Cuda = "cu128",
    [string]$Python = ""
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Resolve-Python {
    if ($Python) { return $Python }
    foreach ($v in "3.12", "3.11") {
        try { & py "-$v" -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { return "py -$v" } } catch {}
    }
    return "python"
}

$py = Resolve-Python
Write-Host "==> Using interpreter: $py"
if (-not (Test-Path ".venv")) {
    Invoke-Expression "$py -m venv .venv"
}
$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $venvPy -m pip install --upgrade pip wheel setuptools
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

Write-Host "==> Installing PyTorch ($Cuda)"
& $venvPy -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"
if ($LASTEXITCODE -ne 0) { throw "PyTorch install failed (try -Cuda cu126 or cu130)" }

Write-Host "==> Installing requirements"
& $venvPy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "requirements install failed" }

Write-Host "==> Verifying environment"
& $venvPy -m csf.env_check
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. .venv\Scripts\Activate.ps1"
Write-Host "  2. huggingface-cli login      (accept the Llama-3.2-11B-Vision licence on huggingface.co first)"
Write-Host "  3. powershell -ExecutionPolicy Bypass -File scripts\run_test.ps1"

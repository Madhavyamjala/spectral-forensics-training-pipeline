# Test run (5 000 videos, 12-16 GB GPU): checks the full training pipeline runs without errors.
& (Join-Path $PSScriptRoot "launch.ps1") -Config configs\test.yaml @args
exit $LASTEXITCODE

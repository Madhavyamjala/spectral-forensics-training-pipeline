# SAFER, the paper's configuration: Real vs AI-Generated on Chrono-66k (configs/safer.yaml).
& (Join-Path $PSScriptRoot "launch.ps1") -Config configs\safer.yaml @args
exit $LASTEXITCODE

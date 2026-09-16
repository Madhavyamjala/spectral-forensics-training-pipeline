# Full training run (all videos, >=24 GB GPU(s)): trains, evaluates, exports and offers to push to the Hub.
& (Join-Path $PSScriptRoot "launch.ps1") -Config configs\full.yaml @args
exit $LASTEXITCODE

param(
    [Parameter(Mandatory = $true)]
    [string]$PythonExe
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = Join-Path $root "src"

& $PythonExe -m bdmtf.cli run-cross-platform `
    --root $root `
    --config (Join-Path $root "configs/external_sources.json") `
    --seeds 0 `
    --max-cascades 5000 `
    --bootstrap-samples 400

exit $LASTEXITCODE

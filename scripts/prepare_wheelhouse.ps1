param(
    [string]$Python = "python",
    [switch]$Dev,
    [switch]$Oasis,
    [switch]$All,
    [switch]$Lock
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Wheelhouse = Join-Path $Root "vendor\wheelhouse"
New-Item -ItemType Directory -Force -Path $Wheelhouse | Out-Null

$Requirements = @()
if ($All) {
    $Requirements += "requirements-all.txt"
} elseif ($Lock) {
    $Requirements += "requirements-lock.txt"
} else {
    $Requirements += "requirements.txt"
}
if ($Dev -and -not $All) {
    $Requirements += "requirements-dev.txt"
}
if ($Oasis -and -not $All) {
    $Requirements += "requirements-oasis.txt"
}

foreach ($Req in $Requirements) {
    & $Python -m pip download -r (Join-Path $Root $Req) -d $Wheelhouse
}

Write-Host "Wheelhouse prepared at $Wheelhouse"

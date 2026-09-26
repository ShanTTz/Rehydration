param(
    [string]$Python = "python",
    [switch]$Dev,
    [switch]$Oasis,
    [switch]$All,
    [switch]$Lock,
    [switch]$Offline
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Venv = Join-Path $Root ".venv"
$PythonExe = Join-Path $Venv "Scripts\python.exe"
$Wheelhouse = Join-Path $Root "vendor\wheelhouse"

if (-not (Test-Path $PythonExe)) {
    & $Python -m venv $Venv
}

$PipOptions = @()
if ($Offline) {
    $PipOptions += "--no-index"
    $PipOptions += "--find-links"
    $PipOptions += $Wheelhouse
}

& $PythonExe -m pip install --upgrade pip

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
    & $PythonExe -m pip install @PipOptions -r (Join-Path $Root $Req)
}

& $PythonExe -m pip install @PipOptions -e $Root
& $PythonExe (Join-Path $Root "scripts\check_environment.py")

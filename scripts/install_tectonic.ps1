$ErrorActionPreference = "Stop"
$Version = "0.16.9"
$Root = Split-Path -Parent $PSScriptRoot
$Target = Join-Path $Root ".tools\tectonic"
$Archive = Join-Path $Target "tectonic.zip"
$Url = "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%40$Version/tectonic-$Version-x86_64-pc-windows-msvc.zip"
$ArchiveSha256 = "131a24604785a9600989a3d91225f597df52ac06f00aeffe86fd529f99ee5cdd"

New-Item -ItemType Directory -Force -Path $Target | Out-Null
if (-not (Test-Path -LiteralPath (Join-Path $Target "tectonic.exe"))) {
    if (-not (Test-Path -LiteralPath $Archive)) {
        Invoke-WebRequest -Uri $Url -OutFile $Archive
    }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLowerInvariant() -ne $ArchiveSha256) {
        throw "Tectonic archive SHA256 mismatch"
    }
    Expand-Archive -LiteralPath $Archive -DestinationPath $Target -Force
}
& (Join-Path $Target "tectonic.exe") --version

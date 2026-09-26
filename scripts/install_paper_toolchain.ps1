$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Tools = Join-Path $Root ".tools"
$PerlVersion = "5.42.2.1"
$PerlTag = "SP_54221_64bit"
$PerlRoot = Join-Path $Tools "perl"
$PerlArchive = Join-Path $PerlRoot "strawberry-perl-$PerlVersion-64bit-portable.zip"
$PerlRuntime = Join-Path $PerlRoot "runtime"
$PerlExe = Join-Path $PerlRuntime "perl\bin\perl.exe"
$PerlUrl = "https://github.com/StrawberryPerl/Perl-Dist-Strawberry/releases/download/$PerlTag/strawberry-perl-$PerlVersion-64bit-portable.zip"
$PerlArchiveSha256 = "32d83be90cf04b807cfb9477482bc36302cdee6f5b04cf57e81adecbd8f07898"
$LatexdiffScript = Join-Path $Root "vendor\latexdiff\latexdiff-so"
$LatexdiffSha256 = "9b0233779f8d9aef304af220ed1b7fc86b66323462de20784b2ef9892d18b53f"

& (Join-Path $PSScriptRoot "install_tectonic.ps1")
$env:LC_ALL = ""
$env:LANG = ""

New-Item -ItemType Directory -Force -Path $PerlRoot | Out-Null
if (-not (Test-Path -LiteralPath $PerlExe)) {
    if (-not (Test-Path -LiteralPath $PerlArchive)) {
        Invoke-WebRequest -Uri $PerlUrl -OutFile $PerlArchive
    }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $PerlArchive).Hash.ToLowerInvariant() -ne $PerlArchiveSha256) {
        throw "Portable Perl archive SHA256 mismatch"
    }
    New-Item -ItemType Directory -Force -Path $PerlRuntime | Out-Null
    Expand-Archive -LiteralPath $PerlArchive -DestinationPath $PerlRuntime -Force
}

if ((Get-FileHash -Algorithm SHA256 -LiteralPath $LatexdiffScript).Hash.ToLowerInvariant() -ne $LatexdiffSha256) {
    throw "Vendored latexdiff SHA256 mismatch"
}

$Manifest = [ordered]@{
    status = "complete"
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
    perl_version = (& $PerlExe -e "print `$^V")
    perl_archive_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $PerlArchive).Hash.ToLowerInvariant()
    latexdiff_sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $LatexdiffScript).Hash.ToLowerInvariant()
    tectonic_version = (& (Join-Path $Tools "tectonic\tectonic.exe") --version)
}
$Manifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Tools "paper_toolchain_manifest.json") -Encoding utf8
& $PerlExe $LatexdiffScript --version

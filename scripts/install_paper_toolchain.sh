#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TOOLS="$ROOT/.tools"
LATEXDIFF_SCRIPT="$ROOT/vendor/latexdiff/latexdiff-so"
LATEXDIFF_SHA256="9b0233779f8d9aef304af220ed1b7fc86b66323462de20784b2ef9892d18b53f"

if ! command -v perl >/dev/null 2>&1; then
  echo "Perl 5.8 or newer is required for latexdiff." >&2
  exit 1
fi

if ! command -v tectonic >/dev/null 2>&1 && [ ! -x "$TOOLS/tectonic/tectonic" ]; then
  echo "Install Tectonic 0.16.9 or use Dockerfile.paper." >&2
  exit 1
fi

if [ "$(sha256sum "$LATEXDIFF_SCRIPT" | awk '{print $1}')" != "$LATEXDIFF_SHA256" ]; then
  echo "Vendored latexdiff SHA256 mismatch." >&2
  exit 1
fi

perl "$LATEXDIFF_SCRIPT" --version

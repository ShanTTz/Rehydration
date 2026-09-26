#!/usr/bin/env sh
set -eu

PYTHON_BIN="${PYTHON:-python3}"
ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_EXE="$VENV_DIR/bin/python"
REQ_FILE="requirements.txt"
PIP_OPTIONS=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --lock) REQ_FILE="requirements-lock.txt" ;;
    --dev) REQ_FILE="requirements-dev.txt" ;;
    --oasis) REQ_FILE="requirements-oasis.txt" ;;
    --all) REQ_FILE="requirements-all.txt" ;;
    --offline) PIP_OPTIONS="--no-index --find-links $ROOT_DIR/vendor/wheelhouse" ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ ! -x "$PYTHON_EXE" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$PYTHON_EXE" -m pip install --upgrade pip
"$PYTHON_EXE" -m pip install $PIP_OPTIONS -r "$ROOT_DIR/$REQ_FILE"
"$PYTHON_EXE" -m pip install $PIP_OPTIONS -e "$ROOT_DIR"
"$PYTHON_EXE" "$ROOT_DIR/scripts/check_environment.py"

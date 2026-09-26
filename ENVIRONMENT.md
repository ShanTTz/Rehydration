# Environment

This repository has two execution modes:

1. Local BDMTF reproduction with bundled frozen intents.
2. Optional API-backed intent regeneration or upstream OASIS integration.

The first mode is enough to run the paper-style pipeline locally. It does not
make network calls during simulation.

## Core Requirements

- Python 3.12 is the primary reproducibility environment. Python 3.13 is also
  covered by the current local test run.
- `setuptools`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`,
  `pyarrow`, and `requests`.
- Reserve at least 10 GB when importing all TBBT archives and retaining HN
  request caches. Raw TBBT archives alone occupy about 4.2 GB and are ignored by
  Git; compact panels and manifests remain in the repository.
- R 4.4.1 is isolated in `Dockerfile.causal`; manuscript compilation uses
  Tectonic 0.16.9, vendored latexdiff 1.4.0, and Perl.

All dependency declarations live in the repository:

- `requirements.txt`: core runtime.
- `requirements-lock.txt`: exact tested core versions.
- `requirements-dev.txt`: test tooling.
- `requirements-oasis.txt`: optional upstream OASIS.
- `requirements-legacy.txt`: old Reddit collection and direct-OASIS scripts.
- `requirements-data.txt`: DVC and public-data workflow support.
- `requirements-all.txt`: development, data, and optional OASIS.

Install from the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Or use the bundled PowerShell bootstrap:

```powershell
.\scripts\bootstrap_environment.ps1
```

On macOS or Linux, use:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

Or:

```bash
sh scripts/bootstrap_environment.sh
```

## Developer/Test Requirements

The built-in unit tests use Python `unittest`, so `pytest` is optional.
Install the dev set if you prefer running tests through pytest:

```bash
python -m pip install -r requirements-dev.txt
```

## Optional API Requirements

The API intent builder uses Python standard-library HTTP calls. It does not
require the `openai` Python package.

Copy `.env.example` to `.env` or set these variables in your shell:

```bash
set OPENAI_API_KEY=replace-with-your-key
set OPENAI_BASE_URL=https://api.openai.com/v1
set OPENAI_MODEL=gpt-4o-mini
```

PowerShell:

```powershell
$env:OPENAI_API_KEY="replace-with-your-key"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
$env:OPENAI_MODEL="gpt-4o-mini"
```

Use the API only when you want to regenerate `frozen_intents.jsonl`. Normal
simulation runs use the bundled frozen intent files.

Check API configuration without a live request:

```bash
python scripts/check_api_environment.py
```

Run one tiny live request when network access is allowed:

```bash
python scripts/check_api_environment.py --live
```

See `LLM_REGENERATION.md` for the full frozen-intent regeneration workflow.

## Optional OASIS Requirements

The core reproduction does not require upstream OASIS. To experiment with the
OASIS bridge:

```bash
python -m pip install -r requirements-oasis.txt
```

The repository vendors the OASIS action, recommendation, platform-state,
database, and agent modules needed by the adapter under `vendor/oasis/upstream`,
plus a lightweight offline shim. It is a licensed, paper-relevant subset rather
than a mirror of every upstream OASIS experiment. Direct upstream execution uses
the separately locked Python 3.11-compatible environment.

To vendor a full upstream OASIS checkout into this repository:

```bash
python scripts/vendor_upstream_oasis.py --source-dir C:\path\to\oasis --replace
```

## Offline Dependency Bundle

For a no-network machine, first prepare wheels in an allowed network
environment:

```powershell
.\scripts\prepare_wheelhouse.ps1 -Dev
```

Then copy the repository, including `vendor/wheelhouse`, and install offline:

```powershell
.\scripts\bootstrap_environment.ps1 -Offline
```

## Sanity Checks

```bash
python scripts/check_environment.py
python -m unittest discover -s tests
python scripts/quick_smoke.py
python scripts/reproduce_paper.py --quick
```

Expected outputs are written under `run_outputs*` directories.

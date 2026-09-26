# Dependencies

The repository keeps dependency information inside the codebase so the local
BDMTF reproduction can be recreated without hunting through old scripts.

## Dependency Files

- `requirements.txt`: core runtime dependencies.
- `requirements-lock.txt`: tested core environment with exact versions.
- `requirements-dev.txt`: core dependencies plus test tooling.
- `requirements-oasis.txt`: core dependencies plus optional upstream OASIS.
- `requirements-legacy.txt`: old Reddit/OASIS collection and executor tools.
- `requirements-data.txt`: DVC and public-data tooling.
- `requirements-all.txt`: dev plus optional OASIS dependencies.
- `.env.example`: optional API variables for regenerating frozen intents.
- `vendor/wheelhouse/`: optional offline wheel cache.

## Core Runtime

The core reproduction uses:

- Python 3.12 primary environment.
- `setuptools` for editable package installation.
- `numpy`.
- `pandas`.
- `scipy`.
- `scikit-learn`.
- `matplotlib`.
- `pyarrow`.
- `requests`.

No OpenAI SDK is required for the core pipeline. The API intent builder uses
Python standard-library HTTP requests.

## Optional Runtime Pieces

OASIS is optional. The repository includes an OASIS bridge, an offline shim,
and the licensed upstream modules required by the paper adapter, but the
paper-style BDMTF mechanism runs locally without installing `camel-oasis`.

Install OASIS only when you want to test direct platform integration:

```bash
python -m pip install -r requirements-oasis.txt
```

Install the legacy collection/extraction tools only when you need to rerun the
old raw Reddit or direct-OASIS scripts:

```bash
python -m pip install -r requirements-legacy.txt
```

## Offline Wheelhouse

The repository includes `vendor/wheelhouse` as the place to store downloaded
Python wheels when you need to move the project into a no-network environment.

In a network-enabled environment:

```powershell
.\scripts\prepare_wheelhouse.ps1 -Dev
```

Then install offline:

```powershell
.\scripts\bootstrap_environment.ps1 -Offline
```

The current checked-in code does not vendor platform-specific binary copies of
`numpy`, `pandas`, or R packages. The OASIS source subset is vendored; Python
wheels remain optional because they differ across Windows, Linux, CPU
architecture, and Python minor version.

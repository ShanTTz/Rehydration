# Upstream OASIS Vendoring

This repository includes a local OASIS/CAMEL compatibility shim for the exact
interfaces used by the sanitized legacy executor. It does not currently contain
a full upstream `camel-ai/oasis` source checkout.

To make the repository fully self-contained for direct upstream-OASIS
experiments, vendor a known OASIS checkout under `vendor/oasis/upstream`.

## From A Local Checkout

```bash
python scripts/vendor_upstream_oasis.py \
  --source-dir C:\path\to\oasis \
  --replace
```

## From A Source Zip

```bash
python scripts/vendor_upstream_oasis.py \
  --source-zip C:\path\to\oasis-main.zip \
  --replace
```

## From GitHub In A Network-Enabled Environment

```bash
python scripts/vendor_upstream_oasis.py \
  --github-ref refs/heads/main \
  --replace
```

The script writes `vendor/oasis/upstream_manifest.json` with source, timestamp,
file count, byte count, and top-level entries.

## Current Boundary

The BDMTF paper reproduction itself remains implemented in `src/bdmtf` because
it needs deterministic frozen intents, comment-level top-k visibility, reply
tree parent selection, and ranking counterfactuals. Direct upstream-OASIS runs
are useful for extension work, but they should be aligned against these BDMTF
mechanism constraints before being treated as paper-equivalent.

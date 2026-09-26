# Frozen-Semantics Counterfactual Replay

Anonymous code and reproduction materials for BDMTF and the Rehydration
protocol, which compare platform policies while reusing a shared pool of
pre-treatment semantic opportunities.

## Contents

- `src/bdmtf/`: simulator, data loaders, policy interfaces, metrics, and
  validation modules.
- `scripts/`, `configs/`, and `tests/`: experiment entry points, frozen
  configurations, and focused checks.
- `data/social_paper/`: the five-community Reddit study inputs.
- `artifacts/evaluation/` and `artifacts/models/`: compact evaluation
  summaries, plots, and fitted community profiles.
- `vendor/oasis/`: OASIS compatibility components and their upstream notices.
- `experiment_app/`: the local participant-study application source.

## Environment

Python 3.12 is required. Install the core package with:

```bash
python -m pip install -e .
```

For the quick synthetic run and bundled-data audit:

```bash
python scripts/quick_smoke.py
python scripts/audit_reproduction_inputs.py
```

The CSV data files use Git LFS rules in `.gitattributes`. Install Git LFS
before committing the bundled data to a GitHub repository.

Run the paper reproduction with:

```bash
python scripts/reproduce_paper_resumable.py
```

The default replay uses cached intents and does not require a model API key.
Optional intent regeneration reads credentials from environment variables;
see `LLM_REGENERATION.md` and `.env.example`. Never commit a populated `.env`.

## Reproduction Scope

The bundled dataset contains 500 selected posts from five Reddit communities
and the associated filtered discussion records. Larger external-platform
collections and raw archive caches are not included. Their collection and
analysis entry points are in `scripts/` and `src/bdmtf/revision/`; see
`REPRODUCIBILITY.md` and `DATA_CARD.md` for the data inputs and evidence scope.

Author labels in the bundled tables and agent profiles have been replaced
with consistent pseudonyms; profile real-name fields are neutralized. Post/
comment IDs and quoted public text remain in the research inputs for joins and
reproducibility, so the dataset is pseudonymized and can still be linkable to
public discussions. Confirm applicable data terms before redistributing it.

## Citation and Licenses

The manuscript is distributed separately. Third-party components retain their
upstream license and attribution notices; see `vendor/oasis/`. No license is
asserted here for third-party platform data.

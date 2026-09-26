# API-Backed Intent Regeneration

The main reproduction does not call an online model. It replays the bundled
`frozen_intents.jsonl` files so each run is deterministic and auditable.

Use this path only when extending the paper and deliberately regenerating the
offline semantic intent cache.

## Configure Credentials

Do not place real keys in the repository. Set environment variables in the
current shell, or copy `.env.api.example` to a private `.env` file.

PowerShell:

```powershell
$env:OPENAI_API_KEY="replace-with-your-api-key"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
$env:OPENAI_MODEL="gpt-4o-mini"
```

Bash:

```bash
export OPENAI_API_KEY="replace-with-your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"
```

Check the environment without making a network call:

```bash
python scripts/check_api_environment.py
```

Check credentials and network with one tiny live request:

```bash
python scripts/check_api_environment.py --live
```

## Generate A Candidate Intent Cache

Example for one community:

```bash
python scripts/build_api_intents.py \
  --output data/social_paper/aww_data/frozen_intents_api.jsonl \
  --agents 50 \
  --intents-per-agent 4 \
  --topic "discussion in r/aww"
```

Repeat for the five paper communities listed in
`configs/api_intent_generation.json`. Keep the generated files as
`frozen_intents_api.jsonl` until reviewed. Replace `frozen_intents.jsonl` only
when you intentionally want a new reproduction cache.

## Reproducibility Rule

For a paper-style result, freeze the generated JSONL files before running
`scripts/reproduce_paper_resumable.py`. Do not mix live API calls into the
simulation loop, because that would make the counterfactual interventions
non-repeatable.

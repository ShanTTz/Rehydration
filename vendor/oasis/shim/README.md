# Local OASIS/CAMEL Compatibility Shim

This directory implements only the subset of OASIS and CAMEL APIs used by the
BDMTF workspace and the sanitized legacy executor:

- `oasis.ActionType`
- `oasis.ManualAction`
- `oasis.LLMAction`
- `oasis.DefaultPlatformType.REDDIT`
- `oasis.generate_reddit_agent_graph`
- `oasis.make`
- `camel.models.ModelFactory`
- `camel.types.ModelPlatformType`
- `camel.types.ModelType`
- `camel.prompts.TextPrompt`

The shim provides a small SQLite-backed Reddit-like environment for import and
smoke-test compatibility. It is not the upstream `camel-ai/oasis` framework and
does not implement million-agent simulation, recommendation services, or full
platform databases.

Use upstream OASIS when network/package installation is available:

```bash
python -m pip install -r requirements-oasis.txt
```

Use the local shim for no-network verification:

```bash
python scripts/check_oasis_integration.py --prefer-shim
```

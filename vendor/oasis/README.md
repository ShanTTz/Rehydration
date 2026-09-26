# OASIS Source Boundary

The paper says BDMTF is built on OASIS. In this local workspace, no standalone
upstream `camel-ai/oasis` source checkout was found. The repository therefore
keeps OASIS support in four reproducible forms:

1. `requirements-oasis.txt` installs the upstream package `camel-oasis`.
2. `src/bdmtf/oasis_bridge.py` exports BDMTF profiles/actions in
   OASIS-compatible shapes.
3. `legacy/social_pipeline/sanitized_scripts/run_oasis_simulation.py` and
   `_oasis_task_executor.py` preserve the old direct-OASIS execution path in a
   credential-sanitized form.
4. `vendor/oasis/shim` implements the subset of OASIS/CAMEL interfaces that the
   old executor and reproduction checks actually use: `ActionType`,
   `ManualAction`, `LLMAction`, `DefaultPlatformType.REDDIT`,
   `generate_reddit_agent_graph`, `oasis.make`, and a small SQLite-backed
   Reddit environment.

To vendor a specific upstream OASIS checkout, use:

```bash
python scripts/vendor_upstream_oasis.py --source-dir C:\path\to\oasis --replace
```

or place it under:

```text
vendor/oasis/upstream/
```

See `OASIS_UPSTREAM_STATUS.md` for source-zip and GitHub archive options.

The current full paper reproduction does not require direct upstream OASIS
execution because BDMTF needs controlled comment-level visibility, fixed intent
replay, and ranking counterfactuals that are implemented explicitly in
`src/bdmtf`.

Run the local compatibility check:

```bash
python scripts/check_oasis_integration.py --prefer-shim
```

Run the sanitized legacy executor against the shim:

```bash
python legacy/social_pipeline/sanitized_scripts/_oasis_task_executor.py \
  --post_data_file examples/oasis_shim_post.json \
  --profile_path data/social_paper/aww_data/population.json \
  --db_path run_outputs/oasis_shim_executor_demo.db \
  --api_key dummy \
  --api_base http://127.0.0.1 \
  --model_name gpt-4o-mini \
  --steps 2 \
  --action_prob 0.0 \
  --action_prob_decay 1.0 \
  --num_llm_agents 0
```

The shim is not a replacement for upstream OASIS. It is a local, testable
compatibility layer for the exact OASIS surface used by this paper workspace.
See `shim/README.md` for the exported API subset.

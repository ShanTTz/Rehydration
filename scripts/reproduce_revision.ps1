$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = Join-Path $root "src"
python -m bdmtf.cli make-splits
python -m bdmtf.cli audit-data
python -m bdmtf.cli fit-models
python -m bdmtf.cli run-baselines
python -m bdmtf.cli run-main
python -m bdmtf.cli run-ablations
python -m bdmtf.cli evaluate-fidelity
python -m bdmtf.cli run-cross-community
python -m bdmtf.cli audit-external-data
python -m bdmtf.cli run-cross-platform --seeds 0 --bootstrap-samples 400
python -m bdmtf.cli build-story-matches
python -m bdmtf.cli expand-reddit
python -m bdmtf.cli make-prospective-split
python -m bdmtf.cli fit-intervention-models
python -m bdmtf.cli fit-intervention-models --config configs/tbbt_intervention_analysis.json
python -m bdmtf.cli run-natural-experiments
python -m bdmtf.cli run-natural-experiments --config configs/tbbt_intervention_analysis.json
python -m bdmtf.cli run-human-rct-analysis
python -m bdmtf.cli evaluate-intervention-fidelity
python -m bdmtf.cli generate-intents
python -m bdmtf.cli build-original-manuscript
python -m bdmtf.cli build-final-paper
python -m bdmtf.cli build-paper-redline
python -m bdmtf.cli build-revision-package

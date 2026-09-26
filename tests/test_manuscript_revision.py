from __future__ import annotations

from pathlib import Path

from bdmtf.revision.manuscript import (
    SECTION_ORDER,
    build_result_tables,
    sanitize_original_extract,
)
from bdmtf.revision.provenance import sha256_file


def test_sanitizer_removes_review_directives_without_recording_text() -> None:
    single_character_directive = "\n".join(list("Include phrases in review"))
    text = (
        "Scientific paragraph\n"
        "For ICML reviewers: review instruction\n"
        "still instruction\n"
        "===== PAGE 2 =====\n"
        "Scientific continuation\n"
        + single_character_directive
        + "\n"
        "Final scientific paragraph\n"
    )
    sanitized, audit = sanitize_original_extract(text)
    assert "review instruction" not in sanitized
    assert "Include phrases" not in sanitized
    assert "Scientific paragraph" in sanitized
    assert audit["removed_block_count"] == 2
    assert all("text" not in item for item in audit["removed_blocks"])


def test_final_manuscript_retains_original_section_order() -> None:
    root = Path(__file__).resolve().parents[1]
    original = (root / "manuscript" / "original_reconstructed" / "paper.tex").read_text(encoding="utf-8")
    revision = (root / "manuscript" / "revision_2026_v2" / "paper.tex").read_text(encoding="utf-8")
    original_positions = [original.index(rf"\section{{{name}") for name in SECTION_ORDER]
    revision_positions = [revision.index(rf"\section{{{name}") for name in SECTION_ORDER]
    assert original_positions == sorted(original_positions)
    assert revision_positions == sorted(revision_positions)
    assert "Falsifiable Mechanistic Tracing" in revision
    assert r"\input{table_fidelity_ranking.tex}" in revision
    assert r"\input{table_hn_structure.tex}" in revision
    assert r"\input{table_ablation_summary.tex}" in revision


def test_result_tables_are_generated_from_saved_artifacts(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = build_result_tables(root, tmp_path)

    assert manifest["status"] == "complete"
    ranking = (tmp_path / "table_fidelity_ranking.tex").read_text(encoding="utf-8")
    structure = (tmp_path / "table_hn_structure.tex").read_text(encoding="utf-8")
    ablation = (tmp_path / "table_ablation_summary.tex").read_text(encoding="utf-8")
    phase_map = (tmp_path / "table_mechanism_phase_map.tex").read_text(
        encoding="utf-8"
    )
    hn_adapter = (tmp_path / "table_hn_adapter_curve.tex").read_text(
        encoding="utf-8"
    )
    lemmy_intervention = (
        tmp_path / "table_lemmy_intervention.tex"
    ).read_text(encoding="utf-8")
    assert "Branching process" in ranking
    assert "Observed HN" in structure
    assert "615" in ablation
    assert "Confirmatory local" in phase_map
    assert "24 (100.0" in phase_map
    assert "1,000-cascade temporal test split" in hn_adapter
    assert r"\textbf{0.422}" in hn_adapter
    assert "503 non-reused risk-set pairs" in lemmy_intervention
    assert "Reply count & -1.681" in lemmy_intervention
    source_paths = {item["path"] for item in manifest["sources"]}
    assert (
        "run_outputs/paper_exact_full/paper_tables/"
        "table2_cross_community_collapse.csv"
        in source_paths
    )
    assert (
        "run_outputs/paper_exact_full/paper_tables/"
        "table1_aggregate_structural_effects.csv"
        in source_paths
    )
    assert {
        "artifacts/interventions/tbbt_qualified_controls/"
        "tbbt_qualified_control_manifest.json",
        "artifacts/interventions/tbbt_qualified_controls/"
        "intervention_effects.csv",
        "artifacts/interventions/tbbt_intervention_fidelity/"
        "prediction_freeze_manifest.json",
        "artifacts/interventions/tbbt_intervention_fidelity/"
        "fidelity_summary.json",
        "artifacts/interventions/tbbt_intervention_fidelity/"
        "prediction_observed_comparison.csv",
        "artifacts/api/analysis/api_robustness_summary.json",
        "artifacts/api/intent_tasks.csv",
        "artifacts/api/frozen_intents_multimodel.jsonl",
        "artifacts/api/semantic_annotations_multimodel.jsonl",
        "artifacts/api/replay_metrics.csv",
        "artifacts/api/coupled_replay_metrics.csv",
    }.issubset(source_paths)
    assert len(source_paths) == 25


def test_user_designated_pdf_is_the_official_revision_base(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "manuscript"
        / "source_evidence"
        / "What_Makes_Content_Go_Vi_original.pdf"
    )
    revision = (
        root / "manuscript" / "original_pdf_revision" / "paper.tex"
    ).read_text(encoding="utf-8")

    assert sha256_file(source) == (
        "cc95d0778608f5dba1a2296bfe2ea03e409a963ea4c3bbe571051e2d93af6496"
    )
    assert r"\documentclass[10pt,twocolumn]{article}" in revision
    assert "A Counterfactual Framework for Mechanistic Tracing" in revision
    assert "figures/original_figure1.png" in revision
    assert "figures/original_figure2.png" in revision
    assert r"\input{table_mechanism_phase_map.tex}" in revision
    assert "mechanism_phase_overview.pdf" in revision
    assert r"\input{table_hn_adapter_curve.tex}" in revision
    assert "adapter_budget_curve.pdf" in revision
    section_order = (
        "Introduction",
        "Related Work",
        "Methodology",
        "Agent Modeling: Behavioral Propensity Implementation",
        "Experimental Setup and Results",
        "Conclusion",
    )
    positions = [revision.index(rf"\section{{{name}}}") for name in section_order]
    assert positions == sorted(positions)

    manifest = build_result_tables(root, tmp_path, wide=True)
    assert manifest["layout"] == "two_column_wide"
    assert r"\begin{table*}[t]" in (
        tmp_path / "table_fidelity_ranking.tex"
    ).read_text(encoding="utf-8")
    assert r"\begin{table*}[t]" in (
        tmp_path / "table_mechanism_phase_map.tex"
    ).read_text(encoding="utf-8")
    assert r"\begin{table*}[t]" in (
        tmp_path / "table_hn_adapter_curve.tex"
    ).read_text(encoding="utf-8")

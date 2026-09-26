from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build traceable parameter-provenance and channel-ablation tables."
    )
    parser.add_argument(
        "--output",
        default=str(
            ROOT / "manuscript" / "iclr2026_revision_staging" / "generated"
        ),
    )
    args = parser.parse_args()

    config_path = ROOT / "configs" / "paper_exact_reproduction.json"
    model_path = ROOT / "artifacts" / "models" / "community_models.json"
    ablation_path = (
        ROOT
        / "artifacts"
        / "reviewer_validation"
        / "agent_channel_ablation"
        / "table_agent_channel_ablation.tex"
    )
    ablation_summary_path = ablation_path.with_name("channel_ablation_summary.csv")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    models = json.loads(model_path.read_text(encoding="utf-8"))
    sim = config["simulation"]
    n_train = sum(
        int(item["n_train_cascades"])
        for item in models["communities"].values()
    )
    eta_eff = float(sim["external_lambda"])
    ablations = pd.read_csv(ablation_summary_path).set_index("scenario")

    def ablation_macro_lines(scenario: str, prefix: str) -> list[str]:
        row = ablations.loc[scenario]
        return [
            f"\\newcommand{{\\{prefix}Volume}}{{{float(row['geometric_volume_ratio']):.3f}}}",
            f"\\newcommand{{\\{prefix}Depth}}{{{float(row['mean_leaf_depth_delta']):.3f}}}",
            f"\\newcommand{{\\{prefix}BlockShare}}{{{100.0 * float(row['shallow_swarm_block_share']):.0f}\\%}}",
        ]

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    macros_path = output / "agent_modeling_macros.tex"
    macro_lines = [
                f"\\newcommand{{\\AgentBaseImpulse}}{{{float(sim['base_impulse']):.2f}}}",
                f"\\newcommand{{\\AgentAlphaConflict}}{{{float(sim['alpha_conflict']):.1f}}}",
                f"\\newcommand{{\\AgentBetaHeat}}{{{float(sim['beta_heat']):.1f}}}",
                f"\\newcommand{{\\AgentGammaConsensus}}{{{float(sim['gamma_consensus']):.1f}}}",
                f"\\newcommand{{\\AgentPolarityKappa}}{{{float(sim['polarity_kappa']):.0f}}}",
                f"\\newcommand{{\\AgentExternalRate}}{{{eta_eff:.0f}}}",
                f"\\newcommand{{\\AgentDecay}}{{{float(sim['external_decay']):.3f}}}",
                f"\\newcommand{{\\AgentDepthLambda}}{{{float(sim['antagonistic_depth_lambda']):.1f}}}",
                f"\\newcommand{{\\AgentTrainingCascades}}{{{n_train}}}",
            ]
    macro_lines.extend(ablation_macro_lines("reference", "AgentReference"))
    macro_lines.extend(ablation_macro_lines("no_conflict", "AgentNoConflict"))
    macro_lines.extend(ablation_macro_lines("no_heat", "AgentNoHeat"))
    macro_lines.extend(
        ablation_macro_lines("no_depth_preference", "AgentNoDepthExponent")
    )
    macro_lines.extend(
        ablation_macro_lines("no_depth_targeting", "AgentNoDepthChannel")
    )
    macros_path.write_text(
        "\n".join(macro_lines) + "\n",
        encoding="utf-8",
    )

    provenance_path = output / "table_agent_parameter_provenance.tex"
    provenance_path.write_text(
        "\n".join(
            [
                r"\begin{table*}[t]",
                r"\caption{Parameter provenance and freeze boundary. Legacy values "
                r"recover the declared controlled mechanism; training-estimated "
                r"values support independent fidelity analyses.}",
                r"\label{tab:agent-parameter-provenance}",
                r"\centering",
                r"\footnotesize",
                r"\setlength{\tabcolsep}{3.5pt}",
                r"\renewcommand{\arraystretch}{1.08}",
                r"\begin{tabularx}{\textwidth}{",
                r"  >{\raggedright\arraybackslash}p{0.18\textwidth}",
                r"  >{\raggedright\arraybackslash}p{0.23\textwidth}",
                r"  >{\raggedright\arraybackslash}p{0.22\textwidth}",
                r"  >{\raggedright\arraybackslash}X}",
                r"\toprule",
                r"Parameter family & Representative values & Origin & Outcome-use boundary \\",
                r"\midrule",
                (
                    r"Impulse and polarity & "
                    rf"$I_{{\rm base}}={float(sim['base_impulse']):.2f}$, "
                    rf"$\alpha={float(sim['alpha_conflict']):.1f}$, "
                    rf"$\beta={float(sim['beta_heat']):.1f}$, "
                    rf"$\gamma={float(sim['gamma_consensus']):.1f}$, "
                    rf"$\kappa_{{\rm pol}}={float(sim['polarity_kappa']):.0f}$ & "
                    r"Theory-specified legacy constants & No current test outcomes; "
                    r"the legacy configuration is paper-aligned \\"
                ),
                (
                    r"Community activation scale & $\tau_{{\rm base},c}$ and "
                    r"$m_{{\rm early},c}$ & Appendix-B community summaries & Frozen "
                    r"before reviewer-validation runs; not re-estimated on test posts \\"
                ),
                (
                    r"Action, depth, and traffic & Treatment multipliers, depth "
                    rf"routing, $\eta_{{\rm eff}}={eta_eff:.0f}$, "
                    rf"$\delta={float(sim['external_decay']):.3f}$ & "
                    r"Legacy paper-aligned calibration & Original paper targets were "
                    r"used; these values provide recovery, not independent validation \\"
                ),
                (
                    r"Data-estimated transition policies & Arrival decay, root share, "
                    r"depth decay, author reuse, correction rates & Chronological "
                    rf"training split ({n_train} cascades) & No paper targets and no "
                    r"validation/test outcomes used for fitting \\"
                ),
                (
                    r"Platform adapters & Ranking, viewport, exposure, and target-platform "
                    r"transition parameters & Target training and validation partitions & "
                    r"Target test partition opened only after adapter selection \\"
                ),
                r"\bottomrule",
                r"\end{tabularx}",
                r"\end{table*}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    copied_ablation = output / "table_agent_channel_ablation.tex"
    copied_ablation.write_text(
        ablation_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "inputs": {
            str(config_path.relative_to(ROOT)): _sha256(config_path),
            str(model_path.relative_to(ROOT)): _sha256(model_path),
            str(ablation_path.relative_to(ROOT)): _sha256(ablation_path),
            str(ablation_summary_path.relative_to(ROOT)): _sha256(
                ablation_summary_path
            ),
        },
        "outputs": {
            path.name: _sha256(path)
            for path in (macros_path, provenance_path, copied_ablation)
        },
        "training_cascades": n_train,
        "legacy_uses_paper_targets": bool(
            config.get("calibration_scope", {}).get("uses_paper_table_targets")
        ),
        "training_models_use_paper_targets": bool(
            models.get("uses_paper_target_values")
        ),
    }
    (output / "agent_modeling_tables.manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()

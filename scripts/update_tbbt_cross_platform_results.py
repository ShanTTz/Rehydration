from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TBBT_DIR = ROOT / "artifacts" / "interventions" / "tbbt_qualified_controls"
EXPANSION_DIR = (
    ROOT
    / "artifacts"
    / "external_validation"
    / "cross_platform_content_expanded"
)
MATCHED_DIR = (
    ROOT
    / "artifacts"
    / "external_validation"
    / "lemmy_content_matched_expanded"
)
PAPER_TABLES = ROOT / "PAPER_READY_RESULT_TABLES.md"
LATEST_STATUS = ROOT / "LATEST_STATUS.md"
LATEST_JSON = ROOT / "artifacts" / "provenance" / "latest_status.json"
REVIEW_MATRIX = ROOT / "REVIEW_RESPONSE_MATRIX.md"
CHANGELOG = ROOT / "CHANGELOG.md"
LATEX_DIR = ROOT / "manuscript" / "paper_ready_tables"
PAPER_START = "<!-- TBBT_CROSS_PLATFORM_EXPANDED_RESULTS_START -->"
PAPER_END = "<!-- TBBT_CROSS_PLATFORM_EXPANDED_RESULTS_END -->"
STATUS_START = "<!-- TBBT_CROSS_PLATFORM_EXPANDED_STATUS_START -->"
STATUS_END = "<!-- TBBT_CROSS_PLATFORM_EXPANDED_STATUS_END -->"
MATRIX_START = "<!-- TBBT_CROSS_PLATFORM_EXPANDED_MATRIX_START -->"
MATRIX_END = "<!-- TBBT_CROSS_PLATFORM_EXPANDED_MATRIX_END -->"
CHANGELOG_START = "<!-- TBBT_CROSS_PLATFORM_EXPANDED_CHANGELOG_START -->"
CHANGELOG_END = "<!-- TBBT_CROSS_PLATFORM_EXPANDED_CHANGELOG_END -->"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_block(path: Path, start: str, end: str, block: str) -> None:
    current = path.read_text(encoding="utf-8") if path.is_file() else ""
    rendered = f"{start}\n\n{block.strip()}\n\n{end}"
    if start in current and end in current:
        prefix, remainder = current.split(start, 1)
        _, suffix = remainder.split(end, 1)
        updated = prefix.rstrip() + "\n\n" + rendered + suffix
    else:
        updated = current.rstrip() + "\n\n" + rendered + "\n"
    path.write_text(updated, encoding="utf-8")


def _effect_text(effect: dict[str, Any]) -> str:
    return (
        f"{float(effect['percent_effect']):+.1f}% "
        f"[{float(effect['percent_ci_low']):+.1f}%, "
        f"{float(effect['percent_ci_high']):+.1f}%]"
    )


def _model_label(model: str) -> str:
    return {
        "lemmy_structure_selected_bdmtf": "Lemmy 结构适配 BDMTF",
        "lemmy_target_fitted_bdmtf": "Lemmy 目标拟合 BDMTF",
        "hackernews_zero_shot_bdmtf": "HN 零样本 BDMTF",
        "lemmy_empirical_bootstrap": "经验重采样",
        "lemmy_branching_process": "分支过程",
        "hawkes": "Hawkes",
    }.get(model, model)


def _refresh_latest_status_rows(
    text: str,
    tbbt: dict[str, Any],
    expansion: dict[str, Any],
    matched: dict[str, Any],
) -> str:
    model = matched["exact_url_model_result"]
    content_row = (
        f"| Lemmy 内容配对跨平台验证 | "
        f"{matched['matching']['exact_post_pairs']} 个相同 URL 帖子对、"
        f"{matched['matching']['exact_structural_pairs']} 个完整树对；适配改善 "
        f"{model['structure_selected_improvement_over_zero_shot']:.3f} "
        f"[{model['structure_selected_improvement_ci'][0]:.3f}, "
        f"{model['structure_selected_improvement_ci'][1]:.3f}] | 完成 |"
    )
    tbbt_row = (
        f"| TBBT 合格对照真实干预 | "
        f"{tbbt['qualified_interventions']}/{tbbt['eligible_interventions']} "
        "通过门槛；隔离 -15.5%，帖子移除 +17.7% | 完成 |"
    )
    voat_row = (
        f"| Reddit→Voat 同作者迁移 | "
        f"{expansion['voat']['matched_authors']} 名同匿名作者；"
        "两组相似度增益约 0.049 | 完成 |"
    )
    refreshed: list[str] = []
    inserted = False
    for line in text.splitlines():
        if line.startswith("| Lemmy 内容配对跨平台验证 |"):
            refreshed.extend([content_row, tbbt_row, voat_row])
            inserted = True
            continue
        if line.startswith("| TBBT 合格对照真实干预 |"):
            if not inserted:
                refreshed.append(tbbt_row)
            continue
        if line.startswith("| Reddit→Voat 同作者迁移 |"):
            if not inserted:
                refreshed.append(voat_row)
            continue
        if line.startswith("| TBBT 确认性对照与干预预测保真度 |"):
            refreshed.append(
                "| TBBT 干预预测保真度 | "
                "合格对照真实效应已完成；尚需冻结 BDMTF 预测后比较 |"
            )
            continue
        if line.startswith("| Voat 扩展 |"):
            continue
        if line.startswith("- 相同 URL 帖子评论量的跨平台 Spearman"):
            post = matched["post_level_result"]
            refreshed.append(
                "- 相同 URL 帖子评论量的跨平台 Spearman 相关为 "
                f"{post['spearman_comment_count']:.3f}"
                f"（p={post['spearman_p_value']:.3f}）；"
                "内容身份相同不代表互动动力学相同。"
            )
            continue
        if line.startswith("- `PAPER_READY_RESULT_TABLES.md`："):
            refreshed.append(
                "- `PAPER_READY_RESULT_TABLES.md`：论文可用表 1--14。"
            )
            continue
        if line.startswith(
            "- `artifacts/external_validation/lemmy_content_matched/`："
        ):
            refreshed.append(
                "- `artifacts/external_validation/"
                "lemmy_content_matched_expanded/`：扩大后的内容配对与适配。"
            )
            refreshed.append(
                "- `artifacts/external_validation/"
                "cross_platform_content_expanded/`：HN/Lemmy 收集审计与 Voat 迁移。"
            )
            continue
        refreshed.append(line)
    return "\n".join(refreshed) + "\n"


def _build_tbbt_report(
    manifest: dict[str, Any],
    diagnostics: list[dict[str, str]],
    effects: list[dict[str, str]],
) -> None:
    primary = [
        row
        for row in effects
        if row["qualified"].lower() == "true" and row["window_days"] == "30"
    ]
    rows = "\n".join(
        (
            f"| {row['community']} | {row['intervention_type']} | "
            f"{float(row['percent_effect']):+.1f}% | "
            f"[{float(row['percent_ci_low']):+.1f}%, "
            f"{float(row['percent_ci_high']):+.1f}%] | "
            f"{float(row['placebo_p_value']):.3f} |"
        )
        for row in primary
    )
    failed = "\n".join(
        f"| {row['community']} | {row['issues'] or 'passed'} |"
        for row in diagnostics
        if row["qualified"].lower() != "true"
    )
    groups = manifest["aggregate_effects_by_intervention_type"]
    report = f"""# TBBT Qualified-Control Natural Experiment

## Purpose

This analysis estimates real moderation responses without treating TBBT `OUT`
activity as an untreated control. TBBT supplies intervention definitions and
an independent source-overlap audit; Arctic Shift supplies treated and donor
daily comment counts on the same measurement scale.

## Qualification

- Eligible interventions: {manifest['eligible_interventions']}
- Passed all pre-outcome gates: {manifest['qualified_interventions']}
- Selection used post-treatment outcomes: no
- Primary window: {manifest['primary_window_days']} days
- Protocol version: {manifest['protocol']['version']}

## Qualified Effects

| Community | Intervention | Activity effect | 95% CI | In-time placebo p |
| --- | --- | ---: | ---: | ---: |
{rows}

## Type-Stratified Summary

| Intervention type | Events | Activity effect [95% CI] | I-squared |
| --- | ---: | ---: | ---: |
| Quarantine | {groups['quarantine']['qualified_interventions']} | {_effect_text(groups['quarantine'])} | {groups['quarantine']['i_squared']:.1f}% |
| Post removal | {groups['post_removal']['qualified_interventions']} | {_effect_text(groups['post_removal'])} | {groups['post_removal']['i_squared']:.1f}% |
| Pooled | {manifest['aggregate_effect']['qualified_interventions']} | {_effect_text(manifest['aggregate_effect'])} | {manifest['aggregate_effect']['i_squared']:.1f}% |

The pooled interval crosses zero because intervention types have opposite
responses. The type-stratified estimates support mechanism-dependent
intervention effects, but intervention type was not randomized across
communities and dates.

## Failed Pre-Outcome Gates

| Community | Failed gate(s) |
| --- | --- |
{failed}

## Evidence Boundary

The qualified synthetic controls identify within-event deviations under
parallel-counterfactual assumptions. The estimates do not show that every
moderation action has the same effect, and the cross-type contrast must not be
described as a randomized causal comparison.
"""
    (TBBT_DIR / "TBBT_QUALIFIED_CONTROL_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )


def _build_cross_platform_report(
    expansion: dict[str, Any],
    matched: dict[str, Any],
    ranking: list[dict[str, str]],
) -> None:
    exact_rows = [
        row for row in ranking if row.get("match_type") == "exact_url"
    ]
    model_rows = "\n".join(
        (
            f"| {_model_label(row['model'])} | "
            f"{float(row['mean_normalized_mae']):.3f} | "
            f"[{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}] |"
        )
        for row in sorted(
            exact_rows,
            key=lambda row: float(row["mean_normalized_mae"]),
        )
    )
    voat_rows = "\n".join(
        (
            f"| {row['intervention']} | {row['matched_authors']} | "
            f"{row['similarity_gain']:.3f} | "
            f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] | "
            f"{row['permutation_p_value']:.3f} |"
        )
        for row in expansion["voat"]["interventions"]
    )
    result = matched["exact_url_model_result"]
    report = f"""# Expanded HN-Lemmy and Voat Content Validation

## Purpose

The exact-URL design holds observed story identity fixed while testing whether
bounded target-platform adaptation improves BDMTF portability between threaded
platforms. The Voat analysis separately tests semantic continuity for the same
pseudonymous users after migration.

## Collection and Matching

- Outcome-blind HN candidate roots: {expansion['hackernews']['candidate_roots']:,}
- Targeted exact-URL pairs: {expansion['matching']['target_exact_pairs']:,}
- Collector-complete exact pairs: {expansion['matching']['complete_exact_pairs']:,}
- Final one-to-one exact post pairs: {matched['matching']['exact_post_pairs']:,}
- Final complete exact tree pairs: {matched['matching']['exact_structural_pairs']:,}
- High-threshold semantic tree pairs, reported separately: {matched['matching']['semantic_structural_pairs']:,}

The collector audit and final validation counts differ slightly because the
validation stage reruns one-to-one matching against the merged complete-tree
pool; it does not reuse outcomes to select pairs.

## Held-Out Model Comparison

| Model | Mean normalized MAE | 95% CI |
| --- | ---: | ---: |
{model_rows}

Structure-selected BDMTF improves over HN zero-shot transfer by
**{result['structure_selected_improvement_over_zero_shot']:.3f}**
[{result['structure_selected_improvement_ci'][0]:.3f},
{result['structure_selected_improvement_ci'][1]:.3f}].

## Voat Same-Author Content Transfer

| Migration | Matched authors | Similarity gain | 95% CI | Permutation p |
| --- | ---: | ---: | ---: | ---: |
{voat_rows}

## Evidence Boundary

The HN-Lemmy result supports bounded adaptation between the two tested
threaded platforms. It does not establish zero-shot universality or transfer
to feed-first platforms. The Voat result tests author-level semantic
continuity, not exact-story matching or a causal platform effect.
"""
    (EXPANSION_DIR / "CROSS_PLATFORM_EXPANSION_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    tbbt_path = TBBT_DIR / "tbbt_qualified_control_manifest.json"
    diagnostics_path = TBBT_DIR / "qualification_diagnostics.csv"
    effects_path = TBBT_DIR / "intervention_effects.csv"
    expansion_path = EXPANSION_DIR / "expansion_manifest.json"
    matched_path = MATCHED_DIR / "content_matched_manifest.json"
    ranking_path = MATCHED_DIR / "model_ranking.csv"
    required = [
        tbbt_path,
        diagnostics_path,
        effects_path,
        expansion_path,
        matched_path,
        ranking_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing completed experiment files:\n" + "\n".join(missing))

    tbbt = _read_json(tbbt_path)
    diagnostics = _read_csv(diagnostics_path)
    effects = _read_csv(effects_path)
    expansion = _read_json(expansion_path)
    matched = _read_json(matched_path)
    ranking = _read_csv(ranking_path)
    _build_tbbt_report(tbbt, diagnostics, effects)
    _build_cross_platform_report(expansion, matched, ranking)

    groups = tbbt["aggregate_effects_by_intervention_type"]
    model = matched["exact_url_model_result"]
    paper_block = f"""## 表 13：TBBT 合格对照真实干预

**作用：** 用真实平台干预检验 BDMTF 所强调的“平台机制会改变互动动力学”。处理组与供体组统一使用 Arctic Shift 日评论计数，TBBT 用于冻结干预定义和独立数据源重叠审计；受影响用户的 `OUT` 活动从不充当未处理对照。

| 干预类型 | 通过门槛事件 | 30 天互动量效应 [95% CI] | 事件方向 | I² |
| --- | ---: | ---: | ---: | ---: |
| 隔离（quarantine） | {groups['quarantine']['qualified_interventions']} | {_effect_text(groups['quarantine'])} | 3/3 下降 | {groups['quarantine']['i_squared']:.1f}% |
| 帖子移除（post removal） | {groups['post_removal']['qualified_interventions']} | {_effect_text(groups['post_removal'])} | 2/2 上升 | {groups['post_removal']['i_squared']:.1f}% |
| 合并汇总 | {tbbt['aggregate_effect']['qualified_interventions']} | {_effect_text(tbbt['aggregate_effect'])} | 3/5 下降 | {tbbt['aggregate_effect']['i_squared']:.1f}% |

8 个可评估干预中 5 个通过来源重叠、验证窗 RMSPE、完整干预前 RMSPE、趋势等效性和供体权重门槛。总体平均区间跨零，是因为两类干预方向相反；可发表结论是**真实干预具有机制类型异质性**，不能写成所有审核都会统一降低互动。

**论文用途：** 这补上了原稿缺少真实干预证据的问题，并支持把审核、退出和曝光机制显式纳入 BDMTF。干预类型并未跨社区随机分配，因此分层差异不能写成随机化的类型间因果比较。

## 表 14：扩展 HN-Lemmy/Voat 内容配对验证

**作用：** 在发布同一 URL 的 HN 与 Lemmy 帖子之间控制可观测内容身份，检验有限目标平台数据能否修正结构参数直接迁移造成的失配；Voat 部分检验同匿名作者迁移后的语义连续性。

表 12 的 44 个完整树对保留为协议冻结试点；本表的 724 对结果是应进入正文的确认性扩展结果。

| 项目 | 扩展规模 | 主要结果 |
| --- | ---: | --- |
| HN 候选根帖 | {expansion['hackernews']['candidate_roots']:,} | 按 URL、时间和评论上限进行结果盲筛选 |
| HN-Lemmy 精确 URL | {matched['matching']['exact_post_pairs']:,} 帖子对 | {matched['matching']['exact_structural_pairs']:,} 对具有双端完整讨论树 |
| 结构适配 BDMTF | {matched['matching']['exact_structural_pairs']:,} 测试对 | 相对 HN 零样本误差改善 **{model['structure_selected_improvement_over_zero_shot']:.3f}** [{model['structure_selected_improvement_ci'][0]:.3f}, {model['structure_selected_improvement_ci'][1]:.3f}] |
| Reddit→Voat 同作者 | {expansion['voat']['matched_authors']:,} 人 | 两个迁移样本的内容相似度增益均约 0.049，区间不跨零，置换 `p≈0.001` |

**论文用途：** HN-Lemmy 结果只有在改善区间高于 0 时支持“BDMTF 可用有限目标平台数据适配到另一线程式平台”；不支持零样本普适性。Voat 结果支持迁移后语义意图具有可重用连续性，但不是精确故事配对，也不识别平台因果效应。
"""
    _replace_block(PAPER_TABLES, PAPER_START, PAPER_END, paper_block)

    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    tbbt_tex = rf"""\begin{{table}}[t]
\centering
\small
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lrrr}}
\toprule
Intervention & Events & 30-day activity effect [95\% CI] & $I^2$ \\
\midrule
Quarantine & {groups['quarantine']['qualified_interventions']} & {_effect_text(groups['quarantine']).replace('%', r'\%')} & {groups['quarantine']['i_squared']:.1f}\% \\
Post removal & {groups['post_removal']['qualified_interventions']} & {_effect_text(groups['post_removal']).replace('%', r'\%')} & {groups['post_removal']['i_squared']:.1f}\% \\
Pooled & {tbbt['aggregate_effect']['qualified_interventions']} & {_effect_text(tbbt['aggregate_effect']).replace('%', r'\%')} & {tbbt['aggregate_effect']['i_squared']:.1f}\% \\
\bottomrule
\end{{tabular}}
}}
\caption{{Qualified-control estimates for real TBBT interventions. Five of eight eligible events pass source-overlap, pre-period predictive-skill, trend-equivalence, and donor-weight gates. Opposite type-specific responses make the pooled interval cross zero.}}
\label{{tab:tbbt-qualified-controls}}
\end{{table}}
"""
    cross_tex = rf"""\begin{{table}}[t]
\centering
\small
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lrr}}
\toprule
Validation component & Sample & Result \\
\midrule
Exact HN--Lemmy posts & {matched['matching']['exact_post_pairs']} & {matched['matching']['exact_structural_pairs']} complete tree pairs \\
Adapted vs. HN zero-shot & {matched['matching']['exact_structural_pairs']} & {model['structure_selected_improvement_over_zero_shot']:.3f} [{model['structure_selected_improvement_ci'][0]:.3f}, {model['structure_selected_improvement_ci'][1]:.3f}] MAE gain \\
Reddit--Voat same author & {expansion['voat']['matched_authors']} & $\sim$0.049 similarity gain \\
\bottomrule
\end{{tabular}}
}}
\caption{{Expanded content-controlled portability evidence. Exact-URL HN--Lemmy pairs and same-pseudonymous-author Reddit--Voat pairs are separate designs and support different claims.}}
\label{{tab:cross-platform-content-expanded}}
\end{{table}}
"""
    tbbt_tex_path = LATEX_DIR / "table_tbbt_qualified_controls.tex"
    cross_tex_path = LATEX_DIR / "table_cross_platform_content_expanded.tex"
    tbbt_tex_path.write_text(tbbt_tex, encoding="utf-8")
    cross_tex_path.write_text(cross_tex, encoding="utf-8")
    preview_path = LATEX_DIR / "preview_tbbt_cross_platform.tex"
    preview_path.write_text(
        "\n".join(
            [
                r"\documentclass[10pt]{article}",
                r"\usepackage[letterpaper,margin=0.65in]{geometry}",
                r"\usepackage{booktabs,graphicx,microtype}",
                r"\begin{document}",
                r"\input{table_tbbt_qualified_controls.tex}",
                r"\input{table_cross_platform_content_expanded.tex}",
                r"\end{document}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    status_sentence = (
        "最新结果：TBBT 8 个可评估干预中 5 个通过合格对照门槛，"
        f"隔离类 30 天互动量 {_effect_text(groups['quarantine'])}，"
        f"帖子移除类 {_effect_text(groups['post_removal'])}；扩大后的 "
        f"HN-Lemmy 验证包含 {matched['matching']['exact_post_pairs']} 个同 URL "
        f"帖子对和 {matched['matching']['exact_structural_pairs']} 个完整树对，"
        f"BDMTF 有界适配相对零样本误差改善 "
        f"{model['structure_selected_improvement_over_zero_shot']:.3f} "
        f"[{model['structure_selected_improvement_ci'][0]:.3f}, "
        f"{model['structure_selected_improvement_ci'][1]:.3f}]；Voat 同作者"
        f"迁移样本为 {expansion['voat']['matched_authors']} 人。"
    )
    status_block = f"""## 2026-07-30 最新补强结果

> {status_sentence}

- TBBT 合格对照估计已完成；总体区间跨零，类型分层显示隔离与帖子移除方向相反。
- HN-Lemmy 内容控制样本已从 44 个完整树对扩展到 {matched['matching']['exact_structural_pairs']} 对。
- Voat 同匿名作者内容迁移验证已完成，不能替代精确故事配对或平台因果估计。
- 尚待：TBBT 干预预测保真度、1,575 个冻结意图任务、伦理审批后的真人 RCT。
"""
    _replace_block(LATEST_STATUS, STATUS_START, STATUS_END, status_block)
    refreshed_status = _refresh_latest_status_rows(
        LATEST_STATUS.read_text(encoding="utf-8"),
        tbbt,
        expansion,
        matched,
    )
    LATEST_STATUS.write_text(refreshed_status, encoding="utf-8")
    latest_lines = refreshed_status.splitlines()
    for index, line in enumerate(latest_lines):
        if line.startswith("> "):
            latest_lines[index] = f"> {status_sentence}"
            break
    LATEST_STATUS.write_text("\n".join(latest_lines) + "\n", encoding="utf-8")

    latest = _read_json(LATEST_JSON) if LATEST_JSON.is_file() else {}
    latest["generated_at"] = datetime.now(timezone.utc).isoformat()
    latest["latest_one_sentence"] = status_sentence
    latest["tbbt_qualified_controls"] = tbbt
    latest["cross_platform_content_expansion"] = expansion
    latest["lemmy_content_matched_validation_expanded"] = matched
    latest["remaining_external_experiments"] = [
        "tbbt_intervention_prediction_fidelity",
        "frozen_agent_intents_1575",
        "ethics_approved_human_rct",
    ]
    latest["pending_experiments"] = [
        {
            "id": "tbbt_intervention_prediction_fidelity",
            "status": "qualified_real_effects_ready_for_simulator_prediction",
        },
        {
            "id": "frozen_agent_intents_1575",
            "status": "dry_run_ready",
        },
        {
            "id": "human_rct",
            "status": "blocked_on_ethics_and_recruitment",
        },
    ]
    latest_sources = {
        str(item.get("path")): item
        for item in latest.get("sources", [])
        if item.get("path")
    }
    for path in required:
        relative = path.relative_to(ROOT).as_posix()
        latest_sources[relative] = {
            "path": relative,
            "sha256": _sha256(path),
        }
    latest["sources"] = list(latest_sources.values())
    LATEST_JSON.write_text(
        json.dumps(latest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    matrix_block = f"""| Reviewer concern | New evidence | Resolution boundary |
| --- | --- | --- |
| No real intervention validation | {tbbt['qualified_interventions']}/{tbbt['eligible_interventions']} TBBT events pass outcome-blind control gates; quarantine and post-removal estimates are reported separately | Resolved for observed Reddit interventions under synthetic-control assumptions; human RCT remains pending |
| Cross-platform evidence is too small | {matched['matching']['exact_post_pairs']} exact HN-Lemmy post pairs, {matched['matching']['exact_structural_pairs']} complete tree pairs, and {expansion['voat']['matched_authors']} Reddit-Voat same-author pairs | Resolved for bounded threaded-platform adaptation and author continuity; no zero-shot or feed-platform universality claim |
"""
    _replace_block(
        REVIEW_MATRIX,
        MATRIX_START,
        MATRIX_END,
        matrix_block,
    )
    changelog_block = """## 2026-07-30 qualified interventions and expanded content matching

- Added outcome-blind TBBT synthetic controls with independent source-overlap, predictive-skill, pretrend-equivalence, and donor-weight gates.
- Added type-stratified real intervention estimates while retaining the non-significant pooled estimate.
- Expanded HN-Lemmy exact content matching and complete-tree validation.
- Added same-pseudonymous-author Reddit-to-Voat content-continuity analysis.
- Added paper-ready Markdown, LaTeX tables, reports, manifests, and focused tests.
"""
    _replace_block(
        CHANGELOG,
        CHANGELOG_START,
        CHANGELOG_END,
        changelog_block,
    )

    outputs = [
        TBBT_DIR / "TBBT_QUALIFIED_CONTROL_REPORT.md",
        EXPANSION_DIR / "CROSS_PLATFORM_EXPANSION_REPORT.md",
        PAPER_TABLES,
        LATEST_STATUS,
        LATEST_JSON,
        REVIEW_MATRIX,
        CHANGELOG,
        tbbt_tex_path,
        cross_tex_path,
        preview_path,
    ]
    manifest = {
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            path.relative_to(ROOT).as_posix(): _sha256(path)
            for path in required
        },
        "outputs": {
            path.relative_to(ROOT).as_posix(): _sha256(path)
            for path in outputs
        },
    }
    (EXPANSION_DIR / "paper_integration_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

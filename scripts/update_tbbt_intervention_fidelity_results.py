from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = (
    ROOT / "artifacts" / "interventions" / "tbbt_intervention_fidelity"
)
SUMMARY_PATH = RESULT_DIR / "fidelity_summary.json"
COMPARISON_PATH = RESULT_DIR / "prediction_observed_comparison.csv"
FREEZE_PATH = RESULT_DIR / "prediction_freeze_manifest.json"

PAPER_START = "<!-- TBBT_INTERVENTION_FIDELITY_START -->"
PAPER_END = "<!-- TBBT_INTERVENTION_FIDELITY_END -->"
STATUS_START = "<!-- TBBT_INTERVENTION_FIDELITY_STATUS_START -->"
STATUS_END = "<!-- TBBT_INTERVENTION_FIDELITY_STATUS_END -->"
MATRIX_START = "<!-- TBBT_INTERVENTION_FIDELITY_MATRIX_START -->"
MATRIX_END = "<!-- TBBT_INTERVENTION_FIDELITY_MATRIX_END -->"


def _replace_block(path: Path, start: str, end: str, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    rendered = f"{start}\n\n{block.strip()}\n\n{end}"
    if start in text and end in text:
        left, rest = text.split(start, 1)
        _, right = rest.split(end, 1)
        text = left.rstrip() + "\n\n" + rendered + right
    else:
        text = text.rstrip() + "\n\n" + rendered + "\n"
    path.write_text(text, encoding="utf-8")


def _signed(value: float, digits: int = 1) -> str:
    return f"{value:+.{digits}f}"


def _event_rows(frame: pd.DataFrame) -> str:
    rows: list[str] = []
    for row in frame.itertuples(index=False):
        rows.append(
            "| {community} | {kind} | {pred}% [{low}, {high}] | "
            "{obs}% [{obs_low}, {obs_high}] | {error:.2f} | {covered} |".format(
                community=row.community,
                kind=(
                    "隔离"
                    if row.intervention_type == "quarantine"
                    else "帖子移除"
                ),
                pred=_signed(row.predicted_percent_effect),
                low=_signed(row.prediction_percent_low),
                high=_signed(row.prediction_percent_high),
                obs=_signed(row.percent_effect),
                obs_low=_signed(row.percent_ci_low),
                obs_high=_signed(row.percent_ci_high),
                error=row.absolute_error_pp,
                covered=(
                    "是" if row.observed_in_prediction_interval else "否"
                ),
            )
        )
    return "\n".join(rows)


def _write_latex(
    summary: dict[str, object],
    frame: pd.DataFrame,
) -> None:
    rows: list[str] = []
    for row in frame.itertuples(index=False):
        community = str(row.community).replace("_", r"\_")
        kind = (
            "Quarantine"
            if row.intervention_type == "quarantine"
            else "Post rem."
        )
        rows.append(
            f"{community} & {kind} & "
            f"{_signed(row.predicted_percent_effect)} "
            f"[{_signed(row.prediction_percent_low)}, "
            f"{_signed(row.prediction_percent_high)}] & "
            f"{_signed(row.percent_effect)} "
            f"[{_signed(row.percent_ci_low)}, "
            f"{_signed(row.percent_ci_high)}] \\\\"
        )
    table = r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{llrr}
\toprule
Community & Action & Prediction [95\% PI] & Effect [95\% CI] \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
}
\caption{Frozen BDMTF predictions and qualified-control TBBT effects over
30 days. Prediction hashes are sealed before scoring opens observed outcomes.}
\label{tab:tbbt-prediction-fidelity}
\end{table}
"""
    improvement = summary[
        "mae_improvement_over_zero_percentage_points"
    ]
    macros = "\n".join(
        [
            f"\\newcommand{{\\TBBTFidelityEvents}}{{{summary['n_interventions']}}}",
            f"\\newcommand{{\\TBBTFidelityDirection}}{{{100 * summary['direction_accuracy']:.1f}}}",
            f"\\newcommand{{\\TBBTFidelityMAE}}{{{summary['mae_percentage_points']:.2f}}}",
            f"\\newcommand{{\\TBBTZeroMAE}}{{{summary['zero_effect_mae_percentage_points']:.2f}}}",
            f"\\newcommand{{\\TBBTMAEGain}}{{{improvement['mean']:.2f}}}",
            f"\\newcommand{{\\TBBTMAEGainLow}}{{{improvement['ci_low']:.2f}}}",
            f"\\newcommand{{\\TBBTMAEGainHigh}}{{{improvement['ci_high']:.2f}}}",
            f"\\newcommand{{\\TBBTPredictionCoverage}}{{{100 * summary['observed_point_prediction_interval_coverage']:.1f}}}",
            f"\\newcommand{{\\TBBTObservedCICoverage}}{{{100 * summary['prediction_point_observed_ci_coverage']:.1f}}}",
            f"\\newcommand{{\\TBBTSpearman}}{{{summary['spearman_rank_correlation']:.3f}}}",
            "",
        ]
    )
    targets = [
        ROOT / "manuscript" / "paper_ready_tables",
        ROOT / "manuscript" / "original_pdf_revision",
    ]
    for directory in targets:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "table_tbbt_intervention_fidelity.tex").write_text(
            table,
            encoding="utf-8",
        )
        (directory / "tbbt_intervention_fidelity_macros.tex").write_text(
            macros,
            encoding="utf-8",
        )
    qualified = (
        ROOT
        / "manuscript"
        / "paper_ready_tables"
        / "table_tbbt_qualified_controls.tex"
    )
    if qualified.is_file():
        (
            ROOT
            / "manuscript"
            / "original_pdf_revision"
            / qualified.name
        ).write_text(qualified.read_text(encoding="utf-8"), encoding="utf-8")


def _update_markdown(
    summary: dict[str, object],
    frame: pd.DataFrame,
    freeze: dict[str, object],
) -> None:
    improvement = summary[
        "mae_improvement_over_zero_percentage_points"
    ]
    paper_block = f"""## 表 15：TBBT 冻结干预预测保真度

**作用：** 检验 BDMTF 的机制映射能否在不读取干预后结果的预测入口中，仅凭干预前作者进入、跨线程切换和既有冻结参数，预判 5 个合格 TBBT 真实干预的方向、幅度与区间。预测先以哈希 `{str(freeze['freeze_id'])[:12]}` 封存，之后由独立评分入口解封真实效应。

| 社区 | 干预 | 冻结预测 [95% PI] | 真实效应 [95% CI] | 绝对误差（百分点） | 真实点在预测区间内 |
| --- | --- | ---: | ---: | ---: | ---: |
{_event_rows(frame)}

汇总方向准确率为 **{100 * summary['direction_accuracy']:.1f}%（5/5）**，MAE 为 **{summary['mae_percentage_points']:.2f} 个百分点**，而零效应基线 MAE 为 **{summary['zero_effect_mae_percentage_points']:.2f}**。BDMTF 相对零效应基线改善 **{improvement['mean']:.2f}** [{improvement['ci_low']:.2f}, {improvement['ci_high']:.2f}] 个百分点；真实点落入预测区间的比例为 **{100 * summary['observed_point_prediction_interval_coverage']:.1f}%（4/5）**，冻结预测落入真实 95% CI 的比例为 **{100 * summary['prediction_point_observed_ci_coverage']:.1f}%（5/5）**，秩相关为 **{summary['spearman_rank_correlation']:.3f}**。

**论文用途：** 结果支持 BDMTF 不仅能描述真实审核效应，还能以固定机制映射预测不同审核类型的响应方向和近似幅度。由于分析者在协议冻结前已经看过聚合真实结果，这属于**回顾性协议冻结检验**，不能表述成前瞻、预注册或分析者盲测；样本只有 5 个事件，也不能据此宣称普遍审核规律。"""
    _replace_block(
        ROOT / "PAPER_READY_RESULT_TABLES.md",
        PAPER_START,
        PAPER_END,
        paper_block,
    )

    status_block = f"""## 2026-07-30 TBBT 预测保真度

TBBT 冻结机制预测已完成：5/5 方向正确，MAE {summary['mae_percentage_points']:.2f} 个百分点，较零效应基线改善 {improvement['mean']:.2f} [{improvement['ci_low']:.2f}, {improvement['ci_high']:.2f}] 个百分点；4/5 真实点落入预测区间，5/5 预测点落入真实效应 95% CI。

证据等级为回顾性协议冻结，不是前瞻盲测。当前仅余 1,575 个冻结智能体意图任务和伦理审批后的真人 RCT。"""
    _replace_block(
        ROOT / "LATEST_STATUS.md",
        STATUS_START,
        STATUS_END,
        status_block,
    )
    latest_path = ROOT / "LATEST_STATUS.md"
    text = latest_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("> 最新结果："):
            lines[index] = (
                "> 最新结果：TBBT 5 个合格真实干预的冻结 BDMTF 预测达到 "
                "**5/5 方向正确**、MAE **3.15 个百分点**，相对零效应"
                "基线改善 **13.45 [6.38, 18.02]** 个百分点；4/5 真实点"
                "落入预测区间。"
            )
    text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    text = text.replace(
        "| TBBT 合格对照真实干预 | 5/8 通过门槛；隔离 -15.5%，帖子移除 +17.7% | 完成 |",
        "| TBBT 合格对照与预测保真度 | 5/8 通过门槛；5/5 预测方向正确，MAE 3.15 个百分点 | 完成 |",
    )
    text = text.replace(
        "| TBBT 干预预测保真度 | 合格对照真实效应已完成；尚需冻结 BDMTF 预测后比较 |\n",
        "",
    )
    text = text.replace(
        "- 尚待：TBBT 干预预测保真度、1,575 个冻结意图任务、伦理审批后的真人 RCT。",
        "- 尚待：1,575 个冻结意图任务、伦理审批后的真人 RCT。",
    )
    text = text.replace(
        "`PAPER_READY_RESULT_TABLES.md`：论文可用表 1--14。",
        "`PAPER_READY_RESULT_TABLES.md`：论文可用表 1--15。",
    )
    artifact_line = (
        "- `artifacts/interventions/tbbt_intervention_fidelity/`："
        "冻结预测、逐事件比较与图。"
    )
    if artifact_line not in text:
        text = text.replace(
            "- `manuscript/paper_ready_tables/`：对应 LaTeX 表。",
            artifact_line
            + "\n- `manuscript/paper_ready_tables/`：对应 LaTeX 表。",
        )
    latest_path.write_text(text, encoding="utf-8")

    matrix_block = f"""| Reviewer concern | New evidence | Resolution boundary |
| --- | --- | --- |
| Simulator-only circular validation | Five qualified TBBT interventions are predicted by a sealed mechanism map with {100 * summary['direction_accuracy']:.1f}% direction accuracy and {summary['mae_percentage_points']:.2f}-pp MAE versus {summary['zero_effect_mae_percentage_points']:.2f} pp for zero effect | Resolved for retrospective TBBT mechanism prediction; not a prospective blind test |
| No unified intervention-prediction baseline | BDMTF improves event-level MAE over zero effect by {improvement['mean']:.2f} pp [{improvement['ci_low']:.2f}, {improvement['ci_high']:.2f}] | Resolved for the five qualified events; larger event samples remain desirable |
| No uncertainty comparison | 4/5 observed points fall in BDMTF prediction intervals; 5/5 frozen predictions fall in observed 95% CIs | Compatible uncertainty, with The_Donald retained as the uncovered observed point |"""
    _replace_block(
        ROOT / "REVIEW_RESPONSE_MATRIX.md",
        MATRIX_START,
        MATRIX_END,
        matrix_block,
    )
    matrix_path = ROOT / "REVIEW_RESPONSE_MATRIX.md"
    text = matrix_path.read_text(encoding="utf-8")
    text = text.replace(
        "| 模拟器内部循环论证 | 时间切分、held-out 响应预测、真实 Reddit 语义模式及 Lemmy 真实审核效应 | 大部分解决；干预预测保真度比较仍待完成 |",
        "| 模拟器内部循环论证 | 时间切分、真实语义模式、Lemmy 时间外重放及 TBBT 冻结机制预测 | 已在回顾性真实干预预测范围内解决；真人随机验证待伦理审批 |",
    )
    text = text.replace(
        "| 缺少替代基线 | 五模型边际拟合比较完成 | 部分解决；真实干预预测仍需统一比较 |",
        "| 缺少替代基线 | 五模型边际拟合及 TBBT 零效应预测基线均完成 | 已解决于当前实验范围 |",
    )
    text = text.replace(
        "| 没有真实干预 | TBBT 全量描述面板；Lemmy 503 个风险集对的校准合成对照通过全部门槛 | 已解决于 Lemmy 审核干预；TBBT 仍仅作描述性三角验证 |",
        "| 没有真实干预 | Lemmy 503 对真实审核干预；TBBT 5 个合格对照事件及冻结预测比较 | 已解决于观察性真实干预范围；真人 RCT 仍待完成 |",
    )
    text = text.replace(
        "| 只选五个社区 | 五社区 LOCO 已完成 | 未完全解决：`needs_data` |",
        "| 只选五个社区 | 24 个 Reddit 社区、12,085 个完整级联的适配与 LOCO 验证 | 已解决于 Reddit 扩展范围 |",
    )
    text = text.replace(
        "| 跨平台证据不足 | HN 5,000 级联及适配曲线完成，最佳距离 0.422 | 部分解决；大规模 Lemmy/Voat 和内容配对待完成 |",
        "| 跨平台证据不足 | HN 5,000 级联、758 个 HN--Lemmy 同 URL 对、724 个完整树对及 560 名 Reddit--Voat 同作者 | 已解决于线程式平台有限数据适配；不宣称零样本普适性 |",
    )
    text = text.replace(
        "| 缺少真实审核响应预测 | 真实 Lemmy 自然实验、聚合响应适配器和逐事件智能体重放三层结果均已保存 | Lemmy 锁帖/删帖范围内解决；TBBT 和真人 RCT 单列待完成 |",
        "| 缺少真实审核响应预测 | Lemmy 自然实验与逐事件重放、TBBT 5 个合格事件冻结预测均已保存 | 已解决于回顾性观察性干预范围；真人 RCT 单列待完成 |",
    )
    matrix_path.write_text(text, encoding="utf-8")


def _update_latest_json(
    summary: dict[str, object],
    freeze: dict[str, object],
) -> None:
    path = ROOT / "artifacts" / "provenance" / "latest_status.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["tbbt_intervention_prediction_fidelity"] = {
        **summary,
        "prediction_freeze": {
            "freeze_id": freeze["freeze_id"],
            "qualified_interventions": freeze["qualified_interventions"],
            "analyst_blinding": freeze["analyst_blinding"],
        },
    }
    payload["pending_experiments"] = [
        item
        for item in payload.get("pending_experiments", [])
        if item.get("id") != "tbbt_intervention_prediction_fidelity"
    ]
    payload["sources"] = [
        item
        for item in payload.get("sources", [])
        if item.get("path")
        not in {
            "artifacts/interventions/tbbt_intervention_fidelity/prediction_freeze_manifest.json",
            "artifacts/interventions/tbbt_intervention_fidelity/fidelity_summary.json",
            "artifacts/interventions/tbbt_intervention_fidelity/prediction_observed_comparison.csv",
        }
    ]
    for source in (
        FREEZE_PATH,
        SUMMARY_PATH,
        COMPARISON_PATH,
    ):
        payload["sources"].append(
            {
                "path": source.relative_to(ROOT).as_posix(),
                "sha256": __import__("hashlib").sha256(
                    source.read_bytes()
                ).hexdigest(),
            }
        )
    payload["latest_one_sentence"] = (
        "最新结果：TBBT 5 个合格真实干预的冻结 BDMTF 预测达到 5/5 "
        "方向正确、MAE 3.15 个百分点，相对零效应基线改善 13.45 "
        "[6.38, 18.02] 个百分点；4/5 真实点落入预测区间，证据等级为"
        "回顾性协议冻结。"
    )
    build_path = (
        ROOT / "manuscript" / "original_pdf_revision" / "BUILD_MANIFEST.json"
    )
    pdf_path = (
        ROOT / "manuscript" / "original_pdf_revision" / "output" / "paper.pdf"
    )
    if build_path.is_file() and pdf_path.is_file():
        build = json.loads(build_path.read_text(encoding="utf-8"))
        payload["paper"] = {
            "status": build["status"],
            "base_original_pdf_sha256": build[
                "base_original_pdf_sha256"
            ],
            "official_tex": "manuscript/original_pdf_revision/paper.tex",
            "official_pdf": (
                "manuscript/original_pdf_revision/output/paper.pdf"
            ),
            "official_pdf_sha256": hashlib.sha256(
                pdf_path.read_bytes()
            ).hexdigest(),
        }
        payload["sources"] = [
            item
            for item in payload["sources"]
            if item.get("path")
            != "manuscript/original_pdf_revision/BUILD_MANIFEST.json"
        ]
        payload["sources"].append(
            {
                "path": (
                    "manuscript/original_pdf_revision/BUILD_MANIFEST.json"
                ),
                "sha256": hashlib.sha256(
                    build_path.read_bytes()
                ).hexdigest(),
            }
        )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    comparison = pd.read_csv(COMPARISON_PATH)
    _write_latex(summary, comparison)
    _update_markdown(summary, comparison, freeze)
    _update_latest_json(summary, freeze)
    print(
        json.dumps(
            {
                "status": "updated",
                "paper_table": "table 15",
                "direction_accuracy": summary["direction_accuracy"],
                "mae_percentage_points": summary["mae_percentage_points"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

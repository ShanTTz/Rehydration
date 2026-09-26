from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "artifacts" / "interventions" / "fidelity"
SUMMARY_PATH = RESULT_DIR / "intervention_fidelity.json"
METRICS_PATH = RESULT_DIR / "fidelity_summary.csv"
IMPROVEMENTS_PATH = RESULT_DIR / "paired_improvements.csv"
MARKDOWN_PATH = ROOT / "PAPER_READY_RESULT_TABLES.md"
LATEX_PATH = (
    ROOT
    / "manuscript"
    / "paper_ready_tables"
    / "table_lemmy_intervention_fidelity.tex"
)
MANIFEST_PATH = LATEX_PATH.with_suffix(".manifest.json")
START = "<!-- LEMMY_INTERVENTION_FIDELITY_START -->"
END = "<!-- LEMMY_INTERVENTION_FIDELITY_END -->"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _name(model: str) -> str:
    return {
        "zero_effect": "Zero-effect baseline",
        "frozen_mechanism": "Frozen mechanism",
        "type_median": "Intervention-type median",
        "additive_ridge": "Additive adapter",
        "bdmtf_interaction": "BDMTF interaction adapter",
        "gradient_boosting": "Gradient boosting",
    }.get(model, model)


def main() -> None:
    result = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    metrics = pd.read_csv(METRICS_PATH)
    primary = metrics[metrics["scope"].eq("all_primary")].set_index("model")
    order = [
        "zero_effect",
        "frozen_mechanism",
        "type_median",
        "additive_ridge",
        "bdmtf_interaction",
        "gradient_boosting",
    ]
    rows = []
    for model in order:
        row = primary.loc[model]
        rows.append(
            [
                _name(model),
                (
                    f"{row['mae']:.3f} "
                    f"[{row['mae_ci_low']:.3f}, {row['mae_ci_high']:.3f}]"
                ),
                f"{100 * row['direction_accuracy']:.1f}%",
                f"{100 * row['interval_coverage']:.1f}%",
            ]
        )

    framework = {
        "supported": "支持",
        "competitive_but_not_significantly_better": "有竞争力但未显著优于加性适配器",
        "not_supported_by_primary_mae": "主 MAE 未支持",
    }.get(result["framework_support"], result["framework_support"])
    markdown_rows = "\n".join(
        f"| {model} | {mae} | {direction} | {coverage} |"
        for model, mae, direction, coverage in rows
    )
    block = f"""{START}

## 表 10：Lemmy 干预预测保真度

**作用：** 检验 BDMTF 能否用较早的真实审核干预校准后，预测时间上更晚且训练阶段未见的锁帖与删帖响应；测试特征仅使用干预前 7 天，测试干预后结果从未参与拟合或调参。

| 模型 | 主结果 MAE [95% CI] | 方向准确率 | 95% 区间覆盖率 |
| --- | ---: | ---: | ---: |
{markdown_rows}

完整数据包含 **{result['n_pairs']} 个无重复风险集对**，按干预类型分别进行 60/20/20 时间划分，最终测试 **{result['test_pairs']} 个较晚干预**、三个预注册主结果。BDMTF 交互适配器的 MAE 为 **{result['bdmtf_primary_mae']:.3f}**，方向准确率为 **{100 * result['bdmtf_direction_accuracy']:.1f}%**，区间覆盖率为 **{100 * result['bdmtf_interval_coverage']:.1f}%**；相对加性适配器的 MAE 改进为 **{result['bdmtf_vs_additive_mae_improvement']:+.3f}**，判定为“**{framework}**”。

**论文用途：** 该结果连接了模拟机制与真实 Lemmy 审核响应，但属于回顾性时间外验证；不能写成前瞻随机实验，也不能推广到未验证平台。

_自动更新时间：{date.today().isoformat()}_

{END}"""
    current = MARKDOWN_PATH.read_text(encoding="utf-8")
    if START in current and END in current:
        prefix, remainder = current.split(START, 1)
        _, suffix = remainder.split(END, 1)
        updated = prefix.rstrip() + "\n\n" + block + suffix
    else:
        updated = current.rstrip() + "\n\n" + block + "\n"
    MARKDOWN_PATH.write_text(updated, encoding="utf-8")

    LATEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    escape_percent = lambda value: value.replace("%", r"\%")
    latex_rows = "\n".join(
        f"{model} & {mae} & {escape_percent(direction)} & "
        f"{escape_percent(coverage)} \\\\"
        for model, mae, direction, coverage in rows
    )
    latex = rf"""\begin{{table}}[t]
\centering
\small
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lrrr}}
\toprule
Model & MAE [95\% CI] & Direction accuracy & 95\% coverage \\
\midrule
{latex_rows}
\bottomrule
\end{{tabular}}
}}
\caption{{Chronological Lemmy intervention-response fidelity on {result['test_pairs']} held-out interventions. Test features use pre-intervention days only.}}
\label{{tab:lemmy-intervention-fidelity}}
\end{{table}}
"""
    LATEX_PATH.write_text(latex, encoding="utf-8")
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "status": "complete",
                "sources": {
                    str(SUMMARY_PATH.relative_to(ROOT)): _sha256(SUMMARY_PATH),
                    str(METRICS_PATH.relative_to(ROOT)): _sha256(METRICS_PATH),
                    str(IMPROVEMENTS_PATH.relative_to(ROOT)): _sha256(
                        IMPROVEMENTS_PATH
                    ),
                },
                "outputs": {
                    str(MARKDOWN_PATH.relative_to(ROOT)): _sha256(MARKDOWN_PATH),
                    str(LATEX_PATH.relative_to(ROOT)): _sha256(LATEX_PATH),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

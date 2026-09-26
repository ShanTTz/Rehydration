from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = (
    ROOT
    / "artifacts"
    / "external_validation"
    / "reddit_expansion"
    / "validation_summary.json"
)
MARKDOWN_PATH = ROOT / "PAPER_READY_RESULT_TABLES.md"
LATEX_PATH = (
    ROOT
    / "manuscript"
    / "paper_ready_tables"
    / "table_reddit_expansion.tex"
)
MANIFEST_PATH = LATEX_PATH.with_suffix(".manifest.json")
START = "<!-- REDDIT_EXPANSION_START -->"
END = "<!-- REDDIT_EXPANSION_END -->"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ranking(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["model"]): row
        for row in summary["ranking"]
    }


def _display(model: str) -> str:
    return {
        "empirical_bootstrap": "Empirical resampling",
        "branching_process": "Branching process",
        "hawkes": "Hawkes",
        "legacy_heuristic": "Theory-specified BDMTF",
        "learned_bdmtf": "Learned BDMTF",
    }.get(model, model)


def _decision(summary: dict[str, Any]) -> str:
    adapted = summary["protocols"]["target_adapted"]
    best = adapted["best_model_by_median_normalized_wasserstein"]
    learned = adapted["learned_bdmtf_median_normalized_wasserstein"]
    ranking = _ranking(adapted)
    best_value = float(ranking[best]["median"])
    if best == "learned_bdmtf":
        return "Learned BDMTF ranks first by the primary median distance."
    gap = float(learned) - best_value
    return (
        "Learned BDMTF remains a comparator but is not first on the primary "
        f"median distance (gap {gap:.3f})."
    )


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    adapted = summary["protocols"]["target_adapted"]
    loco = summary["protocols"]["leave_one_community_out"]
    adapted_ranking = _ranking(adapted)
    loco_ranking = _ranking(loco)
    model_order = [
        "empirical_bootstrap",
        "branching_process",
        "hawkes",
        "legacy_heuristic",
        "learned_bdmtf",
    ]
    rows = []
    for model in model_order:
        rows.append(
            [
                _display(model),
                f"{float(adapted_ranking[model]['median']):.3f}",
                f"{float(loco_ranking[model]['median']):.3f}",
            ]
        )

    markdown_rows = "\n".join(
        f"| {name} | {adapted_value} | {loco_value} |"
        for name, adapted_value, loco_value in rows
    )
    block = f"""{START}

## 表 9：20+ Reddit 社区扩展验证

**作用：** 回应原实验仅覆盖五个社区的选择偏差质疑。社区清单在查看模型结果前按四类用途冻结，并在每个“社区 × viral 标签”内按时间顺序划分训练、验证和测试集。

| 模型 | 目标社区训练段适配 | 留一社区零样本 |
| --- | ---: | ---: |
{markdown_rows}

共覆盖 **{summary['n_communities']:,} 个社区、{summary['n_cascades']:,} 个完整级联**。表中为测试集 normalized Wasserstein 中位数，越低越好；每个模型使用 3 个随机种子。{_decision(summary)}

**论文用途：** 该实验把 Reddit 内部证据从原五个社区扩展到预先分层选择的社区类型，并同时区分“有限目标社区数据适配”和“完全未见社区迁移”。它不替代跨平台验证或真实因果干预。

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
    latex_rows = "\n".join(
        f"{name} & {adapted_value} & {loco_value} \\\\"
        for name, adapted_value, loco_value in rows
    )
    latex = rf"""\begin{{table}}[t]
\centering
\small
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{lrr}}
\toprule
Model & Target-adapted & Reddit LOCO \\
\midrule
{latex_rows}
\bottomrule
\end{{tabular}}
}}
\caption{{Expanded Reddit validation over {summary['n_communities']:,} communities and {summary['n_cascades']:,} structurally complete cascades. Entries are median normalized Wasserstein distances on chronological test splits; lower is better.}}
\label{{tab:reddit-expansion}}
\end{{table}}
"""
    LATEX_PATH.write_text(latex, encoding="utf-8")
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "status": "complete",
                "source": str(SUMMARY_PATH.relative_to(ROOT)),
                "source_sha256": _sha256(SUMMARY_PATH),
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

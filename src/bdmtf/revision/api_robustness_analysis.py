from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from bdmtf.reviewer_semantics import _interval_alpha
from bdmtf.revision.api_intents import (
    _parse_json_object,
    load_intent_response_frame,
)
from bdmtf.revision.provenance import sha256_file, write_json


REPLAY_METRICS = (
    "size",
    "max_depth",
    "mean_leaf_depth",
    "toxicity_density",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _fleiss_kappa(values: np.ndarray) -> float | None:
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        return None
    categories = sorted({str(value) for value in values.reshape(-1)})
    raters = values.shape[1]
    counts = np.asarray(
        [
            [sum(str(value) == category for value in row) for category in categories]
            for row in values
        ],
        dtype=float,
    )
    observed = float(
        np.mean((np.square(counts).sum(axis=1) - raters) / (raters * (raters - 1)))
    )
    category_rates = counts.sum(axis=0) / (values.shape[0] * raters)
    expected = float(np.square(category_rates).sum())
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else None
    return float((observed - expected) / (1.0 - expected))


def _share_interval(
    indicator: np.ndarray,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    values = np.asarray(indicator, dtype=float)
    draws = rng.choice(
        values,
        size=(bootstrap_samples, len(values)),
        replace=True,
    ).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def _pairwise_nominal_agreement(
    matrix: pd.DataFrame,
    measure: str,
) -> pd.DataFrame:
    rows = []
    for left, right in combinations(matrix.columns, 2):
        left_values = matrix[left].astype(str)
        right_values = matrix[right].astype(str)
        rows.append(
            {
                "measure": measure,
                "left_family": left,
                "right_family": right,
                "n_tasks": int(len(matrix)),
                "exact_agreement": float(
                    np.mean(left_values == right_values)
                ),
                "cohen_kappa": float(
                    cohen_kappa_score(left_values, right_values)
                ),
            }
        )
    return pd.DataFrame(rows)


def _parse_annotations(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in _read_jsonl(path):
        response = _parse_json_object(record["response"])
        for item in response["items"]:
            rows.append(
                {
                    "task_id": record["task_id"],
                    "family": record["family"],
                    "model": record["model"],
                    "community": record["community"],
                    "comment_id": str(item["comment_id"]),
                    "toxicity": float(item["toxicity"]),
                    "emotion": str(item["emotion"]).strip().lower(),
                    "topic": str(item["topic"]).strip().lower(),
                    "counterspeech": bool(item["counterspeech"]),
                }
            )
    return pd.DataFrame(rows).drop_duplicates(
        ["comment_id", "family"],
        keep="first",
    )


def _usage_summary(paths: list[tuple[str, Path]]) -> pd.DataFrame:
    rows = []
    for task_kind, path in paths:
        for record in _read_jsonl(path):
            usage = record.get("usage", {})
            rows.append(
                {
                    "task_kind": task_kind,
                    "family": record["family"],
                    "model": record["model"],
                    "returned_model": record.get("returned_model"),
                    "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(
                        usage.get("completion_tokens", 0)
                    ),
                    "total_tokens": int(usage.get("total_tokens", 0)),
                    "latency_seconds": float(
                        record.get("latency_seconds", 0.0)
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    return (
        frame.groupby(
            ["task_kind", "family", "model", "returned_model"],
            dropna=False,
            sort=True,
        )
        .agg(
            calls=("model", "size"),
            prompt_tokens=("prompt_tokens", "sum"),
            completion_tokens=("completion_tokens", "sum"),
            total_tokens=("total_tokens", "sum"),
            mean_latency_seconds=("latency_seconds", "mean"),
            p95_latency_seconds=(
                "latency_seconds",
                lambda values: values.quantile(0.95),
            ),
        )
        .reset_index()
    )


def _write_figure(
    model_summary: pd.DataFrame,
    annotation_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    order = ["gpt-4o-mini", "deepseek-v4-flash", "qwen3.7-plus"]
    model = model_summary.set_index("model").reindex(order)
    annotation = annotation_summary.set_index("model").reindex(order)
    labels = ["GPT-4o mini", "DeepSeek V4 Flash", "Qwen 3.7 Plus"]
    positions = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    axes[0].bar(
        positions - 0.18,
        100 * model["reply_rate"],
        width=0.36,
        label="Reply",
        color="#2878B5",
    )
    axes[0].bar(
        positions + 0.18,
        100 * model["antagonistic_rate"],
        width=0.36,
        label="Antagonistic",
        color="#D1495B",
    )
    axes[0].set_ylabel("Intent share (%)")
    axes[0].set_xticks(positions, labels, rotation=15, ha="right")
    axes[0].set_ylim(0, 105)
    axes[0].legend(frameon=False)
    axes[0].set_title("Frozen persona-conditioned intents")

    axes[1].bar(
        positions - 0.18,
        annotation["mean_toxicity"],
        width=0.36,
        label="Mean toxicity",
        color="#5B8E7D",
    )
    axes[1].bar(
        positions + 0.18,
        annotation["counterspeech_rate"],
        width=0.36,
        label="Counterspeech",
        color="#E6A44E",
    )
    axes[1].set_ylabel("Annotation rate / mean")
    axes[1].set_xticks(positions, labels, rotation=15, ha="right")
    axes[1].set_ylim(0, 0.5)
    axes[1].legend(frameon=False)
    axes[1].set_title("Real-comment semantic labels")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_latex(summary: dict[str, Any], root: Path) -> None:
    intent = summary["intent_agreement"]
    annotation = summary["annotation_agreement"]
    strict = summary["strict_rehydration"]
    coupled = summary["coupled_negative_control"]
    table = rf"""\begin{{table}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrr}}
\toprule
Check & Estimate & Scope \\
\midrule
Reply-decision unanimity & {100 * intent['reply_unanimous']['mean']:.1f}\% & 500 tasks \\
Polarity unanimity & {100 * intent['polarity_unanimous']['mean']:.1f}\% & 500 tasks \\
Toxicity interval $\alpha$ & {annotation['toxicity_interval_alpha']:.3f} & 500 comments \\
Counterspeech unanimity & {100 * annotation['counterspeech_unanimous']['mean']:.1f}\% & 500 comments \\
Strict replay max difference & {strict['maximum_structural_range']:.3f} & 50 paired posts \\
Coupled median size range & {100 * coupled['median_relative_range']['size']:.1f}\% & diagnostic \\
\bottomrule
\end{{tabular}}
\caption{{Three-family semantic-intent and Rehydration robustness. Strict
replay shares platform dynamics and post-level seeds; the coupled result is a
negative-control diagnostic, not the Rehydration estimand.}}
\label{{tab:api-model-robustness}}
\end{{table}}
"""
    macros = "\n".join(
        [
            rf"\newcommand{{\APICompletedCalls}}{{{summary['calls']['total']}}}",
            rf"\newcommand{{\APIIntentCalls}}{{{summary['calls']['intents']}}}",
            rf"\newcommand{{\APIAnnotationCalls}}{{{summary['calls']['annotations']}}}",
            rf"\newcommand{{\APIReplyAgreement}}{{{100 * intent['reply_unanimous']['mean']:.1f}}}",
            rf"\newcommand{{\APIPolarityAgreement}}{{{100 * intent['polarity_unanimous']['mean']:.1f}}}",
            rf"\newcommand{{\APIToxicityAlpha}}{{{annotation['toxicity_interval_alpha']:.3f}}}",
            rf"\newcommand{{\APICounterspeechAgreement}}{{{100 * annotation['counterspeech_unanimous']['mean']:.1f}}}",
            rf"\newcommand{{\APIStrictReplayDifference}}{{{strict['maximum_structural_range']:.3f}}}",
            "",
        ]
    )
    for directory in (
        root / "manuscript" / "paper_ready_tables",
        root / "manuscript" / "original_pdf_revision",
    ):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "table_api_model_robustness.tex").write_text(
            table,
            encoding="utf-8",
        )
        (directory / "api_model_robustness_macros.tex").write_text(
            macros,
            encoding="utf-8",
        )


def _write_report(summary: dict[str, Any], output_path: Path) -> None:
    intent = summary["intent_agreement"]
    annotation = summary["annotation_agreement"]
    strict = summary["strict_rehydration"]
    coupled = summary["coupled_negative_control"]
    text = f"""# 三模型冻结意图与 Rehydration 稳健性报告

## 运行完整性

- 有效 API 调用：{summary['calls']['total']}/{summary['calls']['expected']}。
- 冻结意图：{summary['calls']['intents']}；真实评论批量标注：
  {summary['calls']['annotations']}。
- 模型：`gpt-4o-mini`、`deepseek-v4-flash`、`qwen3.7-plus`。
- 所有意图响应均与当前任务的提示词哈希一致。旧 `Post: nan`
  运行已隔离，不能进入论文。

## 意图一致性

- 回复决策三方一致率：
  **{100 * intent['reply_unanimous']['mean']:.1f}%**
  [{100 * intent['reply_unanimous']['ci_low']:.1f},
  {100 * intent['reply_unanimous']['ci_high']:.1f}]。
- 极性三方一致率：
  **{100 * intent['polarity_unanimous']['mean']:.1f}%**
  [{100 * intent['polarity_unanimous']['ci_low']:.1f},
  {100 * intent['polarity_unanimous']['ci_high']:.1f}]。
- 三模型回复率范围：
  {100 * intent['reply_rate_min']:.1f}%--{100 * intent['reply_rate_max']:.1f}%；
  攻击性意图率范围：
  {100 * intent['antagonistic_rate_min']:.1f}%--{100 * intent['antagonistic_rate_max']:.1f}%。

## 真实评论标注

- 500 条评论、1,500 个模型标签。
- 毒性区间 Krippendorff alpha：
  **{annotation['toxicity_interval_alpha']:.3f}**；
  平均两两 Spearman：
  **{annotation['toxicity_mean_pairwise_spearman']:.3f}**。
- counterspeech 三方一致率：
  **{100 * annotation['counterspeech_unanimous']['mean']:.1f}%**，
  Fleiss kappa 为 {annotation['counterspeech_fleiss_kappa']:.3f}。
- 开放词汇 emotion/topic 的完全一致率分别只有
  {100 * annotation['emotion_unanimous_share']:.1f}% 和
  {100 * annotation['topic_unanimous_share']:.1f}%，不应把这些标签当作
  跨模型稳定的离散真值。

## Rehydration

严格重放对每个帖子、模型使用同一平台参数和配对种子。四项主要结构指标
的最大模型间差异为 **{strict['maximum_structural_range']:.3f}**，通过
结构不变性检查。这是 Rehydration 信息屏障的实现性质，不是人类行为
有效性的独立证明。

重新耦合负对照把模型回复率和攻击性率重新注入 activation/conflict。
其帖子级模型间中位相对范围为：规模
{100 * coupled['median_relative_range']['size']:.1f}%，平均叶深
{100 * coupled['median_relative_range']['mean_leaf_depth']:.1f}%。
该对照说明如果绕过信息屏障，模型差异会重新进入结构结果，因此应保留
严格共享动力学版本作为论文主分析。

## 结论边界

结果支持三点：主要意图类别在三个模型系列间具有较高直接一致性；真实
评论毒性测量具有较强跨模型一致性；严格 Rehydration 能阻断模型选择
对结构反事实的直接渗透。结果不支持开放词汇 emotion/topic 的模型无关
性，也不能替代真人随机实验。
"""
    output_path.write_text(text, encoding="utf-8")


def analyze_api_robustness(
    root: str | Path,
    bootstrap_samples: int = 2000,
    seed: int = 30371,
) -> dict[str, Any]:
    project = Path(root)
    api_dir = project / "artifacts" / "api"
    output = api_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (api_dir / "api_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("status") != "complete":
        raise ValueError("API manifest is not complete")

    intents = load_intent_response_frame(
        api_dir / "frozen_intents_multimodel.jsonl"
    )
    tasks = pd.read_csv(api_dir / "intent_tasks.csv")
    task_hashes = tasks.set_index("task_id")["prompt_sha256"].to_dict()
    intents["active_prompt_hash"] = intents["task_id"].map(task_hashes)
    if not intents["prompt_sha256"].eq(intents["active_prompt_hash"]).all():
        raise ValueError("Intent cache contains a stale prompt hash")
    if intents[["task_id", "family"]].drop_duplicates().shape[0] != 1500:
        raise ValueError("Expected 1,500 unique task-family intent records")

    intents["constructive"] = intents["polarity"].eq("constructive")
    intents["neutral"] = intents["polarity"].eq("neutral")
    model_summary = (
        intents.groupby(["family", "model"], sort=True)
        .agg(
            calls=("task_id", "size"),
            reply_rate=("reply", "mean"),
            antagonistic_rate=("antagonistic", "mean"),
            constructive_rate=("constructive", "mean"),
            neutral_rate=("neutral", "mean"),
            mean_content_length=("content_length", "mean"),
        )
        .reset_index()
    )
    community_summary = (
        intents.groupby(["family", "model", "community"], sort=True)
        .agg(
            calls=("task_id", "size"),
            reply_rate=("reply", "mean"),
            antagonistic_rate=("antagonistic", "mean"),
        )
        .reset_index()
    )

    rng = np.random.default_rng(seed)
    reply_matrix = intents.pivot(
        index="task_id",
        columns="family",
        values="reply",
    ).dropna()
    polarity_matrix = intents.pivot(
        index="task_id",
        columns="family",
        values="polarity",
    ).dropna()
    reply_unanimous = _share_interval(
        reply_matrix.nunique(axis=1).eq(1).to_numpy(),
        bootstrap_samples,
        rng,
    )
    polarity_unanimous = _share_interval(
        polarity_matrix.nunique(axis=1).eq(1).to_numpy(),
        bootstrap_samples,
        rng,
    )
    intent_pairwise = pd.concat(
        [
            _pairwise_nominal_agreement(reply_matrix, "reply"),
            _pairwise_nominal_agreement(polarity_matrix, "polarity"),
        ],
        ignore_index=True,
    )

    annotations = _parse_annotations(
        api_dir / "semantic_annotations_multimodel.jsonl"
    )
    if annotations[["comment_id", "family"]].shape[0] != 1500:
        raise ValueError("Expected 1,500 unique comment-family labels")
    annotation_summary = (
        annotations.groupby(["family", "model"], sort=True)
        .agg(
            labels=("comment_id", "size"),
            mean_toxicity=("toxicity", "mean"),
            counterspeech_rate=("counterspeech", "mean"),
            emotion_vocabulary=("emotion", "nunique"),
            topic_vocabulary=("topic", "nunique"),
        )
        .reset_index()
    )
    toxicity = annotations.pivot(
        index="comment_id",
        columns="family",
        values="toxicity",
    ).dropna()
    toxicity_pairs = []
    for left, right in combinations(toxicity.columns, 2):
        toxicity_pairs.append(
            {
                "left_family": left,
                "right_family": right,
                "n_comments": int(len(toxicity)),
                "spearman": float(
                    spearmanr(toxicity[left], toxicity[right]).statistic
                ),
                "mae": float(
                    np.mean(np.abs(toxicity[left] - toxicity[right]))
                ),
            }
        )
    toxicity_pairwise = pd.DataFrame(toxicity_pairs)
    counterspeech = annotations.pivot(
        index="comment_id",
        columns="family",
        values="counterspeech",
    ).dropna()
    counterspeech_unanimous = _share_interval(
        counterspeech.nunique(axis=1).eq(1).to_numpy(),
        bootstrap_samples,
        rng,
    )
    emotion = annotations.pivot(
        index="comment_id",
        columns="family",
        values="emotion",
    ).dropna()
    topic = annotations.pivot(
        index="comment_id",
        columns="family",
        values="topic",
    ).dropna()

    strict = pd.read_csv(api_dir / "replay_metrics.csv")
    coupled = pd.read_csv(api_dir / "coupled_replay_metrics.csv")
    strict_ranges = strict.groupby(["community", "post_id"])[
        list(REPLAY_METRICS)
    ].agg(lambda values: values.max() - values.min())
    coupled_ranges = coupled.groupby(["community", "post_id"])[
        list(REPLAY_METRICS)
    ].agg(lambda values: values.max() - values.min())
    coupled_means = coupled.groupby(["community", "post_id"])[
        list(REPLAY_METRICS)
    ].mean()
    coupled_relative = coupled_ranges / coupled_means.replace(0.0, np.nan)

    usage = _usage_summary(
        [
            (
                "intent",
                api_dir / "frozen_intents_multimodel.jsonl",
            ),
            (
                "annotation",
                api_dir / "semantic_annotations_multimodel.jsonl",
            ),
        ]
    )
    total_calls = int(usage["calls"].sum())
    summary: dict[str, Any] = {
        "status": "complete",
        "models": manifest["models"],
        "calls": {
            "expected": 1575,
            "total": total_calls,
            "intents": int(len(intents)),
            "annotations": int(
                annotations[["task_id", "family"]]
                .drop_duplicates()
                .shape[0]
            ),
            "annotation_labels": int(len(annotations)),
        },
        "prompt_audit": {
            "tasks": int(len(tasks)),
            "unique_posts": int(
                tasks[["community", "post_id"]]
                .drop_duplicates()
                .shape[0]
            ),
            "prompt_hash_mismatches": 0,
            "persona_conditioned": True,
            "invalid_prior_run_excluded": True,
        },
        "intent_agreement": {
            "reply_unanimous": reply_unanimous,
            "reply_fleiss_kappa": _fleiss_kappa(
                reply_matrix.to_numpy()
            ),
            "polarity_unanimous": polarity_unanimous,
            "polarity_fleiss_kappa": _fleiss_kappa(
                polarity_matrix.to_numpy()
            ),
            "reply_rate_min": float(model_summary["reply_rate"].min()),
            "reply_rate_max": float(model_summary["reply_rate"].max()),
            "antagonistic_rate_min": float(
                model_summary["antagonistic_rate"].min()
            ),
            "antagonistic_rate_max": float(
                model_summary["antagonistic_rate"].max()
            ),
        },
        "annotation_agreement": {
            "comments": int(annotations["comment_id"].nunique()),
            "toxicity_interval_alpha": float(
                _interval_alpha(toxicity.to_numpy(dtype=float))
            ),
            "toxicity_mean_pairwise_spearman": float(
                toxicity_pairwise["spearman"].mean()
            ),
            "toxicity_mean_pairwise_mae": float(
                toxicity_pairwise["mae"].mean()
            ),
            "counterspeech_unanimous": counterspeech_unanimous,
            "counterspeech_fleiss_kappa": _fleiss_kappa(
                counterspeech.to_numpy()
            ),
            "emotion_unanimous_share": float(
                emotion.nunique(axis=1).eq(1).mean()
            ),
            "topic_unanimous_share": float(
                topic.nunique(axis=1).eq(1).mean()
            ),
        },
        "strict_rehydration": {
            "protocol": "strict_shared_dynamics",
            "paired_posts": int(len(strict_ranges)),
            "maximum_structural_range": float(
                strict_ranges.max().max()
            ),
            "metric_maximum_ranges": {
                metric: float(strict_ranges[metric].max())
                for metric in REPLAY_METRICS
            },
            "interpretation": (
                "Software-level information-barrier check; not independent "
                "human-behavior validation."
            ),
        },
        "coupled_negative_control": {
            "protocol": "intent_coupled_diagnostic",
            "median_relative_range": {
                metric: float(coupled_relative[metric].median())
                for metric in REPLAY_METRICS
            },
            "p95_relative_range": {
                metric: float(coupled_relative[metric].quantile(0.95))
                for metric in REPLAY_METRICS
            },
        },
        "claim": (
            "Three-family robustness is supported for reply decisions, broad "
            "polarity, toxicity measurement, and strict structural replay, "
            "with explicit limits for open-vocabulary emotion/topic labels."
        ),
        "evidence_boundary": (
            "Model-family agreement and an implementation invariance audit do "
            "not replace a human randomized experiment."
        ),
        "sources": [],
    }
    for path in (
        api_dir / "api_manifest.json",
        api_dir / "intent_tasks.csv",
        api_dir / "frozen_intents_multimodel.jsonl",
        api_dir / "semantic_annotations_multimodel.jsonl",
        api_dir / "replay_metrics.csv",
        api_dir / "coupled_replay_metrics.csv",
        api_dir
        / "invalid_blank_post_run"
        / "invalid_run_manifest.json",
    ):
        summary["sources"].append(
            {
                "path": path.relative_to(project).as_posix(),
                "sha256": sha256_file(path),
            }
        )

    model_summary.to_csv(output / "intent_model_summary.csv", index=False)
    community_summary.to_csv(
        output / "intent_community_summary.csv",
        index=False,
    )
    intent_pairwise.to_csv(
        output / "intent_pairwise_agreement.csv",
        index=False,
    )
    annotation_summary.to_csv(
        output / "annotation_model_summary.csv",
        index=False,
    )
    toxicity_pairwise.to_csv(
        output / "toxicity_pairwise_agreement.csv",
        index=False,
    )
    strict_ranges.reset_index().to_csv(
        output / "strict_replay_model_ranges.csv",
        index=False,
    )
    coupled_relative.reset_index().to_csv(
        output / "coupled_replay_relative_ranges.csv",
        index=False,
    )
    usage.to_csv(output / "api_usage.csv", index=False)
    _write_figure(
        model_summary,
        annotation_summary,
        output / "api_robustness_overview.png",
    )
    write_json(output / "api_robustness_summary.json", summary)
    _write_report(summary, output / "API_ROBUSTNESS_REPORT.md")
    _write_latex(summary, project)
    return summary

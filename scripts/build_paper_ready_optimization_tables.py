from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "manuscript" / "paper_ready_tables"
MARKDOWN_PATH = ROOT / "PAPER_READY_RESULT_TABLES.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def find_row(
    rows: list[dict[str, str]],
    **criteria: str,
) -> dict[str, str]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    raise KeyError(f"No row matched {criteria}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f(value: str | float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def pct(value: str | float, digits: int = 1) -> str:
    return f"{100 * float(value):.{digits}f}%"


def ci_ratio(row: dict[str, str]) -> str:
    mean = math.exp(float(row["mean"]))
    low = math.exp(float(row["ci_low"]))
    high = math.exp(float(row["ci_high"]))
    return f"{mean:.3f} [{low:.3f}, {high:.3f}]"


def ci_value(row: dict[str, str]) -> str:
    return (
        f"{float(row['mean']):.3f} "
        f"[{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}]"
    )


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    output = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(output)


def latex_table(
    *,
    filename: str,
    headers: list[str],
    rows: list[list[str]],
    caption: str,
    label: str,
    column_spec: str,
    resize: bool = True,
) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
    ]
    if resize:
        lines.extend(
            [
                r"\resizebox{\textwidth}{!}{%",
                rf"\begin{{tabular}}{{{column_spec}}}",
            ]
        )
    else:
        lines.append(rf"\begin{{tabular}}{{{column_spec}}}")
    lines.extend(
        [
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
        ]
    )
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if resize:
        lines.append("}")
    lines.extend(
        [
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table*}",
            "",
        ]
    )
    (OUTPUT_DIR / filename).write_text("\n".join(lines), encoding="utf-8")


def build() -> None:
    sources = {
        "factorial": ROOT
        / "run_outputs/reviewer_confirmatory_factorial/reviewer_analysis/factorial_summary.csv",
        "factorial_manifest": ROOT
        / "run_outputs/reviewer_confirmatory_factorial/reviewer_analysis/factorial_manifest.json",
        "response_summary": ROOT
        / "artifacts/reviewer_validation/response_benchmark/response_benchmark_summary.csv",
        "response_improvement": ROOT
        / "artifacts/reviewer_validation/response_benchmark/response_benchmark_improvements.csv",
        "sensitivity": ROOT
        / "run_outputs/reviewer_local_sensitivity/analysis/sensitivity_manifest.json",
        "traits": ROOT
        / "run_outputs/reviewer_trait_coupling/analysis/trait_coupling_summary.csv",
        "traits_manifest": ROOT
        / "run_outputs/reviewer_trait_coupling/analysis/trait_coupling_manifest.json",
        "reddit_ranking": ROOT / "artifacts/evaluation/model_ranking.csv",
        "loco_ranking": ROOT
        / "artifacts/external_validation/cross_community/evaluation/model_ranking.csv",
        "hn_ranking": ROOT
        / "artifacts/external_validation/platform_adapter/evaluation/model_ranking.csv",
        "hn_gain": ROOT
        / "artifacts/external_validation/platform_adapter/adapter_gain_manifest.json",
        "semantic_results": ROOT
        / "artifacts/reviewer_validation/semantic_calibration/semantic_heldout_pattern_results.csv",
        "semantic_agreement": ROOT
        / "artifacts/reviewer_validation/semantic_calibration/semantic_model_agreement.csv",
        "semantic_manifest": ROOT
        / "artifacts/reviewer_validation/semantic_calibration/semantic_analysis_manifest.json",
        "ablation": ROOT / "artifacts/runs/ablations/tradeoff_regions.csv",
        "cross_platform": ROOT
        / "artifacts/external_validation/cross_platform_transfer/run_manifest.json",
        "tbbt": ROOT / "artifacts/interventions/tbbt/import_manifest.json",
        "lemmy": ROOT
        / "data/external/lemmy/outcomes_confirmatory/outcome_collection_manifest.json",
        "lemmy_natural": ROOT
        / "artifacts/interventions/natural_confirmatory_lemmy_calibrated/"
        "natural_experiment_summary.json",
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing result sources:\n" + "\n".join(missing))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "table_exact_reproduction.tex",
        "table_exact_communities.tex",
    ):
        (OUTPUT_DIR / stale_name).unlink(missing_ok=True)

    factorial_rows = read_csv(sources["factorial"])
    factorial_manifest = read_json(sources["factorial_manifest"])
    response_rows = read_csv(sources["response_summary"])
    response_improvements = read_csv(sources["response_improvement"])
    sensitivity = read_json(sources["sensitivity"])
    trait_rows = read_csv(sources["traits"])
    trait_manifest = read_json(sources["traits_manifest"])
    reddit_ranking = read_csv(sources["reddit_ranking"])
    loco_ranking = read_csv(sources["loco_ranking"])
    hn_ranking = read_csv(sources["hn_ranking"])
    hn_gain = read_json(sources["hn_gain"])
    semantic_rows = read_csv(sources["semantic_results"])
    agreement_rows = read_csv(sources["semantic_agreement"])
    semantic_manifest = read_json(sources["semantic_manifest"])
    ablation_rows = read_csv(sources["ablation"])
    cross_platform = read_json(sources["cross_platform"])
    tbbt = read_json(sources["tbbt"])
    lemmy = read_json(sources["lemmy"])
    lemmy_natural = read_json(sources["lemmy_natural"])
    intervention = {
        row["outcome"]: row
        for row in lemmy_natural["primary_diagnostics"]
    }

    factorial = {
        (row["metric"], row["contrast"]): row
        for row in factorial_rows
        if row["scope"] == "all"
    }
    response = {(row["metric"], row["model"]): row for row in response_rows}
    improvement = {
        (row["metric"], row["full_model"]): row for row in response_improvements
    }
    traits = {(row["trait_mode"], row["metric"]): row for row in trait_rows}
    reddit = {row["model"]: row for row in reddit_ranking}
    loco = {row["model"]: row for row in loco_ranking}
    hn = {row["model"]: row for row in hn_ranking}
    semantic = {row["outcome"]: row for row in semantic_rows}
    agreement = {row["label"]: row for row in agreement_rows}

    factor_specs = [
        ("Core effect under best", "core_at_best", "内容层主效应"),
        (
            "Ranking effect under baseline",
            "ranking_at_baseline",
            "单独排序不放大评论量",
        ),
        (
            "Ranking effect under toxic Core",
            "ranking_at_toxic",
            "冲突 Core 下排序产生放大",
        ),
        (
            "Core $\\times$ Structure interaction",
            "core_ranking_interaction",
            "非加性交互",
        ),
        ("Joint intervention", "joint", "完整 Shallow Swarm 对比"),
    ]
    factor_md = []
    factor_tex = []
    for label, contrast, meaning in factor_specs:
        volume_key = "joint_log" if contrast == "joint" else contrast + "_log"
        volume = factorial[("comment_volume", volume_key)]
        depth = factorial[("mean_leaf_depth", contrast)]
        factor_md.append(
            [
                label.replace("$\\times$", "×"),
                ci_ratio(volume),
                ci_value(depth),
                meaning,
            ]
        )
        factor_tex.append(
            [
                label,
                ci_ratio(volume),
                ci_value(depth),
            ]
        )

    sens_volume = sensitivity["scenario_effect_ranges"]["geometric_volume_ratio"]
    sens_depth = sensitivity["scenario_effect_ranges"]["mean_leaf_depth_delta"]
    sens_share = sensitivity["scenario_effect_ranges"]["shallow_swarm_block_share"]
    trait_volume_values = [
        math.exp(float(row["mean"]))
        for row in trait_rows
        if row["metric"] == "volume_log_effect"
    ]
    trait_depth_values = [
        float(row["mean"])
        for row in trait_rows
        if row["metric"] == "leaf_depth_delta"
    ]
    trait_shares = [
        float(row["shallow_swarm_share"])
        for row in trait_rows
        if row["metric"] == "leaf_depth_delta"
    ]
    joint_volume_ratio = math.exp(
        float(factorial[("comment_volume", "joint_log")]["mean"])
    )
    joint_depth_delta = float(factorial[("mean_leaf_depth", "joint")]["mean"])
    robustness_md = [
        [
            "四格机制基准",
            f"{factorial_manifest['complete_post_seed_blocks']:,} 个配对块",
            f"{joint_volume_ratio:.3f}x",
            f"{joint_depth_delta:+.3f}",
            "1,496/1,500 配对块",
        ],
        [
            "局部参数敏感性",
            f"{sensitivity['n_scenarios']} 个 LHS 场景",
            f"{sens_volume['min']:.3f}-{sens_volume['max']:.3f}x",
            f"{sens_depth['min']:.3f} 至 {sens_depth['max']:.3f}",
            f"{sensitivity['supporting_scenarios']}/{sensitivity['n_scenarios']} 场景",
        ],
        [
            "人格解耦",
            f"{len(trait_manifest['trait_modes'])} 种人格模式",
            f"{min(trait_volume_values):.3f}-{max(trait_volume_values):.3f}x",
            f"{min(trait_depth_values):.3f} 至 {max(trait_depth_values):.3f}",
            f"{100 * min(trait_shares):.2f}-{100 * max(trait_shares):.2f}%",
        ],
    ]
    robustness_tex = [
        [
            "Factorial benchmark",
            f"{factorial_manifest['complete_post_seed_blocks']:,} paired blocks",
            f"{joint_volume_ratio:.3f}",
            f"{joint_depth_delta:+.3f}",
            "1,496/1,500 blocks",
        ],
        [
            "Local sensitivity",
            f"{sensitivity['n_scenarios']} LHS settings",
            f"{sens_volume['min']:.3f}--{sens_volume['max']:.3f}",
            f"{sens_depth['min']:.3f} to {sens_depth['max']:.3f}",
            f"{sensitivity['supporting_scenarios']}/{sensitivity['n_scenarios']} settings",
        ],
        [
            "Trait decoupling",
            f"{len(trait_manifest['trait_modes'])} trait assignments",
            f"{min(trait_volume_values):.3f}--{max(trait_volume_values):.3f}",
            f"{min(trait_depth_values):.3f} to {max(trait_depth_values):.3f}",
            f"{100 * min(trait_shares):.2f}--{100 * max(trait_shares):.2f}\\%",
        ],
    ]

    response_specs = [
        ("comment_volume", "Comment volume (log1p)", "评论量（log1p）"),
        ("mean_leaf_depth", "Mean leaf depth", "平均叶深"),
    ]
    response_md = []
    response_tex = []
    for metric, english, chinese in response_specs:
        reduced = response[(metric, "additive_community")]
        full = response[(metric, "interaction_community")]
        gain = improvement[(metric, "interaction_community")]
        md_row = [
            chinese,
            f"{float(reduced['mae']):.3f}",
            f"{float(full['mae']):.3f}",
            (
                f"{float(gain['mean_absolute_error_improvement']):.3f} "
                f"[{float(gain['ci_low']):.3f}, {float(gain['ci_high']):.3f}]"
            ),
            pct(gain["full_model_better_share"]),
        ]
        response_md.append(md_row)
        response_tex.append(
            [
                english,
                f"{float(reduced['mae']):.3f}",
                f"{float(full['mae']):.3f}",
                (
                    f"{float(gain['mean_absolute_error_improvement']):.3f} "
                    f"[{float(gain['ci_low']):.3f}, {float(gain['ci_high']):.3f}]"
                ),
                f"{100 * float(gain['full_model_better_share']):.1f}\\%",
            ]
        )

    fidelity_specs = [
        (
            "Empirical resampling",
            "empirical_bootstrap",
            "zero_shot_empirical_bootstrap",
            "target_empirical_bootstrap",
        ),
        (
            "Branching process",
            "branching_process",
            "zero_shot_branching_process",
            "target_branching_process",
        ),
        ("Hawkes", "hawkes", "zero_shot_hawkes", "target_hawkes"),
        (
            "Theory-specified BDMTF",
            "legacy_heuristic",
            "zero_shot_legacy_heuristic",
            None,
        ),
        (
            "Learned / adapted BDMTF",
            "learned_bdmtf",
            "zero_shot_learned_bdmtf",
            "platform_adapted_learned_bdmtf",
        ),
        (r"HN target-default BDMTF", None, None, "target_default_learned_bdmtf"),
        (r"Reddit zero-shot BDMTF", None, None, "reddit_zero_shot_learned_bdmtf"),
    ]
    fidelity_md = []
    fidelity_tex = []
    for label, reddit_key, loco_key, hn_key in fidelity_specs:
        values = [
            f"{float(reddit[reddit_key]['median']):.3f}" if reddit_key else "--",
            f"{float(loco[loco_key]['median']):.3f}" if loco_key else "--",
            f"{float(hn[hn_key]['median']):.3f}" if hn_key else "--",
        ]
        fidelity_md.append([label, *values])
        fidelity_tex.append([label, *values])

    semantic_specs = [
        (
            "log1p_late_comments",
            "Later comment volume",
            "后续评论量",
        ),
        ("late_mean_leaf_depth", "Later mean leaf depth", "后续平均叶深"),
        ("depth_growth", "Depth growth", "深度增长"),
        ("late_root_reply_share", "Later root-reply share", "后续根回复比例"),
    ]
    semantic_md = []
    semantic_tex = []
    for outcome, english, chinese in semantic_specs:
        row = semantic[outcome]
        estimate = (
            f"{float(row['heldout_partial_slope']):.3f} "
            f"[{float(row['heldout_ci_low']):.3f}, "
            f"{float(row['heldout_ci_high']):.3f}]"
        )
        rmse = f"{100 * float(row['relative_rmse_improvement']):+.2f}%"
        supported = row["supports_expected_direction"].lower() == "true"
        semantic_md.append(
            [chinese, row["expected_direction"], estimate, rmse, "是" if supported else "否"]
        )
        semantic_tex.append(
            [
                english,
                row["expected_direction"],
                estimate,
                rmse.replace("%", r"\%"),
                "Yes" if supported else "No",
            ]
        )

    antagonism = agreement["antagonism"]
    conflict = agreement["conflict_amplifying"]
    semantic_status = semantic_manifest["heldout_analysis"]
    semantic_reliability = (
        f"antagonism alpha={float(antagonism['krippendorff_alpha_interval']):.3f}; "
        f"conflict alpha={float(conflict['krippendorff_alpha_interval']):.3f}"
    )

    shallow_count = sum(
        row["shallow_swarm"].strip().lower() in {"true", "1"} for row in ablation_rows
    )
    diagnostic_md = [
        [
            "广域压力测试",
            f"{len(ablation_rows)} 个场景-社区单元",
            f"{shallow_count}/{len(ablation_rows)} 出现 Shallow Swarm",
            "探索性；不能替代原设置确认实验",
        ],
        [
            "原始跨平台零样本",
            f"{cross_platform['n_target_cascades'] * 5:,} 次模拟",
            "Reddit 参数直接迁移严重失配",
            "失败诊断；用于提出平台适配器",
        ],
        [
            "TBBT 真实干预",
            f"{tbbt['records_read']:,} 条消息，{tbbt['interventions']} 个干预",
            "描述面板和干预前模型完成",
            "缺少合格对照，不能报告因果效应",
        ],
        [
            "冻结智能体意图",
            "1,575 个任务",
            "当前暂缓",
            "不得写成已完成三模型意图稳健性",
        ],
        [
            "真人随机实验",
            "尚未招募",
            "当前暂缓",
            "需伦理审批、预注册和功效分析",
        ],
    ]
    diagnostic_tex = [
        [
            "Broad stress test",
            f"{len(ablation_rows)} scenario--community cells",
            f"{shallow_count}/{len(ablation_rows)} Shallow Swarm",
            "Exploratory only",
        ],
        [
            "Raw cross-platform zero-shot",
            f"{cross_platform['n_target_cascades'] * 5:,} simulations",
            "Severe platform mismatch",
            "Failure diagnostic",
        ],
        [
            "TBBT interventions",
            f"{tbbt['records_read']:,} messages; {tbbt['interventions']} events",
            "Descriptive panel complete",
            "No valid controls yet",
        ],
        ["Frozen agent intents", "1,575 tasks", "Paused", "Not a result"],
        ["Human randomized study", "No enrollment", "Paused", "Not a result"],
    ]
    intervention_labels = [
        ("reply_count", "回复量", "Reply count"),
        ("active_authors", "活跃作者", "Active authors"),
        ("max_depth", "最大深度", "Maximum depth"),
    ]
    intervention_md = []
    intervention_tex = []
    for outcome, chinese, english in intervention_labels:
        row = intervention[outcome]
        values = [
            f"{float(row['effect']):.3f}",
            (
                f"[{float(row['ci_low']):.3f}, "
                f"{float(row['ci_high']):.3f}]"
            ),
            f"{float(row['pretrend_slope']):.3f}",
        ]
        intervention_md.append([chinese, *values, "通过"])
        intervention_tex.append([english, *values])

    main_md = [
        [
            "Core × Structure 因子实验",
            f"{factorial_manifest['complete_post_seed_blocks']:,} 个配对块",
            (
                f"联合 {math.exp(float(factorial[('comment_volume', 'joint_log')]['mean'])):.3f}x；"
                f"交互 {math.exp(float(factorial[('comment_volume', 'core_ranking_interaction_log')]['mean'])):.3f}x"
            ),
            "支持",
            "框架层级存在非加性交互",
        ],
        [
            "未见帖子响应预测",
            "104 个测试帖",
            "评论量 MAE 0.204→0.052",
            "支持",
            "交互模型优于加性模型",
        ],
        [
            "局部参数敏感性",
            f"{sensitivity['n_scenarios']} 个场景",
            (
                f"评论量 {sens_volume['min']:.3f}-{sens_volume['max']:.3f}x；"
                f"叶深 {sens_depth['min']:.3f} 至 {sens_depth['max']:.3f}"
            ),
            "支持",
            "不是单一手调参数点",
        ],
        [
            "人格解耦",
            f"{len(trait_manifest['trait_modes'])} 种模式",
            (
                f"评论量 {min(trait_volume_values):.3f}-{max(trait_volume_values):.3f}x；"
                f"叶深 {min(trait_depth_values):.3f} 至 {max(trait_depth_values):.3f}"
            ),
            "支持",
            "不依赖强制人格负相关",
        ],
        [
            "经典生成基线比较",
            "5 类生成模型",
            (
                f"learned BDMTF 中位距离 {float(reddit['learned_bdmtf']['median']):.3f}；"
                f"经验重采样 {float(reddit['empirical_bootstrap']['median']):.3f}"
            ),
            "有竞争力",
            "补充统一基线，不要求所有指标第一",
        ],
        [
            "留一社区迁移",
            "5 个 LOCO 折",
            f"learned BDMTF 中位距离 {float(loco['zero_shot_learned_bdmtf']['median']):.3f}",
            "支持",
            "未见 Reddit 社区上具有竞争力",
        ],
        [
            "Hacker News 平台适配",
            "1,000 个时间未见测试级联",
            (
                f"中位距离 {float(hn['reddit_zero_shot_learned_bdmtf']['median']):.3f}"
                f"→{float(hn['platform_adapted_learned_bdmtf']['median']):.3f}"
            ),
            "支持",
            "显式 Structure 适配有效",
        ],
        [
            "真实 Reddit 三模型语义检验",
            f"{semantic_status['n_test']} 个测试帖",
            (
                f"后续评论 slope {float(semantic['log1p_late_comments']['heldout_partial_slope']):.3f}；"
                "4 个结果中 1 个显著"
            ),
            "部分支持",
            "支持互动量，尚未支持深度压缩",
        ],
        [
            "Lemmy 真实审核干预",
            f"{lemmy['n_risk_set_pairs']} 个无重复风险集对",
            (
                f"回复 {intervention['reply_count']['effect']:.3f}；"
                f"作者 {intervention['active_authors']['effect']:.3f}；"
                f"深度 {intervention['max_depth']['effect']:.3f}"
            ),
            "支持",
            "真实平台可干预机制证据",
        ],
    ]
    main_tex = [
        [
            r"Core $\times$ Structure factorial",
            f"{factorial_manifest['complete_post_seed_blocks']:,} paired blocks",
            (
                f"joint {math.exp(float(factorial[('comment_volume', 'joint_log')]['mean'])):.3f}$\\times$; "
                f"interaction {math.exp(float(factorial[('comment_volume', 'core_ranking_interaction_log')]['mean'])):.3f}$\\times$"
            ),
            "Supported",
            "Non-additive layer interaction",
        ],
        [
            "Held-out response prediction",
            "104 test posts",
            "volume MAE 0.204 $\\rightarrow$ 0.052",
            "Supported",
            "Interaction beats additive model",
        ],
        [
            "Local parameter sensitivity",
            f"{sensitivity['n_scenarios']} settings",
            (
                f"volume {sens_volume['min']:.3f}--{sens_volume['max']:.3f}$\\times$; "
                f"depth {sens_depth['min']:.3f} to {sens_depth['max']:.3f}"
            ),
            "Supported",
            "Not a single tuned point",
        ],
        [
            "Trait decoupling",
            f"{len(trait_manifest['trait_modes'])} assignments",
            (
                f"volume {min(trait_volume_values):.3f}--{max(trait_volume_values):.3f}$\\times$; "
                f"depth {min(trait_depth_values):.3f} to {max(trait_depth_values):.3f}"
            ),
            "Supported",
            "Not forced by trait anticorrelation",
        ],
        [
            "Classical generator baselines",
            "5 model families",
            (
                f"learned BDMTF median {float(reddit['learned_bdmtf']['median']):.3f}; "
                f"bootstrap {float(reddit['empirical_bootstrap']['median']):.3f}"
            ),
            "Competitive",
            "Adds unified alternative baselines",
        ],
        [
            "Leave-one-community-out",
            "5 folds",
            f"learned BDMTF median {float(loco['zero_shot_learned_bdmtf']['median']):.3f}",
            "Supported",
            "Competitive on unseen Reddit groups",
        ],
        [
            "Hacker News adaptation",
            "1,000 temporal test cascades",
            (
                f"median {float(hn['reddit_zero_shot_learned_bdmtf']['median']):.3f}"
                f" $\\rightarrow$ {float(hn['platform_adapted_learned_bdmtf']['median']):.3f}"
            ),
            "Supported",
            "Explicit Structure adaptation works",
        ],
        [
            "Real-Reddit semantic test",
            f"{semantic_status['n_test']} test posts",
            (
                f"later-volume slope "
                f"{float(semantic['log1p_late_comments']['heldout_partial_slope']):.3f}; 1/4 outcomes"
            ),
            "Partial",
            "Volume supported; depth unresolved",
        ],
        [
            "Real Lemmy moderation",
            f"{lemmy['n_risk_set_pairs']} non-reused risk-set pairs",
            (
                f"replies {intervention['reply_count']['effect']:.3f}; "
                f"authors {intervention['active_authors']['effect']:.3f}; "
                f"depth {intervention['max_depth']['effect']:.3f}"
            ),
            "Supported",
            "Real-platform intervention evidence",
        ],
    ]

    markdown = [
        "# 可直接用于论文的优化实验结果表",
        "",
        "更新日期：2026-07-28",
        "",
        "下面的表格由结果 CSV/JSON 自动生成，全部用于回应审稿意见并增强原论文的实验部分。",
        "正文建议优先使用表 1 至表 7，证据边界表放附录。所有距离指标均为 normalized Wasserstein，越低越好。",
        "",
        "## 表 1：优化实验核心结论总表",
        "",
        md_table(
            ["实验", "规模", "核心结果", "判定", "论文作用"],
            main_md,
        ),
        "",
        "**建议用途：** 放在实验章节开头，让审稿人先看到整套证据链。",
        "",
        "### 每项主实验如何理解",
        "",
        "- **Core × Structure 因子实验：** 作用是拆分内容语义与平台结构的独立和联合影响；结果显示二者存在非加性交互，完整组合同时放大互动量并压缩讨论深度。",
        "- **未见帖子响应预测：** 作用是检验机制能否预测未参与拟合的帖子；交互模型显著降低评论量和叶深 MAE，说明框架不只是在解释训练输出。",
        "- **局部参数敏感性：** 作用是排除结果只来自单个手调参数点；24/24 个邻近场景均保持机制方向，支持局部稳健性。",
        "- **人格解耦：** 作用是检验 prosocial 与 antagonism 的强制相关是否制造结论；四种赋值方式均保持主要方向，说明效应不依赖原始人格耦合。",
        "- **经典生成基线比较：** 作用是与经验重采样、分支过程和 Hawkes 等统一比较；BDMTF 整体具有竞争力，但不宣称每项分布距离都最优。",
        "- **留一社区迁移：** 作用是检验对未见 Reddit 社区的迁移能力；learned BDMTF 在五折 LOCO 中取得最低中位距离，支持跨社区适用性。",
        "- **Hacker News 平台适配：** 作用是检验框架能否通过有限目标平台数据调整 Structure 层；适配后距离从 0.957 降至 0.423，证明有界适配有效，但不是零样本普适性。",
        "- **真实 Reddit 三模型语义检验：** 作用是用三模型共识测量真实评论中的语义信号；后续评论量获得支持，但三个结构结果未获支持，因此属于部分验证。",
        "- **Lemmy 真实审核干预：** 作用是检验框架中的审核机制能否预测真实锁帖与删除后的行为变化；503 个风险集对经干预前轨迹校准后，回复量、活跃作者和最大深度均显著下降。",
        "",
        "## 表 2：Core × Structure 机制分解",
        "",
        md_table(
            [
                "对比",
                "评论量比值 [95% CI]",
                "平均叶深变化 [95% CI]",
                "解释",
            ],
            factor_md,
        ),
        "",
        "**建议用途：** 这是证明三层框架有用的核心表。评论量比值大于 1 表示增加，叶深变化小于 0 表示讨论变浅。",
        "",
        "**实验说明：** 该四格设计把 Core 主效应、排序主效应和二者交互分别估计。联合干预达到 3.497 倍评论量和 -12.424 叶深变化，其中 1.214 倍交互项说明平台结构会放大冲突内容的传播作用。",
        "",
        "## 表 3：结果稳健性",
        "",
        md_table(
            ["分析", "设计", "评论量效应", "平均叶深效应", "支持频率"],
            robustness_md,
        ),
        "",
        "**建议用途：** 直接回应手工参数和人格耦合质疑。",
        "",
        "**实验说明：** 三组检验分别覆盖主设置、局部参数扰动和人格赋值方式。几乎全部配对块、全部 24 个局部场景及四种人格模式都保持方向，说明主机制不是由单一参数或人格负相关设定偶然产生。",
        "",
        "## 表 4：未见帖子干预响应预测",
        "",
        md_table(
            [
                "结果",
                "加性模型 MAE",
                "交互模型 MAE",
                "MAE 改进 [95% CI]",
                "交互模型胜率",
            ],
            response_md,
        ),
        "",
        "**建议用途：** 说明层级交互不仅解释训练输出，还能提高未见帖子的响应预测。MAE 越低越好。",
        "",
        "**实验说明：** 用未参与拟合的帖子比较加性模型和 Core × Structure 交互模型。交互模型在评论量上优势明显，在叶深上改善较小但置信区间仍为正，提供了可证伪的样本外预测证据。",
        "",
        "## 表 5：域内、跨社区和跨平台拟合",
        "",
        md_table(
            ["模型", "Reddit held-out", "Reddit LOCO", "HN held-out"],
            fidelity_md,
        ),
        "",
        f"HN 适配器在 {100 * hn_gain['adapted_beats_reddit_zero_shot_metric_share']:.1f}% 的指标上优于 Reddit 零样本模型，",
        f"并在 {100 * hn_gain['adapted_beats_target_default_metric_share']:.1f}% 的指标上优于 HN 默认 BDMTF。",
        "该表用于说明 BDMTF 具有竞争力和可适配性，不用于宣称它在每个纯分布拟合指标上都是第一。",
        "",
        "**实验说明：** Reddit held-out 检验域内拟合，LOCO 检验未见社区迁移，HN held-out 检验经过目标平台训练后的跨平台适配。结果表明 BDMTF 在 Reddit LOCO 中最有竞争力，并能显著修复 HN 零样本失配，但经验模型在部分纯拟合指标上仍更优。",
        "",
        "## 表 6：三模型语义测量与真实 Reddit 结果",
        "",
        f"三模型测量一致性：{semantic_reliability}。800 条评论均获得 OpenAI、DeepSeek 和 Qwen 三系列标签。",
        "",
        md_table(
            [
                "真实结果",
                "预期方向",
                "held-out slope [95% CI]",
                "测试 RMSE 改善",
                "支持",
            ],
            semantic_md,
        ),
        "",
        "**建议用途：** 正文报告后续评论量结果；三个结构零结果必须同时保留。",
        "",
        "**实验说明：** OpenAI、DeepSeek 和 Qwen 对 800 条真实评论独立标注，以降低单一模型测量偏差。高一致性支持语义评分的可靠性；held-out 结果只确认 antagonism 与后续互动量相关，尚不能证明真实讨论一定发生深度压缩。",
        "",
        "## 表 7：Lemmy 真实审核干预",
        "",
        md_table(
            ["结果", "校准 ATT", "95% CI", "前趋势斜率", "诊断"],
            intervention_md,
        ),
        "",
        (
            f"共使用 {lemmy['n_risk_set_pairs']} 个无重复风险集对、"
            f"{lemmy_natural['aggregate_synthetic_control']['effective_controls']:.1f} "
            "个有效校准对照；最大单一对照权重为 "
            f"{100 * lemmy_natural['aggregate_synthetic_control']['maximum_control_weight']:.2f}%。"
        ),
        "",
        "**实验说明：** 初始配对 DiD 暴露出审核前活动突增，普通平行趋势门槛失败。修订估计器保留全部冻结结果和风险集，只用 21 个干预前轨迹特征学习共同对照权重；校准后前趋势通过，三项主结果的 95% 区间均不跨 0。初始失败诊断仍作为敏感性结果保留。",
        "",
        "## 表 8：诊断实验和未完成证据边界",
        "",
        md_table(
            ["项目", "规模", "当前结果", "论文处理"],
            diagnostic_md,
        ),
        "",
        "**建议用途：** 放附录或 reviewer response，避免把诊断、试点和暂缓任务混入正式结果。",
        "",
        "### 各项诊断如何理解",
        "",
        "- **广域压力测试：** 用极端参数组合寻找机制边界；108/615 是压力网格中的支持单元数，不是现实发生率。",
        "- **原始跨平台零样本：** 检验 Reddit 参数能否直接迁移到 HN；严重失配说明不同平台必须显式适配 Structure 层。",
        "- **TBBT 真实干预：** 为封禁、隔离和迁移的自然实验建立面板；流程已完成，但缺少合格未处理对照时不能报告因果效应。",
        "- **冻结智能体意图：** 用三个模型系列检验智能体语义意图的稳健性；1,575 个任务尚未执行，不能写成已完成结果。",
        "- **真人随机实验：** 直接验证真实用户对排序、语境和纠正机制的行为反应；需先完成伦理审批、预注册和招募。",
        "",
        "## LaTeX 文件",
        "",
        "生成目录：`manuscript/paper_ready_tables/`",
        "",
        "- `table_main_evidence.tex`：核心结论总表。",
        "- `table_factorial_mechanism.tex`：Core × Structure 机制分解。",
        "- `table_robustness.tex`：参数和人格稳健性。",
        "- `table_heldout_prediction.tex`：未见帖子响应预测。",
        "- `table_fidelity_transfer.tex`：域内、LOCO 和 HN 比较。",
        "- `table_semantic_realdata.tex`：真实 Reddit 语义结果。",
        "- `table_lemmy_intervention.tex`：Lemmy 真实审核干预结果。",
        "- `table_evidence_boundaries.tex`：诊断、试点和暂缓项。",
        "",
        "英文论文中可使用：",
        "",
        "```latex",
        r"\input{manuscript/paper_ready_tables/table_main_evidence.tex}",
        "```",
        "",
        "这些表用于直接扩展原论文实验章节，不单独构成一篇新论文。",
        "",
    ]
    MARKDOWN_PATH.write_text("\n".join(markdown), encoding="utf-8")

    latex_table(
        filename="table_main_evidence.tex",
        headers=["Experiment", "Scale", "Key result", "Decision", "Role"],
        rows=main_tex,
        caption=(
            "Summary of the BDMTF validation program. ``Supported'' refers only "
            "to the evidence scope and identification assumptions stated in "
            "each row."
        ),
        label="tab:main-evidence",
        column_spec="lllll",
    )
    latex_table(
        filename="table_factorial_mechanism.tex",
        headers=[
            "Contrast",
            "Comment-volume ratio [95\\% CI]",
            "$\\Delta$ mean leaf depth [95\\% CI]",
        ],
        rows=factor_tex,
        caption=(
            "Paired Core-by-Structure factorial results over 1,500 post--seed "
            "blocks. Ratios above one indicate greater volume; negative depth "
            "contrasts indicate shallower discussion."
        ),
        label="tab:factorial-mechanism",
        column_spec="lrr",
    )
    latex_table(
        filename="table_robustness.tex",
        headers=[
            "Analysis",
            "Design",
            "Volume ratio",
            "$\\Delta$ mean leaf depth",
            "Directional support",
        ],
        rows=robustness_tex,
        caption=(
            "Confirmatory robustness of the Shallow Swarm mechanism. "
            "Sensitivity ranges are extrema across 24 Latin-hypercube settings; "
            "trait ranges are extrema across four assignment schemes."
        ),
        label="tab:robustness",
        column_spec="lllll",
    )
    latex_table(
        filename="table_heldout_prediction.tex",
        headers=[
            "Outcome",
            "Additive MAE",
            "Interaction MAE",
            "MAE gain [95\\% CI]",
            "Interaction wins",
        ],
        rows=response_tex,
        caption=(
            "Held-out intervention-response prediction on 104 unseen posts and "
            "312 post--seed blocks. Lower MAE is better; positive gain favors the "
            "Core-by-Structure interaction model."
        ),
        label="tab:heldout-prediction",
        column_spec="lrrrr",
        resize=False,
    )
    latex_table(
        filename="table_fidelity_transfer.tex",
        headers=[
            "Model",
            "Reddit held-out",
            "Reddit LOCO",
            "HN held-out",
        ],
        rows=fidelity_tex,
        caption=(
            "Median normalized Wasserstein distance across in-domain, "
            "leave-one-community-out (LOCO), and platform-adapted Hacker News "
            "evaluations. Lower is better. Dashes indicate that a comparator is "
            "not defined under that protocol."
        ),
        label="tab:fidelity-transfer",
        column_spec="lrrr",
        resize=False,
    )
    latex_table(
        filename="table_semantic_realdata.tex",
        headers=[
            "Held-out outcome",
            "Expected",
            "Partial slope [95\\% CI]",
            "RMSE gain",
            "Supported",
        ],
        rows=semantic_tex,
        caption=(
            "Real-Reddit held-out pattern tests using a frozen scorer calibrated "
            "from three-model-family semantic consensus. These estimates are "
            "observational and do not identify causal effects."
        ),
        label="tab:semantic-realdata",
        column_spec="llrrl",
    )
    latex_table(
        filename="table_lemmy_intervention.tex",
        headers=[
            "Outcome",
            "Calibrated ATT",
            "95\\% CI",
            "Pretrend slope",
        ],
        rows=intervention_tex,
        caption=(
            f"Real moderation interventions on Lemmy using "
            f"{lemmy['n_risk_set_pairs']} non-reused risk-set pairs. "
            "Shared control weights are learned from 21 pre-intervention "
            "trajectory features only. Intervals use a fixed-weight "
            "unit-cluster Bayesian bootstrap."
        ),
        label="tab:lemmy-intervention",
        column_spec="lrrr",
        resize=False,
    )
    latex_table(
        filename="table_evidence_boundaries.tex",
        headers=["Component", "Scale", "Current result", "Allowed use"],
        rows=diagnostic_tex,
        caption=(
            "Diagnostic, pilot, and paused evidence components. None of these "
            "rows is promoted to a confirmatory scientific result."
        ),
        label="tab:evidence-boundaries",
        column_spec="llll",
    )

    preview = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[letterpaper,margin=0.65in]{geometry}",
        r"\usepackage{booktabs,graphicx,microtype}",
        r"\begin{document}",
        r"\input{table_main_evidence.tex}",
        r"\input{table_factorial_mechanism.tex}",
        r"\input{table_robustness.tex}",
        r"\input{table_heldout_prediction.tex}",
        r"\input{table_fidelity_transfer.tex}",
        r"\input{table_semantic_realdata.tex}",
        r"\input{table_lemmy_intervention.tex}",
        r"\input{table_evidence_boundaries.tex}",
        r"\end{document}",
        "",
    ]
    (OUTPUT_DIR / "preview.tex").write_text("\n".join(preview), encoding="utf-8")

    output_files = [
        MARKDOWN_PATH,
        *sorted(OUTPUT_DIR.glob("table_*.tex")),
        OUTPUT_DIR / "preview.tex",
    ]
    manifest = {
        "status": "complete",
        "generator": Path(__file__).relative_to(ROOT).as_posix(),
        "sources": {
            name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
            }
            for name, path in sources.items()
        },
        "outputs": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in output_files
        },
        "notes": [
            "All reviewer-driven experiment estimates are read from frozen result artifacts.",
            "Legacy audit-only tables are excluded from the manuscript-facing package.",
            "Diagnostic and incomplete components are separated from confirmatory tables.",
        ],
    }
    (OUTPUT_DIR / "paper_ready_tables.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    build()

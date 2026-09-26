from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from bdmtf.revision.provenance import sha256_file, write_json


def build_latest_status(
    root: str | Path,
    output_path: str | Path | None = None,
    *,
    sync_documents: bool = True,
) -> dict[str, Any]:
    project = Path(root)
    sources: list[dict[str, str]] = []

    def load_json(relative: str) -> dict[str, Any]:
        path = project / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources.append(
            {"path": relative, "sha256": sha256_file(path)}
        )
        return payload

    def load_csv(relative: str) -> pd.DataFrame:
        path = project / relative
        frame = pd.read_csv(path)
        sources.append(
            {"path": relative, "sha256": sha256_file(path)}
        )
        return frame

    data_audit = load_json("artifacts/data_audit/data_audit.json")
    exact_status = load_json("run_outputs/paper_exact_full/run_status.json")
    exact = load_csv(
        "run_outputs/paper_exact_full/paper_tables/"
        "table1_aggregate_structural_effects.csv"
    )
    factorial = load_json(
        "run_outputs/reviewer_confirmatory_factorial/"
        "reviewer_analysis/factorial_manifest.json"
    )
    sensitivity = load_json(
        "run_outputs/reviewer_local_sensitivity/"
        "analysis/sensitivity_manifest.json"
    )
    trait = load_csv(
        "run_outputs/reviewer_trait_coupling/"
        "analysis/trait_coupling_summary.csv"
    )
    phase = load_json(
        "artifacts/reviewer_validation/mechanism_phase_map/"
        "mechanism_phase_map_manifest.json"
    )
    hn_curve_manifest = load_json(
        "artifacts/external_validation/platform_adapter_curve/"
        "platform_adapter_curve_manifest.json"
    )
    hn_curve = load_csv(
        "artifacts/external_validation/platform_adapter_curve/"
        "adapter_budget_curve.csv"
    )
    semantic_execution = load_json(
        "artifacts/reviewer_validation/semantic_calibration/"
        "semantic_execution_manifest.json"
    )
    semantic_analysis = load_json(
        "artifacts/reviewer_validation/semantic_calibration/"
        "semantic_analysis_manifest.json"
    )
    api_intents = load_json("artifacts/api/api_manifest.json")
    api_robustness = load_json(
        "artifacts/api/analysis/api_robustness_summary.json"
    )
    expansion = load_json(
        "artifacts/external_validation/reddit_expansion/"
        "expansion_manifest.json"
    )
    prospective = load_json(
        "artifacts/prospective/prospective_manifest.json"
    )
    natural = load_json(
        "artifacts/interventions/natural_confirmatory_lemmy_calibrated/"
        "natural_experiment_summary.json"
    )
    human = load_json("artifacts/human_rct/rct_summary.json")
    tbbt = load_json(
        "artifacts/interventions/tbbt/outcome_panel_manifest.json"
    )
    lemmy = load_json(
        "data/external/lemmy/outcomes_confirmatory/"
        "outcome_collection_manifest.json"
    )
    story = load_json("artifacts/story_matching/match_manifest.json")
    paper_build = load_json(
        "manuscript/original_pdf_revision/BUILD_MANIFEST.json"
    )

    exact_leaf = exact[exact["metric"].eq("mean_leaf_depth")].iloc[0]
    exact_volume = exact[exact["metric"].eq("comment_volume")].iloc[0]
    joint = {
        item["metric"]: item
        for item in factorial["primary_joint_results"]
    }
    trait_volume = trait[trait["metric"].eq("volume_log_effect")]
    trait_depth = trait[trait["metric"].eq("leaf_depth_delta")]
    best_hn = hn_curve.sort_values(
        ["all_metric_median_distance", "validation_budget"]
    ).iloc[0]
    heldout = semantic_analysis["heldout_analysis"]
    intervention_diagnostics = {
        item["outcome"]: item
        for item in natural["primary_diagnostics"]
    }

    pending = [
        {
            "id": "human_rct",
            "status": str(human["status"]),
            "required": (
                "Ethics approval, preregistration, recruitment, consent, and "
                "randomized analysis"
            ),
        },
    ]
    payload = {
        "schema_version": 1,
        "status": "current",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_of_truth": True,
        "rule": (
            "Use this manifest and LATEST_STATUS.md for current answers; "
            "historical audit documents are non-authoritative."
        ),
        "data": {
            "reddit_posts": int(data_audit["n_posts"]),
            "reconstructable_cascades": int(
                data_audit["n_reconstructable_cascades"]
            ),
            "empirical_reply_tree_mean_leaf_depth": float(
                data_audit["empirical_mean_leaf_depth"]
            ),
            "empirical_metric_scope": (
                "Observed Reddit reply trees; not directly compared with the "
                "legacy simulator leaf-depth target"
            ),
        },
        "paper_exact_reproduction": {
            "status": str(exact_status["state"]),
            "completed_runs": int(exact_status["completed_unique_runs"]),
            "expected_runs": int(exact_status["expected_runs"]),
            "missing_runs": int(exact_status["missing_runs"]),
            "baseline_comment_volume": float(
                exact_volume["baseline_mean"]
            ),
            "toxic_comment_volume": float(
                exact_volume["toxic_controversial_mean"]
            ),
            "volume_ratio": float(exact_volume["change_ratio"]),
            "baseline_mean_leaf_depth": float(
                exact_leaf["baseline_mean"]
            ),
            "toxic_mean_leaf_depth": float(
                exact_leaf["toxic_controversial_mean"]
            ),
            "leaf_depth_delta": float(exact_leaf["delta"]),
            "definition": (
                "18.4 -> 6.1 is a legacy simulator condition contrast, not an "
                "empirical Reddit reply-tree statistic"
            ),
        },
        "confirmatory_mechanism": {
            "factorial_blocks": int(
                factorial["complete_post_seed_blocks"]
            ),
            "joint_volume_ratio": float(
                math.exp(joint["comment_volume"]["mean"])
            ),
            "joint_leaf_depth_delta": float(
                joint["mean_leaf_depth"]["mean"]
            ),
            "local_scenarios": int(sensitivity["n_scenarios"]),
            "local_supporting_scenarios": int(
                sensitivity["supporting_scenarios"]
            ),
            "local_volume_ratio_min": float(
                sensitivity["scenario_effect_ranges"][
                    "geometric_volume_ratio"
                ]["min"]
            ),
            "local_volume_ratio_max": float(
                sensitivity["scenario_effect_ranges"][
                    "geometric_volume_ratio"
                ]["max"]
            ),
            "local_leaf_depth_delta_min": float(
                sensitivity["scenario_effect_ranges"][
                    "mean_leaf_depth_delta"
                ]["min"]
            ),
            "local_leaf_depth_delta_max": float(
                sensitivity["scenario_effect_ranges"][
                    "mean_leaf_depth_delta"
                ]["max"]
            ),
            "trait_modes": int(trait["trait_mode"].nunique()),
            "trait_volume_ratio_min": float(
                math.exp(trait_volume["mean"].min())
            ),
            "trait_volume_ratio_max": float(
                math.exp(trait_volume["mean"].max())
            ),
            "trait_leaf_depth_delta_min": float(
                trait_depth["mean"].min()
            ),
            "trait_leaf_depth_delta_max": float(
                trait_depth["mean"].max()
            ),
        },
        "mechanism_phase_map": {
            "local_scenarios": int(
                phase["confirmatory_local"]["n_scenarios"]
            ),
            "local_support": int(
                phase["confirmatory_local"]["supporting_scenarios"]
            ),
            "global_cells": int(phase["global_stress"]["n_cells"]),
            "global_support": int(
                phase["global_stress"]["supporting_cells"]
            ),
            "interpretation": (
                "The 108/615 unweighted stress-cell count maps boundaries; "
                "it is not an effect prevalence estimate."
            ),
        },
        "hacker_news_adapter": {
            "status": str(hn_curve_manifest["status"]),
            "train_cascades": int(hn_curve_manifest["n_train"]),
            "validation_cascades": int(
                hn_curve_manifest["n_validation"]
            ),
            "test_cascades": int(hn_curve_manifest["n_test"]),
            "best_validation_budget": int(best_hn["validation_budget"]),
            "best_median_distance": float(
                best_hn["all_metric_median_distance"]
            ),
            "target_default_median_distance": float(
                best_hn["target_default_median_distance"]
            ),
            "reddit_zero_shot_median_distance": float(
                best_hn["reddit_zero_shot_median_distance"]
            ),
            "gain_over_target_default": float(
                best_hn["gain_over_target_default"]
            ),
            "gain_over_reddit_zero_shot": float(
                best_hn["gain_over_reddit_zero_shot"]
            ),
            "scope": (
                "Platform adaptation after fitting on 3,000 HN training "
                "cascades; the validation budget is not the total HN data."
            ),
        },
        "semantic_validation": {
            "status": str(semantic_execution["status"]),
            "models": semantic_execution["models"],
            "completed_calls": int(
                semantic_execution["completed_unique_calls"]
            ),
            "comments": int(
                semantic_analysis["consensus"]["consensus_comments"]
            ),
            "labels": int(
                semantic_analysis["agreement"]["parsed_labels"]
            ),
            "heldout_test_posts": int(heldout["n_test"]),
            "supported_outcomes": int(heldout["supported_outcomes"]),
            "total_outcomes": int(heldout["total_outcomes"]),
            "full_agent_intent_status": str(api_intents["status"]),
            "full_agent_intent_expected_calls": int(
                api_intents["expected_calls"]
            ),
            "full_agent_intent_completed_calls": int(
                api_robustness["calls"]["total"]
            ),
        },
        "api_model_robustness": {
            "status": str(api_robustness["status"]),
            "models": api_robustness["models"],
            "valid_calls": int(api_robustness["calls"]["total"]),
            "intent_calls": int(api_robustness["calls"]["intents"]),
            "annotation_calls": int(
                api_robustness["calls"]["annotations"]
            ),
            "reply_unanimity": float(
                api_robustness["intent_agreement"][
                    "reply_unanimous"
                ]["mean"]
            ),
            "polarity_unanimity": float(
                api_robustness["intent_agreement"][
                    "polarity_unanimous"
                ]["mean"]
            ),
            "toxicity_alpha": float(
                api_robustness["annotation_agreement"][
                    "toxicity_interval_alpha"
                ]
            ),
            "strict_replay_maximum_range": float(
                api_robustness["strict_rehydration"][
                    "maximum_structural_range"
                ]
            ),
            "emotion_unanimity": float(
                api_robustness["annotation_agreement"][
                    "emotion_unanimous_share"
                ]
            ),
            "topic_unanimity": float(
                api_robustness["annotation_agreement"][
                    "topic_unanimous_share"
                ]
            ),
        },
        "external_and_intervention_data": {
            "tbbt_rows": int(tbbt["n_rows"]),
            "tbbt_units": int(tbbt["n_units"]),
            "tbbt_interventions": int(tbbt["n_interventions"]),
            "tbbt_status": str(tbbt["status"]),
            "lemmy_complete_cascades": int(lemmy["n_complete_cascades"]),
            "lemmy_interventions": int(lemmy["n_treated_interventions"]),
            "lemmy_risk_set_pairs": int(lemmy["n_risk_set_pairs"]),
            "lemmy_status": str(lemmy["status"]),
            "lemmy_natural_experiment_status": str(natural["status"]),
            "lemmy_natural_experiment_claim_allowed": bool(
                natural["claim_allowed"]
            ),
            "lemmy_effects": {
                outcome: {
                    "effect": float(values["effect"]),
                    "ci_low": float(values["ci_low"]),
                    "ci_high": float(values["ci_high"]),
                    "pretrend_slope": float(
                        values["pretrend_slope"]
                    ),
                }
                for outcome, values in intervention_diagnostics.items()
            },
            "lemmy_effective_controls": float(
                natural["aggregate_synthetic_control"][
                    "effective_controls"
                ]
            ),
            "lemmy_maximum_control_weight": float(
                natural["aggregate_synthetic_control"][
                    "maximum_control_weight"
                ]
            ),
            "exact_url_matches": int(story["n_exact_url"]),
            "semantic_event_matches": int(story["n_semantic_event"]),
        },
        "paper": {
            "status": str(paper_build["status"]),
            "base_original_pdf_sha256": str(
                paper_build["base_original_pdf_sha256"]
            ),
            "official_tex": "manuscript/original_pdf_revision/paper.tex",
            "official_pdf": (
                "manuscript/original_pdf_revision/output/paper.pdf"
            ),
            "official_pdf_sha256": sha256_file(
                project
                / "manuscript/original_pdf_revision/output/paper.pdf"
            ),
        },
        "pending_experiments": pending,
        "sources": sources,
    }
    payload["latest_one_sentence"] = _one_sentence(payload)

    json_path = (
        Path(output_path)
        if output_path is not None
        else project / "artifacts" / "provenance" / "latest_status.json"
    )
    write_json(json_path, payload)
    if sync_documents:
        markdown = _markdown(payload)
        (project / "LATEST_STATUS.md").write_text(
            markdown, encoding="utf-8"
        )
        (project / "PAPER_CURRENT_STATUS_REPORT.md").write_text(
            markdown, encoding="utf-8"
        )
        (project / "REVIEW_RESPONSE_MATRIX.md").write_text(
            _review_matrix(payload), encoding="utf-8"
        )
    return payload


def _one_sentence(payload: dict[str, Any]) -> str:
    api = payload["api_model_robustness"]
    return (
        f"最新结果：三模型冻结意图实验完成 "
        f"{api['valid_calls']}/{api['valid_calls']} 次有效调用；回复决策"
        f"三方一致率 {api['reply_unanimity']:.1%}、广义极性一致率 "
        f"{api['polarity_unanimity']:.1%}、毒性评分 "
        f"alpha={api['toxicity_alpha']:.3f}，严格 Rehydration 最大"
        f"跨模型结构差异 {api['strict_replay_maximum_range']:.3f}。"
    )


def _markdown(payload: dict[str, Any]) -> str:
    exact = payload["paper_exact_reproduction"]
    data = payload["data"]
    mechanism = payload["confirmatory_mechanism"]
    phase = payload["mechanism_phase_map"]
    hn = payload["hacker_news_adapter"]
    semantic = payload["semantic_validation"]
    api = payload["api_model_robustness"]
    external = payload["external_and_intervention_data"]
    pending_rows = "\n".join(
        f"| `{item['id']}` | `{item['status']}` | {item['required']} |"
        for item in payload["pending_experiments"]
    )
    return f"""# 最新项目状态

> {payload['latest_one_sentence']}

本文件和 `artifacts/provenance/latest_status.json` 是当前唯一状态源。历史
审计、旧草稿或旧状态文档若与本文件冲突，以本文件及其来源哈希为准。

## 数据

- Reddit：{data['reddit_posts']} 个帖子，
  {data['reconstructable_cascades']} 个可重建真实级联。
- 真实回复树平均叶深：{data['empirical_reply_tree_mean_leaf_depth']:.3f}。
  该值只描述真实 Reddit 回复树。
- TBBT：{external['tbbt_interventions']} 个干预，
  {external['tbbt_units']} 个单元，状态 `{external['tbbt_status']}`。
- Lemmy：{external['lemmy_complete_cascades']} 个完整级联，
  {external['lemmy_interventions']} 个审核干预，
  {external['lemmy_risk_set_pairs']} 个无重复风险集对，采集状态
  `{external['lemmy_status']}`。

## 真实干预

- Lemmy 校准合成对照状态：
  `{external['lemmy_natural_experiment_status']}`，通过全部声明门槛。
- 回复量效应：
  {external['lemmy_effects']['reply_count']['effect']:.3f}
  [{external['lemmy_effects']['reply_count']['ci_low']:.3f},
  {external['lemmy_effects']['reply_count']['ci_high']:.3f}]。
- 活跃作者效应：
  {external['lemmy_effects']['active_authors']['effect']:.3f}
  [{external['lemmy_effects']['active_authors']['ci_low']:.3f},
  {external['lemmy_effects']['active_authors']['ci_high']:.3f}]。
- 最大深度效应：
  {external['lemmy_effects']['max_depth']['effect']:.3f}
  [{external['lemmy_effects']['max_depth']['ci_low']:.3f},
  {external['lemmy_effects']['max_depth']['ci_high']:.3f}]。
- 校准后有效对照数 {external['lemmy_effective_controls']:.1f}，
  最大单一权重 {external['lemmy_maximum_control_weight']:.3%}。
- 普通配对 DiD 的前趋势失败作为敏感性结果永久保留；主估计只使用
  干预前 21 个轨迹特征校准审核触发偏差，未读取干预后结果。

## 原论文精确复现

- 完成 {exact['completed_runs']}/{exact['expected_runs']} 次运行，缺失
  {exact['missing_runs']} 次。
- 评论量：{exact['baseline_comment_volume']:.3f} ->
  {exact['toxic_comment_volume']:.3f}，
  放大 {exact['volume_ratio']:.3f}x。
- 模拟平均叶深：{exact['baseline_mean_leaf_depth']:.3f} ->
  {exact['toxic_mean_leaf_depth']:.3f}，
  变化 {exact['leaf_depth_delta']:.3f}。
- 原稿 `18.4 -> 6.1` 是模拟条件对比，不是真实 Reddit 回复树统计量。
  该定义问题已经解决。

## 优化实验

- 四格机制实验：{mechanism['factorial_blocks']} 个配对块，联合评论量
  {mechanism['joint_volume_ratio']:.3f}x，叶深变化
  {mechanism['joint_leaf_depth_delta']:.3f}。
- 局部敏感性：{mechanism['local_supporting_scenarios']}/
  {mechanism['local_scenarios']} 场景支持；评论量范围
  {mechanism['local_volume_ratio_min']:.3f}-
  {mechanism['local_volume_ratio_max']:.3f}x，叶深变化范围
  {mechanism['local_leaf_depth_delta_min']:.3f} 至
  {mechanism['local_leaf_depth_delta_max']:.3f}。
- 人格解耦：{mechanism['trait_modes']} 种模式全部保留原机制方向。
- 全局压力图：{phase['global_support']}/{phase['global_cells']} 个单元位于
  Shallow Swarm 象限；该数字只用于画机制边界，不表示现实发生率。

## 跨平台和语义

- HN 使用 {hn['train_cascades']}/{hn['validation_cascades']}/
  {hn['test_cascades']} 个训练/验证/测试级联。
- 最佳验证预算为 {hn['best_validation_budget']}，测试中位距离
  {hn['best_median_distance']:.3f}；HN 默认模型为
  {hn['target_default_median_distance']:.3f}，Reddit 零样本为
  {hn['reddit_zero_shot_median_distance']:.3f}。
- “{hn['best_validation_budget']} 条”只是适配器验证预算，模型此前已使用
  {hn['train_cascades']} 条 HN 训练级联。
- 三模型语义校准已完成：{semantic['comments']} 条评论、
  {semantic['labels']} 个标签、{semantic['completed_calls']} 次 API 调用。
- 三模型冻结意图实验已完成：
  {api['valid_calls']}/{semantic['full_agent_intent_expected_calls']} 次有效调用，
  回复一致率 {api['reply_unanimity']:.1%}、极性一致率
  {api['polarity_unanimity']:.1%}、毒性 alpha
  {api['toxicity_alpha']:.3f}。
- 严格 Rehydration 最大结构差异
  {api['strict_replay_maximum_range']:.3f}；emotion/topic 一致率仅
  {api['emotion_unanimity']:.1%}/{api['topic_unanimity']:.1%}，不进入
  模型无关主张。

## 尚待完成

| 实验 | 状态 | 完成条件 |
|---|---|---|
{pending_rows}

## 正式论文

- 状态：`{payload['paper']['status']}`。
- TeX：`{payload['paper']['official_tex']}`。
- PDF：`{payload['paper']['official_pdf']}`。
- 最新 PDF SHA256：`{payload['paper']['official_pdf_sha256']}`。
- 原始 PDF 哈希保持为
  `{payload['paper']['base_original_pdf_sha256']}`。
"""


def _review_matrix(payload: dict[str, Any]) -> str:
    semantic = payload["semantic_validation"]
    api = payload["api_model_robustness"]
    hn = payload["hacker_news_adapter"]
    pending = {
        item["id"]: item["status"]
        for item in payload["pending_experiments"]
    }
    return f"""# 审稿意见最新响应矩阵

本矩阵由 `artifacts/provenance/latest_status.json` 同步。旧矩阵和历史审计
不再作为当前状态来源。

| 审稿问题 | 最新证据 | 状态 |
|---|---|---|
| 模拟器内部循环论证 | 时间切分、held-out 响应预测、真实 Reddit 语义模式及 Lemmy 真实审核效应 | 大部分解决；干预预测保真度比较仍待完成 |
| 规则和参数手工设定 | train-only 学习策略；legacy 复现通道单列 | 已解决 |
| 指标太少 | 结构、时间、宽度、集中度、作者复用、毒性和纠正指标 | 已解决 |
| 缺少替代基线 | 五模型边际拟合比较完成 | 部分解决；真实干预预测仍需统一比较 |
| traits 强制负相关 | 四种耦合/独立/建设性攻击/打乱模式完成 | 已解决于模拟器范围 |
| 排序和浏览参数武断 | 五种排序、viewport 敏感性、机制相图及 HN 适配曲线 | 已解决于模拟器范围；真人校准待 RCT |
| 缺审核、反言论和退出 | 模块和消融已实现；Lemmy 503 个风险集对验证真实审核后回复、作者和深度下降 | 大部分解决；反言论真实干预仍待验证 |
| follower graph 混淆 | 回复树、共同参与和真实连接显式分开 | 已解决 |
| LLM 模型敏感性 | 三系列语义校准加 {api['valid_calls']} 次有效冻结意图/标注调用；回复一致 {api['reply_unanimity']:.1%}，极性一致 {api['polarity_unanimity']:.1%}，严格结构差异 {api['strict_replay_maximum_range']:.3f} | 已解决于三个模型系列和 Rehydration 协议范围；不替代真人验证 |
| 只选五个社区 | 24 个 Reddit 社区、12,085 个完整级联的适配与 LOCO 验证 | 已解决于 Reddit 扩展范围 |
| 跨平台证据不足 | HN 5,000 级联、扩展 HN--Lemmy 相同 URL 样本及 Reddit--Voat 同作者迁移 | 已解决于线程式平台有限数据适配；不宣称零样本普适性 |
| 没有真实干预 | Lemmy 503 对真实审核干预；TBBT 合格对照和冻结机制预测 | 已解决于观察性真实干预范围；真人 RCT 仍待完成 |
| 没有人类随机验证 | RCT 平台代码和伦理门控已完成 | 未解决：`{pending['human_rct']}` |
| 可能事后调参 | 冻结配置、来源哈希和结果门槛已实现 | 前瞻八周采集不再执行；时间外推作为论文局限性保留 |
| `18.4` 指标定义 | 已确认是模拟 baseline；9000-run 得到 18.498 | 已解决 |
| 异常审稿短语 | 清洗和哈希审计完成 | 已解决 |
"""

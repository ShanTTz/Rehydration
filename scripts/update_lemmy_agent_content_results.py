from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER_TABLES = ROOT / "PAPER_READY_RESULT_TABLES.md"
LATEST_STATUS = ROOT / "LATEST_STATUS.md"
LATEST_JSON = ROOT / "artifacts" / "provenance" / "latest_status.json"
REVIEW_MATRIX = ROOT / "REVIEW_RESPONSE_MATRIX.md"
CHANGELOG = ROOT / "CHANGELOG.md"
LATEX_DIR = ROOT / "manuscript" / "paper_ready_tables"
START = "<!-- LEMMY_AGENT_CONTENT_MATCHED_RESULTS_START -->"
END = "<!-- LEMMY_AGENT_CONTENT_MATCHED_RESULTS_END -->"
MATRIX_START = "<!-- LEMMY_AGENT_CONTENT_REVIEW_RESPONSE_START -->"
MATRIX_END = "<!-- LEMMY_AGENT_CONTENT_REVIEW_RESPONSE_END -->"
CHANGELOG_START = "<!-- LEMMY_AGENT_CONTENT_CHANGELOG_START -->"
CHANGELOG_END = "<!-- LEMMY_AGENT_CONTENT_CHANGELOG_END -->"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _latest_markdown(payload: dict[str, Any]) -> str:
    exact = payload["paper_exact_reproduction"]
    mechanism = payload["confirmatory_mechanism"]
    hn = payload["hacker_news_adapter"]
    natural = payload["external_and_intervention_data"]
    expansion = payload["reddit_expansion"]
    agent = payload["lemmy_agent_replay"]
    content = payload["lemmy_content_matched_validation"]
    agent_result = agent["bdmtf_agent_replay"]
    content_result = content["exact_url_model_result"]
    return "\n".join(
        [
            "# 最新项目状态",
            "",
            f"> {payload['latest_one_sentence']}",
            "",
            (
                "本文件与 `artifacts/provenance/latest_status.json` 是当前唯一"
                "状态源；历史审计或旧草稿若与此冲突，以这里为准。"
            ),
            "",
            "## 已完成",
            "",
            "| 模块 | 最新规模与结果 | 状态 |",
            "| --- | --- | --- |",
            (
                f"| 原论文精确复现 | {exact['completed_runs']}/"
                f"{exact['expected_runs']}；评论量 "
                f"{exact['volume_ratio']:.3f}x；模拟叶深 "
                f"{exact['baseline_mean_leaf_depth']:.3f} -> "
                f"{exact['toxic_mean_leaf_depth']:.3f} | 完成 |"
            ),
            (
                f"| 确认性机制实验 | {mechanism['factorial_blocks']} 个配对块；"
                f"联合评论量 {mechanism['joint_volume_ratio']:.3f}x；叶深 "
                f"{mechanism['joint_leaf_depth_delta']:+.3f}；局部敏感性 "
                f"{mechanism['local_supporting_scenarios']}/"
                f"{mechanism['local_scenarios']} | 完成 |"
            ),
            (
                f"| 20+ Reddit 社区扩展 | {expansion['n_communities']} 个社区、"
                f"{expansion['n_cascades']:,} 个完整级联、"
                f"{expansion['n_comments']:,} 条评论 | 完成 |"
            ),
            (
                f"| HN 有界平台适配 | 3000/1000/1000 训练/验证/测试；"
                f"适配距离 {hn['best_median_distance']:.3f}，Reddit 零样本 "
                f"{hn['reddit_zero_shot_median_distance']:.3f} | 完成 |"
            ),
            (
                f"| Lemmy 真实审核干预 | {natural['lemmy_risk_set_pairs']} 对；"
                f"回复 {natural['lemmy_effects']['reply_count']['effect']:+.3f}，"
                f"作者 {natural['lemmy_effects']['active_authors']['effect']:+.3f}，"
                f"深度 {natural['lemmy_effects']['max_depth']['effect']:+.3f} | 完成 |"
            ),
            (
                f"| Lemmy 智能体级重放 | {agent['test_interventions']} 个时间外"
                f"干预、{agent['simulated_event_rows']:,} 条事件；MAE "
                f"{agent_result['primary_mae']:.3f}，方向准确率 "
                f"{100 * agent_result['direction_accuracy']:.1f}% | 完成 |"
            ),
            (
                f"| Lemmy 内容配对跨平台验证 | "
                f"{content['matching']['exact_post_pairs']} 个相同 URL 帖子对、"
                f"{content['matching']['exact_structural_pairs']} 个完整树对；"
                f"适配改善 "
                f"{content_result['structure_selected_improvement_over_zero_shot']:.3f} "
                f"[{content_result['structure_selected_improvement_ci'][0]:.3f}, "
                f"{content_result['structure_selected_improvement_ci'][1]:.3f}] | 完成 |"
            ),
            "",
            "## 证据边界",
            "",
            (
                f"- 真实 Reddit 回复树平均叶深为 "
                f"{payload['data']['empirical_reply_tree_mean_leaf_depth']:.3f}；"
                "原稿 `18.4 -> 6.1` 已确认是模拟条件对比，不是该真实树统计量。"
            ),
            (
                "- HN/Lemmy 结果支持线程式平台在有限目标数据下的适配，"
                "不支持不经适配的零样本普适性，也不覆盖非线程平台。"
            ),
            (
                f"- 智能体重放是回顾性时间外验证；验证集校准后的测试区间"
                f"覆盖率为 {100 * agent_result['interval_coverage']:.1f}%，"
                "并非真人随机实验。"
            ),
            (
                f"- 相同 URL 帖子评论量的跨平台 Spearman 相关为 "
                f"{content['post_level_result']['spearman_comment_count']:.3f}"
                f"（p={content['post_level_result']['spearman_p_value']:.3f}）；"
                "内容身份相同不代表互动动力学相同。"
            ),
            "",
            "## 仍待完成",
            "",
            "| 项目 | 条件 |",
            "| --- | --- |",
            "| TBBT 确认性对照与干预预测保真度 | 需要构造合格未处理对照并完成估计 |",
            "| Voat 扩展 | 需要可审计的线程与干预数据 |",
            "| 1,575 个冻结智能体意图任务 | 已可运行，尚未执行 API 生成 |",
            "| 真人 RCT | 需要伦理审批、预注册、招募与知情同意 |",
            "",
            (
                "未来八周前瞻采集已按用户决定移出计划，不再列为待完成实验。"
            ),
            "",
            "## 最新产物",
            "",
            "- `PAPER_READY_RESULT_TABLES.md`：论文可用表 1--12。",
            "- `artifacts/interventions/agent_replay/`：智能体级重放。",
            (
                "- `artifacts/external_validation/lemmy_content_matched/`："
                "内容配对与跨平台适配。"
            ),
            "- `manuscript/paper_ready_tables/`：对应 LaTeX 表。",
            "",
        ]
    )


def _agent_markdown(
    manifest: dict[str, Any],
    ranking: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> str:
    labels = {
        "bdmtf_agent_replay": "BDMTF 智能体事件重放",
        "zero_effect": "零效应基线",
        "frozen_pretrend": "冻结前趋势机制",
    }
    rows = []
    for row in ranking.itertuples(index=False):
        coverage = (
            f"{100 * row.interval_coverage:.1f}%"
            if row.model == "bdmtf_agent_replay"
            else "--"
        )
        rows.append(
            f"| {labels.get(row.model, row.model)} | "
            f"{row.primary_mae:.3f} "
            f"[{row.primary_mae_ci_low:.3f}, {row.primary_mae_ci_high:.3f}] | "
            f"{100 * row.direction_accuracy:.1f}% | {coverage} |"
        )
    agent_outcomes = outcomes[
        outcomes["model"].eq("bdmtf_agent_replay")
    ].copy()
    outcome_labels = {
        "reply_count": "回复量",
        "active_authors": "活跃作者",
        "max_depth": "最大深度",
    }
    outcome_rows = [
        f"| {outcome_labels.get(row.outcome, row.outcome)} | "
        f"{row.observed_mean_effect:+.3f} | "
        f"{row.predicted_mean_effect:+.3f} | {row.mae:.3f} | "
        f"{100 * row.direction_accuracy:.1f}% |"
        for row in agent_outcomes.itertuples(index=False)
    ]
    agent = manifest["bdmtf_agent_replay"]
    zero_gain = manifest["bdmtf_vs_zero_effect_mae_improvement"]
    frozen_gain = manifest["bdmtf_vs_frozen_pretrend_mae_improvement"]
    return "\n".join(
        [
            "## 表 11：Lemmy 智能体级干预重放",
            "",
            (
                "**作用：** 检验 BDMTF 能否从真实干预前线程状态出发，逐条生成"
                "到达时间、作者、回复对象和深度，并预测未见锁帖/删帖响应；它不是"
                "用汇总响应回归替代模拟。"
            ),
            "",
            "| 模型 | 三项主结果 MAE [95% CI] | 方向准确率 | 校准区间覆盖率 |",
            "| --- | ---: | ---: | ---: |",
            *rows,
            "",
            "| 结果 | 真实平均效应 | 重放平均效应 | MAE | 方向准确率 |",
            "| --- | ---: | ---: | ---: | ---: |",
            *outcome_rows,
            "",
            (
                f"共使用 **{manifest['n_interventions']} 个干预**，按时间划分为 "
                f"{manifest['split_counts']['train']}/"
                f"{manifest['split_counts']['validation']}/"
                f"{manifest['split_counts']['test']}；最终对 "
                f"**{manifest['test_interventions']} 个时间外干预**各运行 "
                f"{manifest['test_simulation_seeds']} 个种子，生成 "
                f"**{manifest['simulated_event_rows']:,} 条智能体事件**。"
            ),
            "",
            (
                f"BDMTF 重放 MAE 为 **{agent['primary_mae']:.3f}**，方向准确率"
                f" **{100 * agent['direction_accuracy']:.1f}%**。相对零效应基线"
                f"改善 **{zero_gain['mean']:.3f}**"
                f" [{zero_gain['ci'][0]:.3f}, {zero_gain['ci'][1]:.3f}]，"
                f"相对冻结前趋势改善 **{frozen_gain['mean']:.3f}**"
                f" [{frozen_gain['ci'][0]:.3f}, {frozen_gain['ci'][1]:.3f}]；"
                "两者区间均大于 0。"
            ),
            "",
            (
                f"原始种子波动区间覆盖率为 "
                f"{100 * agent['raw_seed_interval_coverage']:.1f}%；只用 99 个"
                f"验证干预进行 conformal 校准后，测试覆盖率为 "
                f"**{100 * agent['interval_coverage']:.1f}%**。测试期结果未参与"
                "事件模型、参数选择或区间校准。"
            ),
            "",
            (
                "**论文用途：** 该结果把“真实审核造成结构变化”的自然实验证据"
                "与“BDMTF 能生成并预测该变化”的机制证据连接起来。结论限于"
                "回顾性 Lemmy 时间外验证，不能写成真人随机实验。"
            ),
        ]
    )


def _content_markdown(
    manifest: dict[str, Any],
    ranking: pd.DataFrame,
) -> str:
    exact = ranking[ranking["match_type"].eq("exact_url")]
    labels = {
        "lemmy_target_fitted_bdmtf": "Lemmy 目标数据拟合 BDMTF",
        "lemmy_structure_selected_bdmtf": "Lemmy 验证集结构适配 BDMTF",
        "lemmy_empirical_bootstrap": "Lemmy 经验重采样",
        "hackernews_zero_shot_bdmtf": "HN 零样本 BDMTF",
        "lemmy_branching_process": "Lemmy 分支过程",
    }
    rows = [
        f"| {labels.get(row.model, row.model)} | "
        f"{row.mean_normalized_mae:.3f} "
        f"[{row.ci_low:.3f}, {row.ci_high:.3f}] |"
        for row in exact.itertuples(index=False)
    ]
    matching = manifest["matching"]
    post = manifest["post_level_result"]
    model = manifest["exact_url_model_result"]
    return "\n".join(
        [
            "## 表 12：Lemmy 内容配对跨平台验证",
            "",
            (
                "**作用：** 在 HN 与 Lemmy 发布相同 URL 的帖子之间比较真实"
                "讨论树，以控制可观测内容身份，再检验有限 Lemmy 数据能否修正"
                "HN 参数直接迁移造成的平台失配。"
            ),
            "",
            "| 模型 | 相同 URL 完整树 normalized MAE [95% CI] |",
            "| --- | ---: |",
            *rows,
            "",
            (
                f"建立 **{matching['exact_post_pairs']} 个一对一相同 URL 帖子对**"
                f"（{matching['exact_unique_urls']} 个唯一 URL），其中 "
                f"**{matching['exact_structural_pairs']} 对**在两平台均有完整"
                f"回复树；另有 {matching['semantic_structural_pairs']} 对高阈值"
                "语义事件匹配，结果单独保存。"
            ),
            "",
            (
                f"相同内容的跨平台评论量 Spearman 相关为 "
                f"**{post['spearman_comment_count']:.3f}**"
                f"（p={post['spearman_p_value']:.3f}），说明仅有内容身份不足以"
                "解释互动规模，平台机制适配是必要步骤。"
            ),
            "",
            (
                f"在 44 个相同 URL 完整树上，HN 零样本误差为 "
                f"**{model['hackernews_zero_shot_mean_normalized_mae']:.3f}**，"
                f"经历史 Lemmy 验证集选择结构参数后降至 "
                f"**{model['lemmy_structure_selected_mean_normalized_mae']:.3f}**；"
                f"配对改善为 "
                f"**{model['structure_selected_improvement_over_zero_shot']:.3f}**"
                f" [{model['structure_selected_improvement_ci'][0]:.3f}, "
                f"{model['structure_selected_improvement_ci'][1]:.3f}]。"
            ),
            "",
            (
                "只拟合 Lemmy 目标分布而不选择结构参数时，误差为 "
                f"{model['lemmy_target_fitted_mean_normalized_mae']:.3f}，"
                f"相对零样本改善 "
                f"{model['target_fitted_improvement_over_zero_shot']:.3f}"
                f" [{model['target_fitted_improvement_ci'][0]:.3f}, "
                f"{model['target_fitted_improvement_ci'][1]:.3f}]。两层适配结果"
                "分别报告，不用测试集在二者之间重新选优。"
            ),
            "",
            (
                "**论文用途：** 支持“BDMTF 可用有限目标平台数据适配到 Lemmy”，"
                "同时否定不经适配的零样本普适性。结论仅覆盖线程式 HN/Lemmy，"
                "不能推广到非线程平台。"
            ),
        ]
    )


def _write_latex(
    agent: dict[str, Any],
    agent_ranking: pd.DataFrame,
    content: dict[str, Any],
    content_ranking: pd.DataFrame,
) -> None:
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    agent_labels = {
        "bdmtf_agent_replay": "BDMTF agent replay",
        "zero_effect": "Zero effect",
        "frozen_pretrend": "Frozen pretrend",
    }
    agent_rows = []
    for row in agent_ranking.itertuples(index=False):
        coverage = (
            f"{100 * row.interval_coverage:.1f}\\%"
            if row.model == "bdmtf_agent_replay"
            else "--"
        )
        agent_rows.append(
            f"{agent_labels.get(row.model, row.model)} & "
            f"{row.primary_mae:.3f} "
            f"[{row.primary_mae_ci_low:.3f}, {row.primary_mae_ci_high:.3f}] & "
            f"{100 * row.direction_accuracy:.1f}\\% & {coverage} \\\\"
        )
    agent_tex = "\n".join(
        [
            "\\refstepcounter{table}",
            "\\begin{center}",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{7pt}",
            "\\begin{tabular}{lccc}",
            "\\toprule",
            "Model & Primary MAE [95\\% CI] & Direction & Coverage \\\\",
            "\\midrule",
            *agent_rows,
            "\\bottomrule",
            "\\end{tabular}",
            "\\label{tab:lemmy-agent-replay}",
            "\\par\\smallskip",
            "\\begin{minipage}{\\columnwidth}",
            (
                "\\small\\textbf{Table \\thetable:} Agent-level replay of 103 "
                "chronological held-out Lemmy interventions. Coverage for "
                "BDMTF uses validation-only conformal calibration."
            ),
            "\\end{minipage}",
            "\\end{center}",
            "",
        ]
    )
    (LATEX_DIR / "table_lemmy_agent_replay.tex").write_text(
        agent_tex,
        encoding="utf-8",
    )

    content_labels = {
        "lemmy_target_fitted_bdmtf": "Lemmy target-fitted BDMTF",
        "lemmy_structure_selected_bdmtf": "Lemmy structure-selected BDMTF",
        "lemmy_empirical_bootstrap": "Empirical bootstrap",
        "hackernews_zero_shot_bdmtf": "HN zero-shot BDMTF",
        "lemmy_branching_process": "Branching process",
    }
    exact = content_ranking[content_ranking["match_type"].eq("exact_url")]
    content_rows = [
        f"{content_labels.get(row.model, row.model)} & "
        f"{row.mean_normalized_mae:.3f} "
        f"[{row.ci_low:.3f}, {row.ci_high:.3f}] \\\\"
        for row in exact.itertuples(index=False)
    ]
    matching = content["matching"]
    content_tex = "\n".join(
        [
            "\\begin{table*}[t]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{lc}",
            "\\toprule",
            "Model & Normalized MAE [95\\% CI] \\\\",
            "\\midrule",
            *content_rows,
            "\\bottomrule",
            "\\end{tabular}",
            (
                "\\caption{Content-matched HN--Lemmy validation on "
                f"{matching['exact_structural_pairs']} exact-URL pairs with "
                "complete reply trees.}"
            ),
            "\\label{tab:lemmy-content-matched}",
            "\\end{table*}",
            "",
        ]
    )
    (LATEX_DIR / "table_lemmy_content_matched.tex").write_text(
        content_tex,
        encoding="utf-8",
    )


def main() -> None:
    agent_path = (
        ROOT / "artifacts" / "interventions" / "agent_replay"
        / "agent_replay_manifest.json"
    )
    content_path = (
        ROOT / "artifacts" / "external_validation"
        / "lemmy_content_matched" / "content_matched_manifest.json"
    )
    agent = _read_json(agent_path)
    content = _read_json(content_path)
    agent_ranking = pd.read_csv(agent_path.parent / "model_ranking.csv")
    agent_outcomes = pd.read_csv(
        agent_path.parent / "comparison_by_model_outcome.csv"
    )
    content_ranking = pd.read_csv(content_path.parent / "model_ranking.csv")

    block = "\n\n".join(
        [
            _agent_markdown(agent, agent_ranking, agent_outcomes),
            _content_markdown(content, content_ranking),
            f"_自动更新时间：{date.today().isoformat()}_",
        ]
    )
    _replace_block(PAPER_TABLES, START, END, block)
    _write_latex(agent, agent_ranking, content, content_ranking)

    review_block = "\n".join(
        [
            "## Lemmy 新增证据响应",
            "",
            "| 审稿问题 | 新增证据 | 当前判定 |",
            "| --- | --- | --- |",
            (
                f"| 模拟器内部循环论证 | 103 个时间外真实干预的智能体级"
                f"事件重放，生成 {agent['simulated_event_rows']:,} 条事件；"
                f"相对零效应 MAE 改善 "
                f"{agent['bdmtf_vs_zero_effect_mae_improvement']['mean']:.3f} "
                f"[{agent['bdmtf_vs_zero_effect_mae_improvement']['ci'][0]:.3f}, "
                f"{agent['bdmtf_vs_zero_effect_mae_improvement']['ci'][1]:.3f}] | "
                "已在回顾性 Lemmy 时间外验证范围内解决；真人随机验证仍待外部审批 |"
            ),
            (
                f"| 跨平台证据不足、平台间内容不同 | "
                f"{content['matching']['exact_post_pairs']} 个 HN--Lemmy 相同 URL "
                f"帖子对，含 {content['matching']['exact_structural_pairs']} 个完整树对；"
                f"验证集结构适配相对 HN 零样本改善 "
                f"{content['exact_url_model_result']['structure_selected_improvement_over_zero_shot']:.3f} "
                f"[{content['exact_url_model_result']['structure_selected_improvement_ci'][0]:.3f}, "
                f"{content['exact_url_model_result']['structure_selected_improvement_ci'][1]:.3f}] | "
                "已支持线程式平台的有限数据适配；不宣称零样本或非线程平台普适性 |"
            ),
            (
                "| 缺少真实审核响应预测 | 真实 Lemmy 自然实验、聚合响应适配器和"
                "逐事件智能体重放三层结果均已保存 | Lemmy 锁帖/删帖范围内解决；"
                "TBBT 和真人 RCT 单列待完成 |"
            ),
        ]
    )
    _replace_block(
        REVIEW_MATRIX,
        MATRIX_START,
        MATRIX_END,
        review_block,
    )
    changelog_block = "\n".join(
        [
            "## 2026-07-29 Lemmy agent replay and content matching",
            "",
            "- Added event-generating Lemmy lock/removal replay with chronological train/validation/test isolation.",
            "- Added validation-only conformal prediction intervals and clustered bootstrap comparisons.",
            "- Added one-to-one exact-URL and separately reported semantic HN--Lemmy matching.",
            "- Added bounded Lemmy target-platform adaptation on held-out content-matched reply trees.",
            "- Added CLI commands, paper-ready Markdown tables, LaTeX tables, reports, manifests, and focused tests.",
        ]
    )
    _replace_block(
        CHANGELOG,
        CHANGELOG_START,
        CHANGELOG_END,
        changelog_block,
    )

    latest_payload = _read_json(LATEST_JSON) if LATEST_JSON.is_file() else {}
    expansion_summary_path = (
        ROOT / "artifacts" / "external_validation" / "reddit_expansion"
        / "validation_summary.json"
    )
    expansion_summary = _read_json(expansion_summary_path)
    latest_payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    latest_payload["lemmy_agent_replay"] = agent
    latest_payload["lemmy_content_matched_validation"] = content
    latest_payload["reddit_expansion"] = {
        "status": expansion_summary["status"],
        "n_communities": expansion_summary["n_communities"],
        "n_cascades": expansion_summary["n_cascades"],
        "n_comments": expansion_summary["n_comments"],
        "target_adapted_learned_bdmtf_median": (
            expansion_summary["protocols"]["target_adapted"][
                "learned_bdmtf_median_normalized_wasserstein"
            ]
        ),
        "loco_learned_bdmtf_median": (
            expansion_summary["protocols"]["leave_one_community_out"][
                "learned_bdmtf_median_normalized_wasserstein"
            ]
        ),
    }
    latest_payload["external_and_intervention_data"]["exact_url_matches"] = (
        content["matching"]["exact_post_pairs"]
    )
    latest_payload["external_and_intervention_data"][
        "exact_url_complete_tree_matches"
    ] = content["matching"]["exact_structural_pairs"]
    latest_payload["external_and_intervention_data"]["semantic_event_matches"] = (
        content["matching"]["semantic_structural_pairs"]
    )
    latest_payload["removed_from_plan"] = [
        "future_eight_week_prospective_collection"
    ]
    latest_payload["remaining_external_experiments"] = [
        "tbbt_confirmatory_controls_and_intervention_fidelity",
        "voat_expansion",
        "frozen_agent_intents_1575",
        "ethics_approved_human_rct",
    ]
    latest_payload["pending_experiments"] = [
        {
            "id": "tbbt_confirmatory_controls_and_intervention_fidelity",
            "status": "needs_qualified_controls",
        },
        {
            "id": "voat_expansion",
            "status": "needs_auditable_thread_and_intervention_data",
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
    latest_payload["latest_one_sentence"] = (
        "最新结果：原论文 9000/9000 次运行完整复现；确认性联合机制为 "
        "3.497x 评论量和 -12.424 叶深，24/24 局部敏感场景支持；"
        "24 个 Reddit 社区共 12,085 个完整级联；Lemmy 503 对真实干预"
        "通过门槛，103 个时间外智能体重放方向准确率 73.1%；HN--Lemmy "
        "形成 268 个相同 URL 对和 44 个完整树对，有限平台适配相对零样本"
        "误差改善 0.114 [0.034, 0.189]。"
    )
    new_source_paths = [
        agent_path,
        content_path,
        expansion_summary_path,
    ]
    new_relative_paths = {
        path.relative_to(ROOT).as_posix() for path in new_source_paths
    }
    existing_sources = [
        record
        for record in latest_payload.get("sources", [])
        if str(record.get("path", "")) not in new_relative_paths
    ]
    latest_payload["sources"] = existing_sources + [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(path),
        }
        for path in new_source_paths
    ]
    LATEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    LATEST_JSON.write_text(
        json.dumps(latest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LATEST_STATUS.write_text(
        _latest_markdown(latest_payload),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "paper_tables": str(PAPER_TABLES),
                "latest_status": str(LATEST_STATUS),
                "latest_json": str(LATEST_JSON),
                "review_matrix": str(REVIEW_MATRIX),
                "changelog": str(CHANGELOG),
                "latex_tables": [
                    str(LATEX_DIR / "table_lemmy_agent_replay.tex"),
                    str(LATEX_DIR / "table_lemmy_content_matched.tex"),
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

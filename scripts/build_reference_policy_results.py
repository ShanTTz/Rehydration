"""Publish completed strengthening results without editing archived manuscripts."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "manuscript/iclr2027_overleaf_package_strengthened_20260903"
ART = ROOT / "artifacts/reviewer_validation/reference_policy_strengthening_20260903_v2"
LABELS = {"openai": "OpenAI", "deepseek": "DeepSeek", "qwen": "Qwen",
          "original": "Multiplicative", "additive": "Additive", "saturating": "Saturating"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def marked_update(path, name, body):
    start, end = f"<!-- {name}_START -->", f"<!-- {name}_END -->"
    text = path.read_text(encoding="utf-8")
    block = f"{start}\n\n{body.rstrip()}\n\n{end}"
    if start in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before + block + after
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def main():
    result = json.loads((ART / "result_manifest.json").read_text(encoding="utf-8"))
    if result["status"] != "complete" or result["runs"] != 3800:
        raise ValueError("Do not publish partial results")
    for name, value in result["artifacts"].items():
        if digest(ART / name) != value:
            raise ValueError(f"Changed result: {name}")
    summary = pd.read_csv(ART / "effect_summary.csv")
    usage = pd.read_csv(ART / "pool_usage.csv")
    generated = PAPER / "generated"
    macros = ["% Generated from completed reference_policy_strengthening artifacts."]
    tables = {}
    md = []
    for number, panel, axis, title in [
        (30, "qref", "family", "同一政策四格下的三模型缓存参考池比较"),
        (31, "activation", "form", "训练期门槛通过率匹配后的激活函数比较"),
    ]:
        group = summary[summary.panel == panel]
        variants = ["openai", "deepseek", "qwen"] if panel == "qref" else ["original", "additive", "saturating"]
        columns = ["joint_volume_log", "interaction_volume_log", "joint_depth", "interaction_depth"]
        tex = [r"\begin{tabular}{lrrrr}", r"\toprule",
               r"Variant & Joint volume & Volume interaction & $\Delta$ leaf depth & Depth interaction \\", r"\midrule"]
        md += [f"## 表 {number}：{title}", "",
               "| 条件 | 联合评论量比 | 评论量交互比 | 联合叶深变化 | 叶深交互 |",
               "| --- | ---: | ---: | ---: | ---: |"]
        for variant in variants:
            rows = group[group[axis] == variant].set_index("metric")
            estimates = [f"{rows.loc[m, 'estimate']:.3f}" for m in columns]
            tex.append(LABELS[variant] + " & " + " & ".join(estimates) + r" \\")
            cells = [f"{rows.loc[m, 'estimate']:.3f} [{rows.loc[m, 'ci_low']:.3f}, {rows.loc[m, 'ci_high']:.3f}]" for m in columns]
            md.append("| " + LABELS[variant] + " | " + " | ".join(cells) + " |")
        tex += [r"\bottomrule", r"\end{tabular}"]
        table_name = f"table_reference_policy_{panel}.tex"
        (generated / table_name).write_text("\n".join(tex) + "\n", encoding="utf-8")
        tables[panel] = table_name
        prefix = "RefPool" if panel == "qref" else "ActForm"
        for suffix, metric in [("Volume", "interaction_volume_log"), ("Depth", "interaction_depth")]:
            values = group[group.metric == metric].estimate
            macros.append(f"\\newcommand{{\\{prefix}{suffix}Min}}{{{values.min():.3f}}}")
            macros.append(f"\\newcommand{{\\{prefix}{suffix}Max}}{{{values.max():.3f}}}")
        md += ["", "评论量比和交互比为配对 log 对比取均值后指数化；叶深使用加性差值。区间按社区分层、帖子聚类重采样，保留三个配对种子。", ""]
        if panel == "qref":
            md += ["**作用：** 将旧的‘语义耦合开关’诊断推进到同一 Behavioral×Ranking 四格对比。使用50帖、每帖10个缓存参与者、三模型族；经验背景、帖子和随机调度保持一致。", "",
                   "冻结的 reply/polarity 统计通过已有的有界映射影响行动率、冲突系数和极性门；表面文本不反馈到结构状态。每模型每提示词仅有一个真实缓存响应，本表不代表多次独立生成、全人口模型池或 prompt/temperature 稳健性。", ""]
        else:
            md += ["**作用：** 区分特定乘积函数与更一般的激活响应。加性模型用训练期均值对三个乘积通道作一阶展开；饱和模型对通道和施加 tanh。仅用25个训练帖匹配基线门槛通过率，不匹配实验评论量、深度或干预效应。", ""]
    max_exhaustion = float(usage.exhaustion_rate.max())
    share = usage[usage.panel == "qref"].api_realized_share
    macros.extend([
        f"\\newcommand{{\\ReferencePolicyRuns}}{{{result['runs']:,}}}",
        f"\\newcommand{{\\ReferencePoolUseMin}}{{{100 * share.min():.1f}}}",
        f"\\newcommand{{\\ReferencePoolUseMax}}{{{100 * share.max():.1f}}}",
        f"\\newcommand{{\\ReferenceMaxExhaustion}}{{{100 * max_exhaustion:.3f}}}",
    ])
    (generated / "reference_policy_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    md += ["## 表 32：补强实验的执行与负对照检查", "",
           "| 检查 | 结果 |", "| --- | --- |",
           "| 模型参考池四格重放 | 1,800 次 |", "| 激活函数四格重放 | 1,800 次 |",
           "| 排序断开负对照 | 200 次；排序对比精确为0 |",
           f"| 最大分组耗尽率 | {100 * max_exhaustion:.3f}% |",
           f"| API缓存意图实际回复占比 | {100 * share.min():.1f}%--{100 * share.max():.1f}% |",
           "| 新增API调用 | 0；复用已验证的1,500条响应 |",
           "| 激活率校准数据 | 25个训练帖，与50个比较帖无重叠 |", "",
           "排序断开检查将 Ranking 坐标及其下游读取统一固定为 best，只检验干预接口与随机耦合实现，不作为现实零效应证据。每次重放均保存事件级Parquet及SHA256。", ""]
    marked_update(ROOT / "PAPER_READY_RESULT_TABLES.md", "REFERENCE_POLICY_20260903", "\n".join(md))

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.4), constrained_layout=True)
    colors = ["#246B83", "#9C3D54", "#477A45"]
    for i, (panel, axis) in enumerate([("qref", "family"), ("activation", "form")]):
        for j, metric in enumerate(["interaction_volume_log", "interaction_depth"]):
            rows = summary[(summary.panel == panel) & (summary.metric == metric)]
            for k, row in enumerate(rows.itertuples()):
                axes[i, j].errorbar(row.estimate, k, xerr=[[row.estimate - row.ci_low], [row.ci_high - row.estimate]],
                                    fmt="o", color=colors[k], capsize=3)
            axes[i, j].set_yticks(range(len(rows)), [LABELS[v] for v in rows[axis]])
            axes[i, j].axvline(1 if j == 0 else 0, color=".6", linewidth=.8, linestyle="--")
            axes[i, j].set_xlabel("Volume interaction (ratio of ratios)" if j == 0 else "Leaf-depth interaction")
            axes[i, j].set_title("Cached reference pools" if i == 0 else "Activation functions", fontsize=11)
            axes[i, j].spines[["top", "right"]].set_visible(False)
    fig.savefig(generated / "reference_policy_interactions.pdf")
    fig.savefig(generated / "reference_policy_interactions.png", dpi=180)
    plt.close(fig)

    report = ["# 2026-09-03 实验补强与论文写作报告", "",
              "## 已完成", "", "\n".join(md), "## 论文写法", "",
              "- 新稿保留旧稿结构，集中陈述实验目标、结果和贡献；重复的范围说明集中在 Discussion。",
              "- 8.1%严格有效率保留在正文语义结果表，与最低可接受率共同呈现；删去正文及Discussion重复强调‘只有8.1%’的表述。",
              "- 既有深度、Lemmy基线、HN比较及全部置信区间保留，不根据结果是否有利决定是否报告。",
              "- 原始 hardening_20260825 论文包只读保留；本次修改在 strengthened_20260903 中。", "",
              "## 仍需要外部输入", "",
              "- 新的多次独立模型生成池：当前比较复用既有缓存，一个响应不能被重采样冒充多次独立API生成。",
              "- 三臂动态parent强基线及真人评分：冻结200任务尚待API与评审，不生成虚构评分。", "",
              "## 执行审计", "",
              "首次启动在写出首个结果前发现事件时间字段名不匹配，改为created_step并新增Parquet往返测试；失败目录保留，不进入统计。",
              f"有效结果清单位于 `{ART.relative_to(ROOT)}/result_manifest.json`。",
              "复现：`python scripts/run_reference_policy_strengthening.py --workers 5`，随后 `python scripts/build_reference_policy_results.py`。", ""]
    (ROOT / "EXPERIMENT_STRENGTHENING_REPORT_20260903.md").write_text("\n".join(report), encoding="utf-8")
    latest = ROOT / "artifacts/provenance/latest_status.json"
    state = json.loads(latest.read_text(encoding="utf-8-sig"))
    state["generated_at"] = datetime.now(timezone.utc).isoformat()
    state["reference_policy_strengthening"] = {**result, "result_path": str(ART.relative_to(ROOT)),
                                              "paper_path": str(PAPER.relative_to(ROOT))}
    paper_state = state.setdefault("paper", {})
    if "preserved_previous_release" not in paper_state:
        paper_state["preserved_previous_release"] = {
            key: paper_state.get(key) for key in ("official_tex", "official_pdf", "official_pdf_sha256")
        }
    paper_state.update({"status": "awaiting_compile", "official_tex": str((PAPER / "paper.tex").relative_to(ROOT)),
                        "official_pdf": str((PAPER / "paper.pdf").relative_to(ROOT)), "official_pdf_sha256": None})
    state["latest_one_sentence"] = "50帖三模型缓存参考池与激活函数补强共3,800次重放完成；关键指标完整保留，强语义基线和独立新生成池仍待外部输入。"
    state["pending_experiments"] = [
        {"id": "live_semantic_ceiling", "status": "tasks_ready_api_and_human_ratings_pending"},
        {"id": "independent_regenerated_reference_catalogs", "status": "not_run_requires_new_api_generations"},
        {"id": "human_rct", "status": "blocked_on_ethics_and_recruitment"},
    ]
    state["iclr2027_review_hardening"]["qref_sensitivity"]["scope"] = "Semantic-linked versus strict diagnostic, not the same-policy factorial; see reference_policy_strengthening for cached factorial comparisons."
    latest.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    marked_update(ROOT / "LATEST_STATUS.md", "REFERENCE_POLICY_20260903",
                  "## 2026-09-03 最新补强\n\n50帖、3个种子：三模型缓存参考池四格1,800次、三种激活函数四格1,800次、排序断开检查200次，共3,800次。全部完成；结果见PAPER_READY_RESULT_TABLES表30--32。\n\n"
                  "本次完成的是缓存leader参考分布的主四格对比，不是全人口独立多次生成池。旧表26仍是语义耦合开关诊断。三臂强语义基线仍待API和真人评分。\n\n"
                  "当前论文：`manuscript/iclr2027_overleaf_package_strengthened_20260903/paper.pdf`。")
    status_path = ROOT / "LATEST_STATUS.md"
    lines = status_path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.startswith("> 最新结果："):
            lines[i] = "> 最新结果：50帖上的三模型缓存参考池四格、激活函数形式与排序断开检查共 **3,800次重放已完成**；结果见表30--32，新稿位于 `manuscript/iclr2027_overleaf_package_strengthened_20260903/`。"
            break
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    publish = {"source_manifest": digest(ART / "result_manifest.json"), "publisher_sha256": digest(Path(__file__)),
               "tables": tables, "generated": {p.name: digest(p) for p in generated.glob("*reference_policy*")}}
    (ART / "paper_publication_manifest.json").write_text(json.dumps(publish, indent=2), encoding="utf-8")
    print(json.dumps({"published": result["runs"], "paper": str(PAPER)}, indent=2))


if __name__ == "__main__":
    main()

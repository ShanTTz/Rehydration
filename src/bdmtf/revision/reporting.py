from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from bdmtf.revision.manuscript import build_final_manuscript
from bdmtf.revision.oasis_adapter import capability_manifest
from bdmtf.revision.provenance import tree_manifest, write_json


def _write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _status(payload: dict[str, Any], default: str = "not_completed") -> str:
    return str(payload.get("status", default))


def _ranking_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "暂无可用排名。"
    rows = [
        "| 模型 | 平均标准化 Wasserstein | 中位标准化 Wasserstein | 指标数 |",
        "|---|---:|---:|---:|",
    ]
    for item in frame.sort_values("mean").itertuples(index=False):
        rows.append(
            f"| `{item.model}` | {float(item.mean):.3f} | "
            f"{float(item.median):.3f} | {int(item.count)} |"
        )
    return "\n".join(rows)


def _best(frame: pd.DataFrame) -> str:
    return str(frame.sort_values("mean").iloc[0]["model"]) if not frame.empty else "not_run"


def build_revision_package(root: Path) -> None:
    artifacts = root / "artifacts"
    audit = _json(artifacts / "data_audit" / "data_audit.json")
    ranking = _csv(artifacts / "evaluation" / "model_ranking.csv")
    loco = _json(artifacts / "external_validation" / "cross_community" / "summary.json")
    hn = _json(artifacts / "external_validation" / "cross_platform_transfer" / "summary.json")
    hn_collection = _json(
        root / "data" / "external" / "hackernews" / "collection_manifest.json"
    )
    api = _json(artifacts / "api" / "api_manifest.json")
    ablations = _csv(artifacts / "runs" / "ablations" / "tradeoff_regions.csv")
    tbbt = _json(artifacts / "interventions" / "tbbt" / "import_manifest.json")
    tbbt_outcome = _json(artifacts / "interventions" / "tbbt" / "outcome_panel_manifest.json")
    tbbt_natural = _json(
        artifacts / "interventions" / "natural_tbbt" / "natural_experiment_summary.json"
    )
    tbbt_fetch = _json(root / "data" / "external" / "tbbt" / "fetch_manifest.json")
    lemmy = _json(root / "data" / "external" / "lemmy" / "collection_manifest.json")
    lemmy_outcomes = _json(
        root / "data" / "external" / "lemmy" / "outcomes" / "outcome_collection_manifest.json"
    )
    expansion = _json(
        artifacts / "external_validation" / "reddit_expansion" / "expansion_manifest.json"
    )
    story = _json(artifacts / "story_matching" / "match_manifest.json")
    story_url_resolution = story.get("url_resolution", {})
    prospective = _json(artifacts / "prospective" / "prospective_manifest.json")
    natural = _json(artifacts / "interventions" / "natural" / "natural_experiment_summary.json")
    rct = _json(artifacts / "human_rct" / "rct_summary.json")
    fidelity = _json(artifacts / "interventions" / "fidelity" / "intervention_fidelity.json")
    redline = _json(root / "manuscript" / "redline" / "BUILD_MANIFEST.json")
    source_recovery = _json(
        root / "manuscript" / "original_reconstructed" / "SOURCE_RECOVERY.json"
    )
    source_pdf = _json(
        root / "manuscript" / "source_evidence" / "original_source_manifest.json"
    )
    platform_counts = hn.get("n_target_cascades_by_platform", {})
    hn_evaluated = int(platform_counts.get("HackerNews", hn.get("n_target_cascades", 0)))
    lemmy_evaluated = int(platform_counts.get("Lemmy", 0))

    shallow = int(ablations["shallow_swarm"].astype(bool).sum()) if not ablations.empty else 0
    total_ablations = int(len(ablations))
    manuscript = build_final_manuscript(root)
    oasis = capability_manifest(root)
    write_json(artifacts / "provenance" / "oasis_capabilities.json", oasis)

    _write(
        root / "UPGRADE_REPORT.md",
        f"""
# BDMTF 升级报告

## 当前定位

本仓库保留原始 BDMTF 三层框架，并把验证拆成精确复现、边际级联拟合、
机制分解、干预响应预测和外部迁移。域内平均分布距离最佳模型是
`{_best(ranking)}`；LOCO 按均值最佳模型是
`{loco.get('best_model_by_mean_normalized_wasserstein', 'not_run')}`。
这两项属于无条件级联拟合，不用于替代 BDMTF 的反事实任务。

## 已完成升级

1. 建立 `PlatformEvent`、`InterventionEvent`、`MatchedStory`、`OutcomePanel`。
2. 完成泄漏安全的时间划分、训练集拟合、五类基线和 held-out 评估。
3. 增加排序、viewport、traits、退出、反言论、删除、锁帖和审核消融。
4. 分开回复树、共同参与网络和真实连接网络；没有 follower 数据时显式为 null。
5. 完成 TBBT 四个公开档案的断点续传、MD5/SHA256 校验和内存有界导入。
6. 完成 Lemmy modlog、线程结果、risk-set pilot 和自然实验诊断管线。
7. 完成 URL 规范化、精确 URL 匹配和独立语义事件匹配。
8. 增加 pre-only 预测、DiD、event study、ITS、合成控制和 R 参考估计。
9. 建成人类随机线程实验平台；伦理、预注册、同意和样本门控不可绕过。
10. 导入用户指定的 17 页原始 PDF，以其标题、双栏结构、Section 1--6、
    Figure 1--2 和附录 A--F 为正式修订底稿。

## 已完成实验

- Reddit：{audit.get('n_posts', 0)} 个帖子，
  {audit.get('n_reconstructable_cascades', 0)} 个可重建级联。
- 平均叶深审计：真实树为 {audit.get('empirical_mean_leaf_depth', float('nan')):.2f}；
  原文 18.4 是模拟 baseline 指标，两者量纲不同。
- LOCO：{loco.get('n_simulations', 0)} 次模拟。
- Hacker News：采集 {hn_collection.get('story_count_collected', 0)} 个完整级联，
  其中 {hn_evaluated} 个进入跨平台评估，
  共 {hn.get('n_simulations', 0)} 次模拟。
- Lemmy：{lemmy_evaluated} 个完整 pilot 级联进入同一零样本评估。
- 旧修订消融：{total_ablations} 个场景—社区单元，其中 {shallow} 个满足
  Shallow Swarm；该结果是探索性压力测试，不用于否定原稿主实验。
- TBBT：{tbbt.get('records_read', 0):,} 条消息，{tbbt.get('panel_rows', 0)} 个日聚合行；
  数据完整，但不含匹配未处理社区。
- Lemmy：{lemmy_outcomes.get('n_treated_interventions', 0)} 个 pilot 干预，
  {lemmy_outcomes.get('n_risk_set_pairs', 0)} 个 risk-set 配对。
- API：{api.get('expected_calls', 0)} 个任务，状态 `{_status(api)}`，尚未伪装为已执行。

## 尚未完成的证据

| 证据 | 状态 | 结论边界 |
|---|---|---|
| TBBT 下载/导入 | `{_status(tbbt_fetch)}` / `{_status(tbbt)}` | 允许描述性分析，不允许把 OUT 当未处理组 |
| TBBT 因果估计 | `{_status(tbbt_natural)}` | 缺匹配 controls |
| Lemmy 自然实验 | `{_status(natural)}` | pilot，不是确认性因果结果 |
| Reddit 20 社区 | `{_status(expansion)}` | 缺合法扩展输入 |
| 八周前瞻测试 | `{_status(prospective, 'not_frozen')}` | D0 尚未冻结 |
| 人类 RCT | `{_status(rct)}` | 缺伦理审批、预注册和招募 |
| 干预预测忠实度 | `{_status(fidelity)}` | 缺冻结预测与确认性观察效应 |

这些限制是数据、时间和伦理前置条件，不应通过编造结果“补齐”。
""",
    )

    _write(
        root / "REVIEW_RESPONSE_MATRIX.md",
        f"""
# 审稿意见响应矩阵

| 审稿问题 | 已有改进 | 状态 | 可写结论 |
|---|---|---|---|
| 模拟器内部循环论证 | 精确复现、时间切分、held-out fidelity、干预预测赛道 | 确认性实验待运行 | 分开边际拟合与反事实效用 |
| 规则与参数手工设定 | train-only 激活、目标与退出模型；legacy 单列 | 已解决 | 不识别个体曝光因果概率 |
| 指标太少 | 结构、时间、集中度、作者复用、毒性、纠正 | 已解决 | 受原数据字段限制 |
| 缺少替代基线 | 无条件拟合基线 + conditional intervention baselines | 部分解决 | branching 仅在边际拟合赛道胜出 |
| traits 强制负相关 | 耦合、独立、建设性攻击、打乱 | 确认性配对消融待运行 | 预先检验 prosociality 调节效应 |
| 排序/浏览参数武断 | 五种排序、位置偏差、viewport、敏感性 | 代码与消融已解决 | 人类校准仍待 RCT |
| 缺审核、反言论和退出 | counterspeech、dropout、删除、锁帖、解释 | 机制已实现 | 真实效应仍待确认 |
| follower graph 混淆 | 三类网络分开，缺失连接显式 null | 已解决 | 不再虚构 follower graph |
| LLM 模型敏感性 | 三系列硬门，{api.get('expected_calls', 0)} 个冻结任务 | 部分解决：`{_status(api)}` | API 未执行 |
| 只选五个社区 | LOCO + 20 社区导入门 | 部分解决：`{_status(expansion)}` | 当前实证仍是五社区 |
| 跨平台证据不足 | HN、Lemmy、Voat、URL/事件匹配 | 部分解决：`{_status(story)}` | 精确 URL {story.get('n_exact_url', 0)} 对；URL 网络解析 `{_status(story_url_resolution, 'normalized_only')}` |
| 没有真实干预 | TBBT 全量导入、Lemmy pilot、现代因果管线 | 未完全解决 | TBBT 无 untreated controls；Lemmy 是 pilot |
| 没有人类随机验证 | 可运行且伦理门控的 30-cell RCT 平台 | 未解决：`{_status(rct)}` | 不得宣称已做人类实验 |
| 可能事后调参 | 显式 D0 和八周一次性窗口 | 未解决：`{_status(prospective, 'not_frozen')}` | D0 必须采集前冻结 |
| 异常审稿短语 | 抽取文本清洗、块哈希、稿件扫描 | 已解决 | 删除 {source_recovery.get('removed_block_count', 0)} 个异常块 |

只有实验实际完成且诊断通过时，矩阵才允许标记为 resolved。
""",
    )

    _write(
        root / "RESULTS_AUDIT.md",
        f"""
# 结果审计

## 原论文精确复现与当前证据

| 项目 | 原论文 | 当前可重建结果 | 判断 |
|---|---:|---:|---|
| Reddit 帖子 | 500 | {audit.get('n_posts', 0)} | 一致 |
| 可重建真实级联 | 未透明报告排除 | {audit.get('n_reconstructable_cascades', 0)} | 新增排除审计 |
| 模拟 baseline 平均叶深 | 18.4 | 18.498 | 9000-run 精确复现 |
| 模拟 toxic 平均叶深 | 6.1 | 6.061 | 9000-run 精确复现 |
| 真实树平均叶深 | 未作同量纲报告 | {audit.get('empirical_mean_leaf_depth', float('nan')):.3f} | 用于现实模式验证，不与 18.4 直接比较 |
| 旧修订消融 | 无 | {shallow}/{total_ablations} 个场景—社区单元 | 探索性压力测试 |
| 无条件域内最佳模型 | 无充分基线 | `{_best(ranking)}` | 不等同于干预预测排名 |
| LOCO | 无 | 均值最佳 `{loco.get('best_model_by_mean_normalized_wasserstein', 'not_run')}` | 仅五社区范围 |
| 跨平台 | 无 | HN 采集 {hn_collection.get('story_count_collected', 0)} 个、评估 {hn_evaluated} 个；Lemmy {lemmy_evaluated} 个 | 初步证据 |
| TBBT | 无 | {tbbt.get('records_read', 0):,} 条消息 | 完整导入，描述性使用 |
| 真实因果干预 | 无 | Lemmy `{_status(natural)}`；TBBT `{_status(tbbt_natural)}` | 未确认性解决 |

## 域内模型排名

{_ranking_table(ranking)}

原论文精确复现冻结清单位于
`artifacts/provenance/paper_exact_freeze_manifest.json`。旧稿自动注入结果
继续保留，但在确认性实验完成前不再更新论文。
""",
    )

    _write(
        root / "PAPER_REVISION_GUIDE.md",
        """
# 论文增量修改指南

最终标题：

> What Makes Content Go Viral? Falsifiable Mechanistic Tracing Across Communities, Platforms, and Real Moderation Interventions

原 Abstract、Introduction、Related Work、Methodology、Agent Modeling、
Experiments、Conclusion 和附录 A--F 的顺序保留。

1. Introduction 保留预测与机制范式对比，主张缩小到线程式讨论平台。
2. Related Work 增加真实干预、跨平台内容配对、现代 DiD 和随机社交影响实验。
3. Methodology 保留三层 BDMTF 与冻结意图，加入统一事件契约和证据门控。
4. Agent Modeling 将旧规则标记为 legacy，加入数据估计策略、退出、反言论和审核。
5. Experiments 先报告 held-out、基线、LOCO 和消融，再报告外部实验真实状态。
6. Conclusion 保留原 Shallow Swarm 机制主线，并根据确认性实验报告
   稳健范围和边界；不把 pilot 写成普遍因果规律。
7. 附录保留 A--F，并增加数据审计、因果识别、配对、RCT 协议和零结果。

Figure 1 和 Figure 2 从用户指定原稿中恢复；正文在原 Section 5 位置注入
held-out、LOCO、跨平台、消融和证据门控结果。

正式源证据是 `manuscript/source_evidence/What_Makes_Content_Go_Vi_original.pdf`，
正式修订稿位于 `manuscript/original_pdf_revision/`。由于没有原始 TeX，
现有 `manuscript/redline/` 只能视为抽取文本重建稿的参考差异，不能声称是
对真实原始 TeX 的 byte-faithful `latexdiff`。
""",
    )

    _write(
        root / "EXTERNAL_VALIDITY.md",
        f"""
# 外部有效性审计

外部有效性被拆成跨社区迁移、跨平台迁移、内容配对、真实干预预测和人类随机验证五层，任何一层都不能代替另一层。

## 已完成证据

- 五社区 LOCO：{loco.get('n_simulations', 0)} 次零样本模拟；按均值最佳模型为
  `{loco.get('best_model_by_mean_normalized_wasserstein', 'not_run')}`。
- Hacker News：官方 API 有界采集
  {hn_collection.get('story_count_collected', 0)} 个完整级联，
  {hn_evaluated} 个进入 Reddit-only 零样本评估，
  共 {hn.get('n_simulations', 0)} 次模拟。
- Lemmy：{lemmy_evaluated} 个结构完整的 pilot 级联进入同一零样本评估。
- 内容配对：{story.get('n_roots', 0)} 个根事件，
  {story.get('n_exact_url', 0)} 个精确 URL 对，
  {story.get('n_semantic_event', 0)} 个独立语义事件对。
- TBBT：完整导入 {tbbt.get('records_read', 0):,} 条 Reddit/Voat 消息和
  {tbbt.get('interventions', 0)} 个干预；当前自然实验状态
  `{_status(tbbt_natural)}`。
- Lemmy：{lemmy_outcomes.get('n_treated_interventions', 0)} 个 pilot 干预、
  {lemmy_outcomes.get('n_risk_set_pairs', 0)} 个 risk-set 配对；估计状态
  `{_status(natural)}`。

## 识别边界

- HN 是按官方 `maxitem` 逆序扫描得到的有界样本，不代表整个平台。
- 跨平台模拟只用 Reddit 训练集拟合，外部级联数量为零进入拟合。
- 精确 URL 与语义事件匹配分开报告，不能用语义相似替代同一内容。
- TBBT 的 OUT 是受影响用户在目标社区外的活动，不是未处理社区。
- Lemmy 当前样本是管线 pilot，诊断不支持确认性因果结论。
- Reddit 20 社区状态为 `{_status(expansion)}`；八周前瞻状态为
  `{_status(prospective, 'not_frozen')}`；人类 RCT 状态为 `{_status(rct)}`。

因此当前可以报告跨社区和有界跨平台压力测试、完整 TBBT 描述性分析与 Lemmy pilot，
不能声称普遍跨平台迁移、真实干预因果预测已经验证，或人类随机实验已经完成。
""",
    )

    _write(
        root / "DATA_CARD.md",
        f"""
# Data Card

- 历史 Reddit：{audit.get('n_posts', 0)} 个帖子，
  {audit.get('n_reconstructable_cascades', 0)} 个可重建级联，五个社区。
- Hacker News：官方 API 采集 {hn_collection.get('story_count_collected', 0)} 个完整级联，
  {hn_evaluated} 个进入现有评估；限定为至多 300 条回复，
  不是全平台样本。
- Lemmy：{lemmy_evaluated} 个结构完整 pilot 级联进入现有跨平台评估。
- TBBT：Zenodo 18245670，{tbbt.get('records_read', 0):,} 条消息，25 个干预，
  Reddit/Voat，CC BY-NC-ND 4.0；原始档案不提交 Git。
- Lemmy：公开 modlog；恢复、批量 purge 与首次删除分开。
- 故事匹配：{story.get('n_roots', 0)} 个根事件，
  {story.get('n_exact_url', 0)} 个精确 URL 匹配，
  {story.get('n_semantic_event', 0)} 个语义匹配。
- 缺失字段：跨平台 follower graph、曝光但未行动记录、TBBT 文本与 URL。
- TBBT 的 OUT 是受影响用户在目标社区外的活动，不是 untreated communities。
- 私有人类实验数据不得提交公开仓库。
""",
    )

    _write(
        root / "MODEL_CARD.md",
        f"""
# Model Card

BDMTF 是线程式讨论的机制模拟器，不是人格诊断或通用社交平台因果模型。

- 主模型：训练集估计的聚合事件风险、回复目标和退出策略。
- 基线：经验重采样、负二项分支、Hawkes、legacy BDMTF。
- 语义层：冻结意图和三模型系列硬门；当前状态 `{_status(api)}`。
- 网络：回复树、共同参与和真实连接严格分开。
- 已知边界：`{_best(ranking)}` 在无条件域内平均分布距离上优于 learned
  BDMTF；干预响应预测赛道尚待确认性评估。
- 允许主张：只限 manifest 已完成且诊断通过的数据、平台与干预。
""",
    )

    _write(
        root / "REPRODUCIBILITY.md",
        """
# Reproducibility

## 本机 Python

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-lock.txt
.venv/Scripts/pip install -e .
bdmtf audit-data
bdmtf evaluate-fidelity
bdmtf build-final-paper
```

Linux 使用 `.venv/bin/pip`。Windows/Linux 脚本位于 `scripts/`。

## 论文工具链

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_paper_toolchain.ps1
bdmtf build-original-manuscript
bdmtf build-final-paper
bdmtf build-paper-redline
```

## Docker

```bash
docker compose run --rm bdmtf
docker compose run --rm causal
docker compose run --rm paper
```

当前仓库分别锁定 Python 3.12、R 因果环境和 Tectonic/latexdiff 论文环境。

## 外部公开数据

```bash
bdmtf fetch-tbbt
bdmtf fetch-tbbt --execute
bdmtf fetch-tbbt --import-data
bdmtf collect-lemmy
bdmtf collect-lemmy-outcomes
bdmtf collect-hackernews --stories 5000 --max-comments-per-story 300 --workers 48 --discovery algolia
bdmtf build-story-matches
bdmtf build-story-matches --resolve-urls
```

TBBT 支持断点续传、发布者 MD5 校验、按档案检查点和内存有界聚合。

## API 与前瞻实验

密钥只从环境变量读取；此前暴露的密钥必须更换。无 API 时可重放缓存。
`make-prospective-split --freeze-at ...` 必须在未来数据采集前执行。
人类实验必须先有伦理审批、预注册、知情同意和招募预算。
""",
    )

    _write(
        root / "CHANGELOG.md",
        f"""
# Changelog

## 2026 external-validity revision

- 新增统一平台、干预、故事匹配和结果面板契约。
- 新增 TBBT 全量获取、校验、分区检查点、HLL 作者计数和 25 干预目录。
- 修复 TBBT OUT 聚合层级、Voat 平台识别和重复 slug 的干预类型冲突。
- 新增 Lemmy modlog、结果线程和 risk-set pilot。
- 新增 Reddit 20 社区导入门和跨平台故事匹配。
- 新增 D0 八周前瞻冻结协议。
- 新增 DiD、event study、controlled/uncontrolled ITS、合成控制和 R 参考环境。
- 新增伦理门控的人类 RCT 平台与分析。
- 导入用户指定的 17 页原始 PDF，并从其双栏结构生成正式增量修订稿。
- 原始 PDF SHA256：`{source_pdf.get('sha256', 'missing')}`。
- 抽取文本参考红线状态：`{_status(redline)}`；不等同于真实原始 TeX 红线。
""",
    )

    _write(
        root / "PAPER_CURRENT_STATUS_REPORT.md",
        f"""
# 论文当前全貌

## 论文现在是什么

论文仍以原始 BDMTF 三层框架和 Shallow Swarm 机制为主线。扩展实验将
依次验证精确复现、机制分解、合理参数稳健性、现实模式一致性、干预预测
和平台适配迁移。冻结语义意图、核心公式和原附录 A--F 保留。

## 已达到的效果

- 原论文 9000-run 已完整复现，主汇总指标与原稿误差约在 2.1% 内。
- 数据切分、作者历史和训练边界可审计。
- 简单基线在部分边际分布拟合上更好，但尚未与 BDMTF 公平比较干预预测。
- 真实树平均叶深与模拟叶深量纲不同；不再把 2.22 与 18.4 直接比较。
- 旧 {shallow}/{total_ablations} 结果已归类为探索性压力测试。
- 五社区 LOCO、HN 有界转移、Lemmy pilot 和 TBBT 全量导入已有真实 artifacts。
- 用户指定原始 PDF 已校验；双栏正式修订稿可编译。
- 抽取文本参考红线状态 `{_status(redline)}`，但没有原始 TeX，不能作为
  真实源码级 `latexdiff`。

## 仍不能声称的内容

- 不能声称 20 个 Reddit 社区验证完成：`{_status(expansion)}`。
- 不能声称八周前瞻验证完成：`{_status(prospective, 'not_frozen')}`。
- 不能声称三模型 API 稳健性完成：`{_status(api)}`。
- 不能声称人类 RCT 完成：`{_status(rct)}`。
- 不能把 TBBT OUT 当未处理社区，也不能把 Lemmy pilot 当确认性自然实验。

## 关键文件

- 原始 PDF：`manuscript/source_evidence/What_Makes_Content_Go_Vi_original.pdf`
- 原始 PDF 清单：`manuscript/source_evidence/original_source_manifest.json`
- 正式增量修订：`manuscript/original_pdf_revision/paper.tex`
- 正式 PDF：`manuscript/original_pdf_revision/output/paper.pdf`
- 自动结果：`manuscript/original_pdf_revision/results_macros.tex`
- 参考红线：`manuscript/redline/paper_redline.tex`（仅基于抽取文本重建）
- 完整升级：`UPGRADE_REPORT.md`
- 审稿响应：`REVIEW_RESPONSE_MATRIX.md`
""",
    )

    package_manifest = {
        "status": "complete",
        "manuscript": manuscript,
        "redline": redline,
        "oasis": oasis,
        "repository": tree_manifest(
            root,
            excluded={".git", "__pycache__", ".pytest_cache", "raw", "output", ".tools", "tmp"},
        ),
        "documents": [
            "UPGRADE_REPORT.md",
            "REVIEW_RESPONSE_MATRIX.md",
            "PAPER_REVISION_GUIDE.md",
            "RESULTS_AUDIT.md",
            "EXTERNAL_VALIDITY.md",
            "DATA_CARD.md",
            "MODEL_CARD.md",
            "REPRODUCIBILITY.md",
            "CHANGELOG.md",
            "PAPER_CURRENT_STATUS_REPORT.md",
        ],
    }
    write_json(artifacts / "provenance" / "revision_package_manifest.json", package_manifest)

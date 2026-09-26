# 结果审计

## 原论文精确复现

原论文中的 `18.4 -> 6.1` 是两个模拟条件之间的平均叶深变化，不是
真实 Reddit 回复树的平均叶深。两者不应放在同一行比较。

| 模拟指标 | 原论文 | 9000-run 精确复现 | 判断 |
|---|---:|---:|---|
| Baseline comment volume | 427.1 | 431.593 | 一致，误差 1.05% |
| Toxic comment volume | 1372.4 | 1385.412 | 一致，误差 0.95% |
| Pooled volume amplification | 3.20x | 3.210x | 一致，误差 0.31% |
| Baseline max depth | 22.2 | 21.988 | 一致，误差 0.95% |
| Toxic max depth | 18.5 | 18.790 | 一致，误差 1.57% |
| Baseline mean leaf depth | 18.4 | 18.498 | 一致，误差 0.54% |
| Toxic mean leaf depth | 6.1 | 6.061 | 一致，误差 0.64% |

运行共 9000 条，零缺失、零重复、零非法记录。冻结哈希见
`artifacts/provenance/paper_exact_freeze_manifest.json`。

## 现实数据与扩展实验

| 项目 | 当前结果 | 正确解释 |
|---|---:|---|
| Reddit 帖子 | 500 | 与原数据范围一致 |
| 可重建真实级联 | 482 | 用于现实模式验证 |
| 真实树平均叶深 | 2.222 | 与模拟器指标量纲不同，比较标准化方向而非绝对值 |
| 旧修订消融 | 108/615 场景—社区单元 | 探索性压力测试，不能否定原主结果 |
| 无条件域内拟合 | `branching_process` 均值距离最佳 | 说明简单基线拟合部分边际统计量更好 |
| LOCO | `zero_shot_branching_process` 均值最佳 | 只评价静态迁移，不评价干预预测 |
| 跨平台 zero-shot | HN 5000；Lemmy 7 | 评价 Reddit 参数可移植性，不等于框架可移植性 |
| TBBT | 38,700,732 条消息 | 可用于退出、活动和审核模块的现实验证 |
| 真实因果干预 | Lemmy pilot；TBBT descriptive | 尚未达到确认性因果门槛 |

## 两条基线赛道

现有模型排名属于“无条件级联拟合”：

| 模型 | 平均标准化 Wasserstein | 中位标准化 Wasserstein | 指标数 |
|---|---:|---:|---:|
| `branching_process` | 1.582 | 1.119 | 75 |
| `hawkes` | 2.989 | 0.823 | 75 |
| `learned_bdmtf` | 3.130 | 0.812 | 75 |
| `empirical_bootstrap` | 3.471 | 0.809 | 75 |
| `legacy_heuristic` | 3.821 | 0.848 | 75 |

该排名继续保留，但不能回答 Core、Structure、Context 干预的预测问题。
确认性修订将新增“干预响应预测”赛道，比较 conditional branching、
conditional Hawkes、三层消融和完整 BDMTF。

旧稿自动注入结果继续保留在
`manuscript/original_pdf_revision/results_macros.manifest.json`，在新确认性
实验完成前不再更新论文。

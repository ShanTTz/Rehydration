# Model Card

BDMTF 是线程式讨论的机制模拟器，不是人格诊断或通用社交平台因果模型。

- 主模型：训练集估计的聚合事件风险、回复目标和退出策略。
- 基线：经验重采样、负二项分支、Hawkes、legacy BDMTF。
- 语义层：冻结意图和三模型系列硬门；当前状态 `dry_run_ready`。
- 网络：回复树、共同参与和真实连接严格分开。
- 已知失败：`branching_process` 在域内平均分布距离上优于 learned BDMTF。
- 允许主张：只限 manifest 已完成且诊断通过的数据、平台与干预。

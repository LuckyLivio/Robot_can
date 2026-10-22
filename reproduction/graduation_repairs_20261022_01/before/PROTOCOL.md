# 研究协议、范围与版本索引

本文为事后文档整合，不是追溯性预注册。真正执行前的protocol/design/study_plan/lock保留在各实验目录，禁止改写。

范围：PickPlaceCan、Panda、23维含物体真值状态、20Hz、D=2、400步。当前从世界模型辅助候选选择扩展到共享失败恢复；两类贡献分别分析。

| 证据线 | 原登记 |
|---|---|
| 三种子匹配训练/开发 | [multiseed protocol](reproduction/multiseed_20261012_01/protocol.json) |
| 原世界模型确认 | [design](reproduction/confirmation_20261012_01/design.json)、[lock](reproduction/confirmation_20261012_01/lock.json) |
| 补充诊断 | [extensions protocol](reproduction/extensions_20261012_01/protocol.json) |
| 冻结恢复主线 | [v5 study_plan](reproduction/recovery_v5_20261012_01/study_plan.json)、[freeze](reproduction/recovery_v5_20261012_01/freeze_before_states.json)、[lock](reproduction/recovery_v5_20261012_01/confirmation_lock.json) |
| 本阶段整合 | [before_manifest](reproduction/consolidation_20261012_01/before_manifest.json)，不新增策略实验 |

主线训练/开发按父轨迹隔离；保留强BC k=2及全部种子，不挑确认表现最好的种子。旧sealed和全部既有确认已查看，不能继续冒充未见数据。人工压力测试与原生初始化、开发与确认、重复物理回放与新策略回合分开。

持续性、短期oracle、H=1/候选数补充已经完成；原协议的纯视觉/真机/多任务等扩展未完成。已有v2/v3四组对照见[新描述性汇总](reproduction/consolidation_20261012_01/FACTORIAL_REVIEW.md)，不算追加确认。

若改算法，另建目录并在运行前固定开发推进标准、配置、种子与统计方案，通过后冻结再生成新起点。后续候选草案不算已执行登记，见[FOLLOWUP_PLAN](reproduction/consolidation_20261012_01/FOLLOWUP_PLAN.md)。

文档复核：2026-10-15；本页2026-10-12实验口径保持原协议。旧版归档：reproduction/consolidation_20261012_01/previous_documents/

## 后续候选与当前审查

v6—v11各有独立协议，见[EVIDENCE_INDEX](EVIDENCE_INDEX.md)。最新[v11计划](reproduction/improvement_v11_20261015_01/protocols/PLAN.md)与[开发冻结](reproduction/improvement_v11_20261015_01/protocols/development_freeze.json)保留原字节。本轮审查不新增实验、不修改历史门槛；[当前报告](reproduction/project_audit_20261015_01/REPORT.md)。

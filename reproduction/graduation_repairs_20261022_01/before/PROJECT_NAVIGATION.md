# 项目分类导航 · 当前入口

**v21 已完成六组共54回合小测试与消融，未通过预定门槛，未采用。**

实际 v19 2/9；实际 v20 3/9；原预算候选 7/9；仅 A 重定位衔接 2/9；仅 B 终端放置 2/9；完整 v21 2/9。

每组9个条件共用8个不同物理起点：8个标准条件与1个复用6000005起点的clearance专用配置。A、B及组合均基于完整v19。A保留下降前三步原反馈，随后仅延长风险方向反馈；B只在实际终端放置启用，搬运沿用v19；原预算候选保护不同，仅作成功数参考。54回合不是54个独立起点，这组困难案例成功率不能当作项目总体成功率。

保持400步、真实D2 FIFO、原保护阈值与成功定义。564回合扩展诊断、三版各240回合开发对照（合计720）及1440回合独立确认均未运行，未生成新确认起点。保留v20、v19、预算候选与全部旧代码、权重和结果。未做泛化训练，未写论文。

[本轮完整报告](reproduction/improvement_v21_20261019_01/REPORT.md) · [v21 前端](reproduction/improvement_v21_20261019_01/index.html) · [正式协议](reproduction/improvement_v21_20261019_01/protocols/PLAN.md) · [v20 原报告](reproduction/improvement_v20_20261019_01/REPORT.md)



[逐条件配对表](reproduction/improvement_v21_20261019_01/analysis/smoke_a_pairs.csv) · [图表与来源](reproduction/improvement_v21_20261019_01/figures/CATALOG.json) · [完整度核验](reproduction/improvement_v21_20261019_01/verify_integrity.py)

## 历史内容（按各自版本和时间解释）

# 项目分类导航 · 当前入口

**v20 已完成六组共54回合小测试与消融，未通过预定门槛，未采用。**

实际 v17 3/9；实际 v19 2/9；原预算候选 7/9；仅 A 方向衔接 2/9；仅 B 延迟制动 3/9；完整 v20 3/9。

每组9个条件共用8个不同物理起点：8个标准条件与1个复用6000005起点的clearance专用配置。A、B及组合均基于完整v19；原预算候选保护不同，仅作成功数参考。54回合不是54个独立起点，这组困难案例成功率不能当作项目总体成功率。

保持400步、真实D2 FIFO、原保护阈值与成功定义。564回合扩展诊断、三版各240回合开发对照（合计720）及1440回合独立确认均未运行，未生成新确认起点。保留v19、预算候选与全部旧代码、权重和结果。未做泛化训练，未写论文。

[本轮完整报告](reproduction/improvement_v20_20261019_01/REPORT.md) · [v20 前端](reproduction/improvement_v20_20261019_01/index.html) · [正式协议](reproduction/improvement_v20_20261019_01/protocols/PLAN.md) · [v19 原报告](reproduction/improvement_v19_20261019_01/REPORT.md)



[逐条件配对表](reproduction/improvement_v20_20261019_01/analysis/smoke_a_pairs.csv) · [图表与来源](reproduction/improvement_v20_20261019_01/figures/CATALOG.json) · [完整度核验](reproduction/improvement_v20_20261019_01/verify_integrity.py)

## 历史内容（按各自版本和时间解释）

# 项目分类导航 · 当前入口

**v19 已完成八组共72回合小测试与消融，未通过预定门槛，未采用。**

实际 v17 3/9；实际 v18 2/9；原预算候选 7/9；仅 A 衔接窗口 2/9；仅 B 候选限额 2/9；仅 C 积分制动 2/9；A+B 2/9；完整 v19 2/9。

每组9个条件共用8个不同物理起点：8个标准条件与1个复用6000005起点的clearance专用配置。A/B/C/AB均加在完整v18上，原预算候选保护不同，仅作成功数参考。72回合不是72个独立起点，本组困难案例成功率不能当作整个项目总体成功率。

保留400步、D2 FIFO、原保护阈值与成功定义。752次扩展诊断、三版各240次开发对照（合计720）及1440次独立确认均未运行，未生成新确认起点；保留v18、预算候选及所有旧代码、权重、结果。未做泛化训练，未写论文。

[本轮完整报告](reproduction/improvement_v19_20261019_01/REPORT.md) · [v19 前端](reproduction/improvement_v19_20261019_01/index.html) · [正式协议](reproduction/improvement_v19_20261019_01/protocols/PLAN.md) · [v18 原报告](reproduction/improvement_v18_20261019_01/REPORT.md)



[配对表](reproduction/improvement_v19_20261019_01/analysis/smoke_a_pairs.csv) · [图表及来源](reproduction/improvement_v19_20261019_01/figures/CATALOG.json) · [文件完整度核验](reproduction/improvement_v19_20261019_01/verify_integrity.py)

## 历史内容（按各自版本与时间解释）

# 项目分类导航 · 当前入口

**v18 已完成六组共 54 回合小测试及单项消融：实际 v16 2/9、实际 v17 3/9、原预算候选 7/9、仅衔接修复 2/9、仅短路线修复 2/9、完整 v18 2/9。未通过预定门槛，完整 v18 未采用。**

每组 9 个条件包括 8 个标准条件和 1 个复用 6000005 起点的 clearance-only 配置，只有 8 个不同起点。两个单项修复分别加在完整 v17 上；预算候选仍为原 v15 仅预算 C，保护条件不同，仅用于成功数参考。54 次执行不是 54 个独立起点。

本轮实现下降指令连续衔接，以及原朝向、内移端点的直接路线候选。组件和静态认证通过不等于整回合成功。未绕过小测试门槛：564 次扩展诊断、三版各 240 次开发（共 720 次）及 1440 次全新起点确认均未运行，未生成新确认起点。保留 v17、预算候选及全部旧代码、权重和结果。未做泛化训练，未写论文。

[完整报告](reproduction/improvement_v18_20261019_01/REPORT.md) · [v18 前端](reproduction/improvement_v18_20261019_01/index.html) · [执行前协议](reproduction/improvement_v18_20261019_01/protocols/PLAN.md) · [v17 原报告](reproduction/improvement_v17_20261019_01/REPORT.md)



[逐条件配对表](reproduction/improvement_v18_20261019_01/analysis/smoke_a_pairs.csv) · [图表及来源](reproduction/improvement_v18_20261019_01/figures/CATALOG.json) · [完整度核验](reproduction/improvement_v18_20261019_01/verify_integrity.py)

## 历史内容（按各自版本与时间解释）

# 项目分类导航 · 当前入口

**v17 已完成七组共 63 次小测试：v15 4/9、v16 2/9、原预算候选 7/9、路径单项 2/9、箱沿单项 2/9、持物单项 2/9、完整 v17 3/9。未通过预定门槛，完整 v17 未采用。**

每组 9 个条件包括 8 个标准条件和 1 个复用 6000005 起点的 clearance-only 父配置 profile，只有 8 个不同起点；不得将 63 次执行视为 63 个独立样本。三个单项修复分别加在完整 v16 上，预算候选仍为原 v15 仅预算 C。

公共路径复用、短路线和反馈控制已实现并接受闭环检验；组件检查通过不等于整回合成功。预算候选的 7/9 仅表示任务成功，其中两个成功条件在 CARRY/PLACE 阶段低于现行 0.04 rad 关节余量，不能整体回退采用。658 次扩展消融、v15/v16/v17 各 240 次开发（共 720 次）及 1440 次全新起点确认均未运行。保留预算候选，既有主成果不替换。

[完整报告](reproduction/improvement_v17_20261019_01/REPORT.md) · [v17 前端](reproduction/improvement_v17_20261019_01/index.html) · [执行前协议](reproduction/improvement_v17_20261019_01/protocols/PLAN.md) · [v16 原报告](reproduction/improvement_v16_20261016_01/REPORT.md)



[逐条件配对表](reproduction/improvement_v17_20261019_01/analysis/smoke_a_pairs.csv) · [图表及来源](reproduction/improvement_v17_20261019_01/figures/CATALOG.json) · [完整度核验](reproduction/improvement_v17_20261019_01/verify_integrity.py)

## 历史内容（以下按各自版本与时间解释）

# 项目分类导航 · 当前入口

**v16已完成63次七组小测试：v14 / v15 / 预算候选 / IK修复 / 持物修复 / 箱沿修复 / 完整v16为6/9、4/9、7/9、2/9、4/9、2/9、2/9。门槛未通过，完整v16未采用。**

9个条件包含8个标准条件和1个复用6000005起点的预算配置profile，只有8个不同起点。标准条件下v16为2/8、v15为3/8、v14为6/8。三个修复单项分别加在完整v15之上；预算候选沿用原v15仅预算配置。

局部IK找到更多合法路径，部分持物转向达标，但没有提高整回合成功率。新的等待、规划与运输耗时导致预算不足，另有等价大角转向超时及观测失抓信号并触发恢复。预算模块继续保留为候选。既有正式成果不替换；658次扩展消融、三版各240次开发（共720次）和1440次全新起点确认均未运行。

[本轮完整报告](reproduction/improvement_v16_20261016_01/REPORT.md) · [v16前端](reproduction/improvement_v16_20261016_01/index.html) · [执行前协议](reproduction/improvement_v16_20261016_01/protocols/PLAN.md) · [v15原报告](reproduction/improvement_v15_20261016_01/REPORT.md)



[本轮逐条件配对表](reproduction/improvement_v16_20261016_01/analysis/smoke_a_pairs.csv) · [图表与来源](reproduction/improvement_v16_20261016_01/figures/CATALOG.json) · [完整度检查](reproduction/improvement_v16_20261016_01/verify_integrity.py)

## 历史内容（以下按各自版本和时间解释）

# 项目分类导航 · 当前入口

**v15已完成45次五组小测试：v14 / 仅搬运A / 仅分段B / 仅预算C / 完整v15为6/9、3/9、6/9、7/9、4/9。小测试未通过，v15未采用。**

9个条件包含8个标准条件和1个复用6000005起点的预算配置profile，不能当作9个独立起点。标准条件下完整v15为3/8、父v14为6/8；预算专用profile为v15成功、父版失败。

预算模块单独运行救回时间不足案例且未损失旧成功；搬运模块的局部IK误拒和转向耗时导致三个旧成功退步，两个主难例仍未完成。因此不替换既有正式成果，不进入470次消融、两版各240次开发或1440次全新起点确认。

[本轮完整报告](reproduction/improvement_v15_20261016_01/REPORT.md) · [v15前端](reproduction/improvement_v15_20261016_01/index.html) · [执行前协议](reproduction/improvement_v15_20261016_01/protocols/PLAN.md) · [v14原报告](reproduction/improvement_v14_20261016_01/REPORT.md)



[本轮逐条件配对表](reproduction/improvement_v15_20261016_01/analysis/smoke_a_pairs.csv) · [图表与来源](reproduction/improvement_v15_20261016_01/figures/CATALOG.json) · [完整度检查](reproduction/improvement_v15_20261016_01/verify_integrity.py)

## 历史内容（以下按各自版本和时间解释）

# 项目导航 · 当前入口

**v14 两轮小测试共64次；最终 v13 / T / C / v14 为6/8、6/8、5/8、6/8。门槛未通过，未采用；未运行大规模消融、240次开发或全新起点确认。**

[本轮检查报告](reproduction/improvement_v14_20261016_01/REPORT.md) · [v14前端](reproduction/improvement_v14_20261016_01/index.html) · [执行前协议](reproduction/improvement_v14_20261016_01/protocols/PLAN.md) · [v13原报告](reproduction/improvement_v13_20261016_01/REPORT.md)

[本轮原始配对表](reproduction/improvement_v14_20261016_01/analysis/smoke_b_pairs.csv) · [本轮图表](reproduction/improvement_v14_20261016_01/figures/CATALOG.json) · [完整度检查脚本](reproduction/improvement_v14_20261016_01/verify_integrity.py)

## 历史内容（以下为此前文档，按原版本时间解释）

> **v13更新（2026-10-16）**：v13完成：开发236/240；全新60起点确认v6 445/480、v12 437/480、v13 448/480。未通过本轮对照采用门槛，保留为候选；主要配对区间未完全高于零，不能宣称总体成功率改善。 [报告](reproduction/improvement_v13_20261016_01/REPORT.md) · [前端](reproduction/improvement_v13_20261016_01/index.html)。旧代码、权重、结果保留。

# 项目总导航

- 最新实现、结果和视频：[reproduction/improvement_v13_20261016_01/](reproduction/improvement_v13_20261016_01/README.md)。
- 直接查看：[v13前端](reproduction/improvement_v13_20261016_01/index.html)；总报告：[v13完整报告](reproduction/improvement_v13_20261016_01/REPORT.md)。
- 启动与复核：[README](README.md)、[PROGRAM_GUIDE](PROGRAM_GUIDE.md)、[DEMO_GUIDE](DEMO_GUIDE.md)。
- 全阶段证据：[EVIDENCE_INDEX](EVIDENCE_INDEX.md)、[RESULTS](RESULTS.md)。
- 跨设备：[THESIS_HANDOFF_LATEST](THESIS_HANDOFF_LATEST.md)。旧包不含v12/v13，使用完整同步项目。
- 旧路径未移动；分类入口仍在[WORKSPACE](WORKSPACE/index.html)。

## 历史记录（截至2026-10-15；按原批次解释）

> 历史阶段：2026-10-15论文图表复查：[新图与数据总览](reproduction/paper_quality_20261015_01/index.html) · [全项目报告](reproduction/paper_quality_20261015_01/REPORT.md) · [当前提纲与证据映射](reproduction/paper_quality_20261015_01/THESIS_OUTLINE_REVIEWED.md)。36张图重新生成；没有新增训练、仿真或论文正文。

> [全项目审查与清理/调整计划](reproduction/project_audit_20261015_01/REPORT.md) · [审查网页](reproduction/project_audit_20261015_01/index.html)

# 项目分类总导航

原路径保留；新增用途分组，不搬移冻结源码、权重、数据或历史结果。

- [打开分类导航](WORKSPACE/index.html)
- [源码与模块](WORKSPACE/01_code/README.md)
- [文档与规范](WORKSPACE/02_documents/README.md)
- [实验与证据](WORKSPACE/03_experiments/README.md)
- [数据与权重](WORKSPACE/04_data_weights/README.md)
- [前端与图表](WORKSPACE/05_visualization/README.md)
- [当前论文交接说明](THESIS_HANDOFF_LATEST.md)；[旧迁移包](THESIS_HANDOFF_20261015/00_READ_FIRST.md)
- [该历史阶段修改前10,270文件索引](reproduction/paper_quality_20261015_01/FILE_INVENTORY.csv)；[旧3,946文件快照分类](WORKSPACE/FILE_INDEX.csv)

建议收藏WORKSPACE及该历史阶段目录`reproduction/paper_quality_20261015_01`。新图和当前提纲优先使用该历史阶段版本；旧写作包用于既有40表与冻结资料，v6—v11另有增补。

演示服务仍从原项目根运行：

```powershell
& .\.venv\Scripts\python.exe -m http.server 8766 --bind 127.0.0.1
```

打开 http://127.0.0.1:8766/ ，分类导航为 http://127.0.0.1:8766/WORKSPACE/index.html 。写论文看迁移包03_figures/index.html可直接离线打开。

# 实验结果索引 · 当前入口

**v21 已完成六组共54回合小测试与消融，未通过预定门槛，未采用。**

实际 v19 2/9；实际 v20 3/9；原预算候选 7/9；仅 A 重定位衔接 2/9；仅 B 终端放置 2/9；完整 v21 2/9。

每组9个条件共用8个不同物理起点：8个标准条件与1个复用6000005起点的clearance专用配置。A、B及组合均基于完整v19。A保留下降前三步原反馈，随后仅延长风险方向反馈；B只在实际终端放置启用，搬运沿用v19；原预算候选保护不同，仅作成功数参考。54回合不是54个独立起点，这组困难案例成功率不能当作项目总体成功率。

保持400步、真实D2 FIFO、原保护阈值与成功定义。564回合扩展诊断、三版各240回合开发对照（合计720）及1440回合独立确认均未运行，未生成新确认起点。保留v20、v19、预算候选与全部旧代码、权重和结果。未做泛化训练，未写论文。

[本轮完整报告](reproduction/improvement_v21_20261019_01/REPORT.md) · [v21 前端](reproduction/improvement_v21_20261019_01/index.html) · [正式协议](reproduction/improvement_v21_20261019_01/protocols/PLAN.md) · [v20 原报告](reproduction/improvement_v20_20261019_01/REPORT.md)



[逐条件配对表](reproduction/improvement_v21_20261019_01/analysis/smoke_a_pairs.csv) · [图表与来源](reproduction/improvement_v21_20261019_01/figures/CATALOG.json) · [完整度核验](reproduction/improvement_v21_20261019_01/verify_integrity.py)

## 历史内容（按各自版本和时间解释）

# 实验结果索引 · 当前入口

**v20 已完成六组共54回合小测试与消融，未通过预定门槛，未采用。**

实际 v17 3/9；实际 v19 2/9；原预算候选 7/9；仅 A 方向衔接 2/9；仅 B 延迟制动 3/9；完整 v20 3/9。

每组9个条件共用8个不同物理起点：8个标准条件与1个复用6000005起点的clearance专用配置。A、B及组合均基于完整v19；原预算候选保护不同，仅作成功数参考。54回合不是54个独立起点，这组困难案例成功率不能当作项目总体成功率。

保持400步、真实D2 FIFO、原保护阈值与成功定义。564回合扩展诊断、三版各240回合开发对照（合计720）及1440回合独立确认均未运行，未生成新确认起点。保留v19、预算候选与全部旧代码、权重和结果。未做泛化训练，未写论文。

[本轮完整报告](reproduction/improvement_v20_20261019_01/REPORT.md) · [v20 前端](reproduction/improvement_v20_20261019_01/index.html) · [正式协议](reproduction/improvement_v20_20261019_01/protocols/PLAN.md) · [v19 原报告](reproduction/improvement_v19_20261019_01/REPORT.md)



[逐条件配对表](reproduction/improvement_v20_20261019_01/analysis/smoke_a_pairs.csv) · [图表与来源](reproduction/improvement_v20_20261019_01/figures/CATALOG.json) · [完整度核验](reproduction/improvement_v20_20261019_01/verify_integrity.py)

## 历史内容（按各自版本和时间解释）

# 实验结果索引 · 当前入口

**v19 已完成八组共72回合小测试与消融，未通过预定门槛，未采用。**

实际 v17 3/9；实际 v18 2/9；原预算候选 7/9；仅 A 衔接窗口 2/9；仅 B 候选限额 2/9；仅 C 积分制动 2/9；A+B 2/9；完整 v19 2/9。

每组9个条件共用8个不同物理起点：8个标准条件与1个复用6000005起点的clearance专用配置。A/B/C/AB均加在完整v18上，原预算候选保护不同，仅作成功数参考。72回合不是72个独立起点，本组困难案例成功率不能当作整个项目总体成功率。

保留400步、D2 FIFO、原保护阈值与成功定义。752次扩展诊断、三版各240次开发对照（合计720）及1440次独立确认均未运行，未生成新确认起点；保留v18、预算候选及所有旧代码、权重、结果。未做泛化训练，未写论文。

[本轮完整报告](reproduction/improvement_v19_20261019_01/REPORT.md) · [v19 前端](reproduction/improvement_v19_20261019_01/index.html) · [正式协议](reproduction/improvement_v19_20261019_01/protocols/PLAN.md) · [v18 原报告](reproduction/improvement_v18_20261019_01/REPORT.md)



[配对表](reproduction/improvement_v19_20261019_01/analysis/smoke_a_pairs.csv) · [图表及来源](reproduction/improvement_v19_20261019_01/figures/CATALOG.json) · [文件完整度核验](reproduction/improvement_v19_20261019_01/verify_integrity.py)

## 历史内容（按各自版本与时间解释）

# 实验结果索引 · 当前入口

**v18 已完成六组共 54 回合小测试及单项消融：实际 v16 2/9、实际 v17 3/9、原预算候选 7/9、仅衔接修复 2/9、仅短路线修复 2/9、完整 v18 2/9。未通过预定门槛，完整 v18 未采用。**

每组 9 个条件包括 8 个标准条件和 1 个复用 6000005 起点的 clearance-only 配置，只有 8 个不同起点。两个单项修复分别加在完整 v17 上；预算候选仍为原 v15 仅预算 C，保护条件不同，仅用于成功数参考。54 次执行不是 54 个独立起点。

本轮实现下降指令连续衔接，以及原朝向、内移端点的直接路线候选。组件和静态认证通过不等于整回合成功。未绕过小测试门槛：564 次扩展诊断、三版各 240 次开发（共 720 次）及 1440 次全新起点确认均未运行，未生成新确认起点。保留 v17、预算候选及全部旧代码、权重和结果。未做泛化训练，未写论文。

[完整报告](reproduction/improvement_v18_20261019_01/REPORT.md) · [v18 前端](reproduction/improvement_v18_20261019_01/index.html) · [执行前协议](reproduction/improvement_v18_20261019_01/protocols/PLAN.md) · [v17 原报告](reproduction/improvement_v17_20261019_01/REPORT.md)



[逐条件配对表](reproduction/improvement_v18_20261019_01/analysis/smoke_a_pairs.csv) · [图表及来源](reproduction/improvement_v18_20261019_01/figures/CATALOG.json) · [完整度核验](reproduction/improvement_v18_20261019_01/verify_integrity.py)

## 历史内容（按各自版本与时间解释）

# 实验结果索引 · 当前入口

**v17 已完成七组共 63 次小测试：v15 4/9、v16 2/9、原预算候选 7/9、路径单项 2/9、箱沿单项 2/9、持物单项 2/9、完整 v17 3/9。未通过预定门槛，完整 v17 未采用。**

每组 9 个条件包括 8 个标准条件和 1 个复用 6000005 起点的 clearance-only 父配置 profile，只有 8 个不同起点；不得将 63 次执行视为 63 个独立样本。三个单项修复分别加在完整 v16 上，预算候选仍为原 v15 仅预算 C。

公共路径复用、短路线和反馈控制已实现并接受闭环检验；组件检查通过不等于整回合成功。预算候选的 7/9 仅表示任务成功，其中两个成功条件在 CARRY/PLACE 阶段低于现行 0.04 rad 关节余量，不能整体回退采用。658 次扩展消融、v15/v16/v17 各 240 次开发（共 720 次）及 1440 次全新起点确认均未运行。保留预算候选，既有主成果不替换。

[完整报告](reproduction/improvement_v17_20261019_01/REPORT.md) · [v17 前端](reproduction/improvement_v17_20261019_01/index.html) · [执行前协议](reproduction/improvement_v17_20261019_01/protocols/PLAN.md) · [v16 原报告](reproduction/improvement_v16_20261016_01/REPORT.md)



[逐条件配对表](reproduction/improvement_v17_20261019_01/analysis/smoke_a_pairs.csv) · [图表及来源](reproduction/improvement_v17_20261019_01/figures/CATALOG.json) · [完整度核验](reproduction/improvement_v17_20261019_01/verify_integrity.py)

## 历史内容（以下按各自版本与时间解释）

# 实验结果索引 · 当前入口

**v16已完成63次七组小测试：v14 / v15 / 预算候选 / IK修复 / 持物修复 / 箱沿修复 / 完整v16为6/9、4/9、7/9、2/9、4/9、2/9、2/9。门槛未通过，完整v16未采用。**

9个条件包含8个标准条件和1个复用6000005起点的预算配置profile，只有8个不同起点。标准条件下v16为2/8、v15为3/8、v14为6/8。三个修复单项分别加在完整v15之上；预算候选沿用原v15仅预算配置。

局部IK找到更多合法路径，部分持物转向达标，但没有提高整回合成功率。新的等待、规划与运输耗时导致预算不足，另有等价大角转向超时及观测失抓信号并触发恢复。预算模块继续保留为候选。既有正式成果不替换；658次扩展消融、三版各240次开发（共720次）和1440次全新起点确认均未运行。

[本轮完整报告](reproduction/improvement_v16_20261016_01/REPORT.md) · [v16前端](reproduction/improvement_v16_20261016_01/index.html) · [执行前协议](reproduction/improvement_v16_20261016_01/protocols/PLAN.md) · [v15原报告](reproduction/improvement_v15_20261016_01/REPORT.md)



[本轮逐条件配对表](reproduction/improvement_v16_20261016_01/analysis/smoke_a_pairs.csv) · [图表与来源](reproduction/improvement_v16_20261016_01/figures/CATALOG.json) · [完整度检查](reproduction/improvement_v16_20261016_01/verify_integrity.py)

## 历史内容（以下按各自版本和时间解释）

# 实验结果索引 · 当前入口

**v15已完成45次五组小测试：v14 / 仅搬运A / 仅分段B / 仅预算C / 完整v15为6/9、3/9、6/9、7/9、4/9。小测试未通过，v15未采用。**

9个条件包含8个标准条件和1个复用6000005起点的预算配置profile，不能当作9个独立起点。标准条件下完整v15为3/8、父v14为6/8；预算专用profile为v15成功、父版失败。

预算模块单独运行救回时间不足案例且未损失旧成功；搬运模块的局部IK误拒和转向耗时导致三个旧成功退步，两个主难例仍未完成。因此不替换既有正式成果，不进入470次消融、两版各240次开发或1440次全新起点确认。

[本轮完整报告](reproduction/improvement_v15_20261016_01/REPORT.md) · [v15前端](reproduction/improvement_v15_20261016_01/index.html) · [执行前协议](reproduction/improvement_v15_20261016_01/protocols/PLAN.md) · [v14原报告](reproduction/improvement_v14_20261016_01/REPORT.md)



[本轮逐条件配对表](reproduction/improvement_v15_20261016_01/analysis/smoke_a_pairs.csv) · [图表与来源](reproduction/improvement_v15_20261016_01/figures/CATALOG.json) · [完整度检查](reproduction/improvement_v15_20261016_01/verify_integrity.py)

## 历史内容（以下按各自版本和时间解释）

# 实验结果索引 · 当前入口

**v14 两轮小测试共64次；最终 v13 / T / C / v14 为6/8、6/8、5/8、6/8。门槛未通过，未采用；未运行大规模消融、240次开发或全新起点确认。**

[本轮检查报告](reproduction/improvement_v14_20261016_01/REPORT.md) · [v14前端](reproduction/improvement_v14_20261016_01/index.html) · [执行前协议](reproduction/improvement_v14_20261016_01/protocols/PLAN.md) · [v13原报告](reproduction/improvement_v13_20261016_01/REPORT.md)

[本轮原始配对表](reproduction/improvement_v14_20261016_01/analysis/smoke_b_pairs.csv) · [本轮图表](reproduction/improvement_v14_20261016_01/figures/CATALOG.json) · [完整度检查脚本](reproduction/improvement_v14_20261016_01/verify_integrity.py)

## 历史内容（以下为此前文档，按原版本时间解释）

> **v13更新（2026-10-16）**：v13完成：开发236/240；全新60起点确认v6 445/480、v12 437/480、v13 448/480。未通过本轮对照采用门槛，保留为候选；主要配对区间未完全高于零，不能宣称总体成功率改善。 [报告](reproduction/improvement_v13_20261016_01/REPORT.md) · [前端](reproduction/improvement_v13_20261016_01/index.html)。旧代码、权重、结果保留。

# 实验结果总览（更新至v13）

本轮完整阶段共1976次评估执行，另保留中断批次的17个完成单任务与2个不完整任务。没有泛化训练或论文正文。正式v5冻结证据继续保留；v13与v6/v12的本批比较不能直接证明优于另批v5。

| 阶段 | 结果 | 证据边界 |
|---|---|---|
| 最终小测试 | v12 3/6、S 5/6、R 3/6、v13 6/6 | 已知诊断 |
| 四组消融 | v12 32/50、S 40/50、R 32/50、v13 41/50 | 50执行条件，复用部分起点 |
| 开发 | v6/v12/v13均236/240，无新增退步 | 30固定起点×8条件；没有提高成功率 |
| 独立确认 | v6 445/480、v12 437/480、v13 448/480 | 本轮冻结后60个新起点×8条件，按起点配对统计 |

[v13完整报告](reproduction/improvement_v13_20261016_01/REPORT.md)，配对原始记录见[reproduction/improvement_v13_20261016_01/analysis/confirmation_pairs.csv](reproduction/improvement_v13_20261016_01/analysis/confirmation_pairs.csv)。恢复控制与世界模型收益分别评价。

## 历史记录（截至2026-10-15；按原批次解释）

> 更新说明（2026-10-15）：原主线数值不变，新增候选版本汇总见文末；主版本仍v5。

# 当前实验结果：按阶段分别解释

## A. 世界模型原生确认（无共享恢复）

60个新起点，BC固定一个权重，Full/Plain各三个训练种子；14任务840回合，共享起点与扰动。

| 方法 | 干净 | 噪声0.05 |
|---|---|---|
| BC k=2 | 51/60，85.00% | 49/60，81.67% |
| Full | 54/54/54，每种子60；90.00% | 52/52/51，每种子60；86.11% |
| Plain | 53/53/53，每种子60；88.33% | 50/51/50，每种子60；83.89% |

Full−BC：干净+5.00个百分点，[0,11.67]；噪声+4.44，[0,10.00]。预登记等权综合+4.72，[0.56,10.28]；Full−Plain综合+1.94，[-0.56,6.11]。这是有边界的正向证据，不代表全部条件显著或排序模块已确立。原表：[确认报告](reproduction/confirmation_20261012_01/analysis/report.txt)。

## B. 补充开发诊断

88任务900回合，30个已知开发起点。短期oracle为干净25/30、噪声25/30；BC为25/30、24/30。持续性、H=1、K=1全回退k=2。D=2时H=1看不到新候选执行。K=8新增候选超出k=0…3训练支持，不能当纯计算量消融。详见[补充报告](reproduction/extensions_20261012_01/REPORT.md)。

## C. v5共享恢复独立确认

另一批60个新起点，每版两条件×BC/Full三种子=480次执行，五版共2400回合。与A不是同批起点。

| 版本 | BC干净 /60 | BC噪声 /60 | Full干净 /180 | Full噪声 /180 | 合计 /480 |
|---|---:|---:|---:|---:|---:|
| v2 | 59 | 57 | 173 | 168 | 457 |
| v4 | 59 | 57 | 173 | 168 | 457 |
| 仅时间 | 60 | 58 | 175 | 170 | 463 |
| 仅侧倒 | 59 | 57 | 174 | 168 | 458 |
| v5组合 | 60 | 58 | 176 | 170 | 464 |

v5−v4预定执行混合差+1.46个百分点，[0,4.17]；7次救回、0退步，集中在2个起点。相对v2救回8次、退步1次。v5比时间单项仅多一次执行成功，不宣称组合具有稳定附加优势。v5 BC干净100%、噪声96.67%；Full分别97.78%、94.44%，均值低于BC。

v5仍有16次失败、7个起点；6次物体低于恢复高度、4次失抓后耗尽尝试、6次其他尝试耗尽。15次是五版共同失败。几何压力试验另外16起点，v2/v4/T/S/v5为8/11/12/11/12，不并入原生成功率。详见[v5评估](reproduction/recovery_v5_20261012_01/ASSESSMENT.md)。

## D. 有无恢复的贡献

v2/v3已经在同批起点比较BC、Full、BC+恢复、Full+恢复。新[四组描述性分析](reproduction/consolidation_20261012_01/FACTORIAL_REVIEW.md)显示恢复同时帮助两类方法，不能将其全部收益归世界模型。最新v5缺少该批的关闭恢复组，不能借用旧轮次。

所有种子均报告。开发、不同确认批次、事后诊断和演示不合并；原始数据与输出未改。当前界限见[贡献](CONTRIBUTIONS.md)与[统计](STATISTICS.md)。

文档复核：2026-10-15；本页2026-10-12实验口径保持原协议。旧版归档：reproduction/consolidation_20261012_01/previous_documents/

## 后续候选v6—v11（2026-10-15）

| 候选 | 最终已知诊断 | 开发 | 采用情况 |
|---|---:|---:|---|
| v6 | 31/37 | 236/240 | 无净增，未采用 |
| v7合用 | 30/37 | 未启动 | 未过门槛 |
| v8合用 | 30/37 | 未启动 | 未过门槛 |
| v9合用 | 30/37 | 未启动 | 未过门槛 |
| v10合用 | 30/37 | 未启动 | 未过门槛 |
| v11合用 | 31/37 | 235/240 | 未采用；箱沿局部救回，继承等待退步 |

以上为反复使用的诊断/开发条件，不是新独立确认；不替换前文v5主结果。逐组证据见[EVIDENCE_INDEX](EVIDENCE_INDEX.md)。

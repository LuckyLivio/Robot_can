# CAN LAB · 机器人抓取与恢复控制

2026-10-22：完成高优先级修复、新BC配对训练、360次开发与1680次冻结后确认；新增13页世界模型系统详细设计。

[本轮完整报告](reproduction/graduation_repairs_20261022_01/REPORT.md) · [结果前端](reproduction/graduation_repairs_20261022_01/index.html) · [新活动代码与命令](robot_can_v2/README.md)

[世界模型系统详细设计（13页）](docs/world_model_design_20261022_01/README.md) · [可编辑Word](docs/world_model_design_20261022_01/基于世界模型的机器人抓取搬运系统设计与实现详细设计.docx) · [设计源稿](docs/world_model_design_20261022_01/详细设计.md) · [原52页版本存档](docs/detailed_design_20261022_01/README.md)

新实现统一观测时序，修复prediction_only与评估契约，拒绝覆盖旧产物。旧源码、权重、数据和实验结果保留；原正式v5仍是冻结参考，预算候选不因开发门槛通过就自动采用。

| 同批冻结策略 | 无恢复，干净/噪声 | v5恢复，干净/噪声 |
|---|---|---|
| BC | 56/60 / 55/60 | 59/60 / 60/60 |
| Full（三固定种子） | 168/180 / 168/180 | 178/180 / 174/180 |

确认共60个独立起点。Full重复种子与噪声执行不增加起点数。新BC观测确认另计240次，不能与旧BC混合。预算候选存在一次救回和一次退步，具体配对区间见本轮完整报告；本轮保留为未采用候选。

启动展示：

```powershell
& '.\.venv\Scripts\python.exe' -m http.server 8766 --bind 127.0.0.1
```

浏览 http://127.0.0.1:8766/ 。已有同目录服务时直接刷新。

[分类导航](PROJECT_NAVIGATION.md) · [运行指南](PROGRAM_GUIDE.md) · [结果](RESULTS.md) · [证据](EVIDENCE_INDEX.md) · [此前README原字节](reproduction/graduation_repairs_20261022_01/before/README.md)

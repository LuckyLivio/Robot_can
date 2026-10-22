# CAN LAB · 机器人世界模型与恢复控制

项目研究延迟控制下的罐体抓取搬运：冻结BC策略提供动作候选，世界模型预测评分，恢复控制器处理失抓、支撑姿态及受约束搬运。目前使用特权状态仿真，尚无视觉输入或实机验证。

## 当前状态（2026-10-19）

**v21 已完成六组共54回合小测试与消融，未通过预定门槛，未采用。**

实际 v19 2/9；实际 v20 3/9；原预算候选 7/9；仅 A 重定位衔接 2/9；仅 B 终端放置 2/9；完整 v21 2/9。

每组9个条件共用8个不同物理起点：8个标准条件与1个复用6000005起点的clearance专用配置。A、B及组合均基于完整v19。A保留下降前三步原反馈，随后仅延长风险方向反馈；B只在实际终端放置启用，搬运沿用v19；原预算候选保护不同，仅作成功数参考。54回合不是54个独立起点，这组困难案例成功率不能当作项目总体成功率。

保持400步、真实D2 FIFO、原保护阈值与成功定义。564回合扩展诊断、三版各240回合开发对照（合计720）及1440回合独立确认均未运行，未生成新确认起点。保留v20、v19、预算候选与全部旧代码、权重和结果。未做泛化训练，未写论文。

[本轮完整报告](reproduction/improvement_v21_20261019_01/REPORT.md) · [v21 前端](reproduction/improvement_v21_20261019_01/index.html) · [正式协议](reproduction/improvement_v21_20261019_01/protocols/PLAN.md) · [v20 原报告](reproduction/improvement_v20_20261019_01/REPORT.md)

## 启动前端

```powershell
Set-Location 'C:\Users\Livio\Desktop\robot_can'
& '.\.venv\Scripts\python.exe' -m http.server 8766 --bind 127.0.0.1 --directory 'C:\Users\Livio\Desktop\robot_can'
```

打开完整地址：**http://127.0.0.1:8766/reproduction/improvement_v21_20261019_01/index.html** 。已有8766服务时直接打开，不重复启动；服务终端保持运行，Ctrl+C停止。根地址会跳转到本轮前端。

页面包含六组实际结果、预定门槛、逐条件证据和分类图表。本轮没有新拍的v21机器人视频，历史演示按其真实版本标注。

## 导航

- [分类导航](PROJECT_NAVIGATION.md) · [环境与运行指南](PROGRAM_GUIDE.md)
- [项目状态](STATUS.md) · [结果索引](RESULTS.md) · [证据索引](EVIDENCE_INDEX.md)
- [演示指南](DEMO_GUIDE.md) · [写作材料交接](THESIS_HANDOFF_LATEST.md)
- [本轮目录和核验命令](reproduction/improvement_v21_20261019_01/README.md)
- [此前README原样快照](reproduction/improvement_v21_20261019_01/before/README.md)

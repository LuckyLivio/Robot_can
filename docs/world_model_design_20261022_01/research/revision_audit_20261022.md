# 13页详细设计复核与修订记录

复核基线：`da37c1f1babdc6632f2853406a29ae731e6813b0`。本次按当前源码、冻结训练配置和实际数据文件重新检查，不以旧版质量回执代替技术核对。修改仅限本目录的设计文稿、图稿、生成器及检查记录；原52页稿、控制代码、模型、数据和实验结果保持原样。旧13页稿可由该Git提交恢复。

## 已修正的问题与遗漏

| 范围 | 原问题或歧义 | 修订及实现依据 |
|---|---|---|
| 状态输入 | 绝对位姿描述可能暗示含有目标坐标 | 明确23维不含目标位置，固定目标几何供恢复器使用；`common.py:14–18`、`robot_can_v2/evaluate.py:165–170` |
| 标准化 | 统计来源未区分；递推表述可能理解为使用反标准化状态 | BC用训练窗口起点，WM用训练锚点；递推及任务头均使用标准化状态，输出才反标准化；`robot_can_v2/train_policy.py:114–115`、`reproduction/multiseed_20261012_01/train_suite.py:81`、`world_model.py:18–29` |
| BC窗口 | “该点之后”可能造成动作错一帧 | 明确动作块从a_t开始，包含当前动作；`common.py:143–150` |
| BC检查点 | 新旧训练入口的保存规则混述 | 区分新入口best.pt/final.pt与冻结基线policy_15000.pt；`robot_can_v2/train_policy.py:162–168`、`robot_can_v2/evaluate.py:125` |
| 预测损失 | “状态残差平方”易被理解为惩罚状态更新增量 | 明确为预测与真实状态误差除以尺度后的平方均值；`robot_can_v2/train_world.py:235–247` |
| 排序损失 | 真值差的绝对值、取对方式及空集行为不明 | 补充i<j、绝对差大于0.025及空集为零；`world_model.py:58–65` |
| 正式Full训练 | 缺训练超参、验证频率及检查点选用规则 | 补充16组、3,000更新、学习率0.0003、梯度上限10、每250步验证及最小开发后悔值；`reproduction/multiseed_20261012_01/train_suite.py:79–129`及三个冻结config.json |
| D2与评分 | 加噪前队首命令与实际动作符号可能混淆 | 明确式4-6的a_t为加噪前命令，Full评分覆盖全部8步；`robot_can_v2/evaluate.py:68–86,192–200` |
| 恢复判据 | 抓持边界、参考高度与失持锁存条件不全 | 补充严格开区间、与当前相对向量比较、CLOSE锚点高度和五帧失败；`recovery_controller.py:107–116,305–319` |
| 停滞与时间压力 | 触发名称不能充分解释实现 | 补充剩余步数估计、全局回合clock、距离改善与抬升的AND条件；`recovery_controller.py:186–202,305–347` |
| 图4-4 | 将不同异常概括为统一重试；忽略特殊旁路 | 默认重试回RETRACT，NORMAL持物停滞直接LIFT；活动恢复越界直接EXHAUSTED，目标区失持直接RELEASE；`recovery_controller.py:118–128,324–361,380–385` |
| 尝试次数 | 达到第二次尝试可能被理解为立即退出 | 明确第二次尝试允许完成，再次申请时才检测耗尽；`recovery_controller.py:118–128` |
| 成功与诊断 | 成功位置判据缺失，物体位置不应称质心 | 依据body_xpos说明罐体模型原点；末端距离阈值约0.0424 m由1−tanh(10d)<0.6导出；本地`robosuite/environments/manipulation/pick_place.py:398–416,726–751` |
| 稳定诊断 | 20步观察的动作、队列、噪声与稳定定义不明 | 张爪诊断保留原FIFO和噪声流；20步成功且末10步最大差分速度<0.02 m/s；`robot_can_v2/evaluate.py:97–117,206` |
| 分支字段 | 缺少queue/commands；point语义不具体 | 补齐形状，明确point为已执行前缀步数，非原示范帧索引；`collect_rollout_branches.py:156–181` |
| 历史元数据 | 容易理解为所有正式文件都显式保存语义及同一配置字段 | 区分实际历史缺省legacy、BC与WM配置字段、新canonical显式声明；`robot_can_v2/train_world.py:79–124`、`robot_can_v2/evaluate.py:55–64,125–131` |
| 接口与日志 | 标签与logits、实例名与类名、单作业与批次状态混淆 | 返回张量标明logits；采用RecoveryController.choose；区分mode/next_mode及result.json.status与批次完成回执；`world_model.py:18–29`、`robot_can_v2/evaluate.py:77–86,140–149,214–218`、`robot_can_v2/study.py:93–125` |

上表中`recovery_controller.py`指`reproduction/recovery_v5_20261012_01/recovery_controller.py`。正式恢复确认使用`robot_can_v2.evaluate`；历史`queue_v2.py`的预测补偿、基线索引和日志不作为本稿契约。通用预测对照入口也不替代正式冻结Full的评分规则。

## 数值与文风检查

- 重新读取实际分支：训练160组、20个父轨迹；开发64组、8个父轨迹；每组4候选、8步，数组形状与表4-4一致。
- 轻量CPU检查：BC为827,445个参数，动作块输出为B×8×7；Full为87,321个参数，返回B×8×23状态与B×8×2 logits。没有重新训练或运行物理仿真。
- 六个编号公式保留原运算关系，修订其变量空间、监督对象及调用时序说明。公式转换后逐对象核对减号与未转换命令，最终结果见quality_report.json。
- 保留第4章的结构，补齐BC、MLP、GRU、D2、FIFO及操作空间控制等首次出现的定义，统一单位和正式技术表述。未把未采用的预算候选写成正式成果，也未将恢复收益归因于世界模型。
- 运行学术缩写检查脚本。其对中文定义、软件名、状态常量和参考文献作者存在误报，已逐项人工判断；不据此改写MuJoCo、API标识或文献作者。英文及中文摘要不属于该章节的检查范围。
- 六图已逐张查看原始PNG；Word经本机Microsoft Word原生导出及随工具提供的光栅化程序转换。每页实际视觉检查及最终哈希见quality_report.json。

本记录是项目实现与设计稿的一致性复核，不是学校评审结论，也不声称完成完整ARS投稿审稿或机构标准认证。

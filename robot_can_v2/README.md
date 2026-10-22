# 新活动实现与复核入口

本目录修复输入观测和实验入口契约，并提供独立的预算候选及配对实验。根目录旧Python文件、旧权重、旧数据和历史结果保留；不要把旧脚本作为本轮修复后的入口。

本轮结果、运行计划、输入锁和原始轨迹见 [graduation_repairs_20261022_01](../reproduction/graduation_repairs_20261022_01/)。研究说明见该目录 METHODS.md；最终状态以该目录 REPORT.md 为准。不会自动采用未证明改善的预算候选，也不把新旧BC权重混成一个基线。

## 当前环境

使用项目 .venv/Scripts/python.exe；不需要修改现有虚拟环境。原 requirements-repro.txt 是历史建议，不是当前完整环境锁，不能按其旧torch注释降级当前环境。真实包版本记录在每个阶段 started.json 中。

## 只读复核

在项目根目录PowerShell运行：

    & '.\.venv\Scripts\python.exe' -B -X utf8 -m unittest discover -s robot_can_v2/tests -q
    & '.\.venv\Scripts\python.exe' -B -X utf8 reproduction/graduation_repairs_20261022_01/preserve.py

前一命令含临时目录中的微型训练测试，不覆盖正式产物。后一命令读取原资产并更新本轮专属校验回执；原导航字节从before/检查。

verify_traces.py、replay_physics.py和analyze.py的正式输出都拒绝覆盖，已有验证回执可直接阅读，不必删除或重跑。重复核查应另建输出或只读比较，不能把回放当新确认样本。

## 新训练入口

以下输出目录必须不存在；这里的路径是将来复现示例，不会覆盖本轮训练权重：

    & '.\.venv\Scripts\python.exe' -B -X utf8 -m robot_can_v2.train_policy --semantics canonical --output-dir reproduction/new_training/canonical_s17
    & '.\.venv\Scripts\python.exe' -B -X utf8 -m robot_can_v2.train_world --method prediction_only --dataset queuepair_20261012_01_real --future-only --output-dir reproduction/new_training/prediction_only_s17

BC默认horizon8、width512、mixtures5、15000更新、batch256、seed17、lr0.0003。canonical与legacy必须匹配相同初始化、批次、训练预算与在线观测。不要给旧冻结世界模型直接切换canonical输入；应重新采集并标记同契约分支，再训练新模型。

prediction_only不训练也不使用任务评分头；新评估使用物理代价。完整参数说明：

    & '.\.venv\Scripts\python.exe' -B -X utf8 -m robot_can_v2.train_policy --help
    & '.\.venv\Scripts\python.exe' -B -X utf8 -m robot_can_v2.train_world --help
    & '.\.venv\Scripts\python.exe' -B -X utf8 -m robot_can_v2.evaluate_delay --help

## 冻结作业重放

study.py run会验证确认锁，任何源码、权重、作业或状态漂移都拒绝执行。复现同一批作业时使用新的stage路径，例如：

    & '.\.venv\Scripts\python.exe' -B -X utf8 -m robot_can_v2.study run --jobs reproduction/graduation_repairs_20261022_01/confirmation_jobs.json --stage reproduction/graduation_repairs_20261022_01/recheck_confirmation --workers 4

这仍是同60个已知起点的复验，不是另一批独立确认。不要删除原确认目录以重复计数。

## 模块与数据流

observations统一同刻特征；train_policy/train_world拒绝输出覆盖；evaluate_delay处理候选与真实FIFO；evaluate处理共享恢复、完整计算计时和独立稳定性诊断；budget仅改变预算估计；metrics显式处理预测平局；study管理前瞻锁和作业；analyze按物理起点配对统计。

恢复控制的输入是特权23维状态和固定任务几何。奖励、接触和成功标签只用于事后评价，不进入恢复决策。当前仍为仿真系统，没有新增视觉或实机验证，也没有写论文。

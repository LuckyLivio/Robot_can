> 2026-10-15论文图表复查：[新图与数据总览](reproduction/paper_quality_20261015_01/index.html) · [全项目报告](reproduction/paper_quality_20261015_01/REPORT.md) · [当前提纲与证据映射](reproduction/paper_quality_20261015_01/THESIS_OUTLINE_REVIEWED.md)。36张图重新生成；没有新增训练、仿真或论文正文。

# 当前环境、冻结版本与新设备说明

更新：2026-10-15。PyCharm使用项目 `.venv\Scripts\python.exe`。本轮实测Python3.10.9、torch2.14.1+cu130、NumPy1.26.4、robosuite1.5.1、MuJoCo3.3.0、h5py3.11.0、SciPy1.15.3。pip check通过；RTX5080 Laptop CUDA可用且小张量计算通过；BC及Full/Plain各三种子在一个保存观测上前向通过。

[本轮环境与推理凭据](reproduction/paper_quality_20261015_01/environment_and_inference.json) · [完整pip快照](reproduction/project_audit_20261015_01/environment-freeze-current.txt)。这是实际环境快照，不是已验证可在任意机器安装的锁文件。本轮图表检查未新建仿真环境或运行闭环机器人；上一轮便携展示已在本机新标准库环境和中文移位目录验证。

`requirements-repro.txt`保留历史安装建议，其torch2.9.1注释不是实际冻结版本。不要按旧注释降级当前环境；新设备需核对该设备系统、GPU和可用包，再在新环境验证。[新设备方案](reproduction/project_audit_20261015_01/IMPROVEMENT_PLAN.md)

Windows兼容记录在 `data/windows_compatibility.json`；setup_windows.py会写项目虚拟环境中的DLL/WGL配置，已工作环境看演示不需要反复执行。不把.venv直接复制到另一台电脑。

原示范来源与SHA见[data manifest](data/can_ph_lowdim_v15.manifest.json)。BC主权重为 `policy/bc_chunk_dev_v4/policy_15000.pt`，Full/Plain为 `models/multiseed_20261012_01_{full,plain}_s{17,23,41}/model.pt`。原用户model.pt存于 `models/full_future_delay2_shift_v0/model.pt`；它保留为早期模型，并非当前三种子模型。两份导入权重均与最初替换凭据的SHA一致；桌面原文件目前不在该路径，不能重新直接比对桌面副本。

前端已有视频只需标准库HTTP服务；生成新视频另需ffmpeg和渲染依赖。历史任务JSON含原机器命令路径，新机器不能原样照搬命令数组。

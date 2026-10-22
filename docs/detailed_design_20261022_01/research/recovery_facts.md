# 控制、观测与恢复模块：源码核验事实

核验日期：2026-10-22。本文仅依据当前工作区源码读取，不运行训练或仿真，不修改原有代码、权重与实验结果。面向详细设计写作；本文件为核验笔记，不是新增实验结论。

## 1. 信息边界与研究对象

- 当前恢复确认实验使用 **仿真器提供的 23 维特权低维状态、已知固定目标位置和已知容器底面高度**，不是由图像估计这些量的视觉闭环系统。`robot_can_v2/evaluate.py:142` 明确标记 `PRIVILEGED_23D_STATE_AND_FIXED_TARGET`。`common.py:34–45` 在 `visual=False` 时不启用相机观测，但仍启用物体状态观测。
- 目标由 `env.target_bin_placements[env.object_to_id['can']]` 取出，底面由 `env.bin2_pos[2]` 提供给恢复器（`evaluate.py:164–170`）。这是已知任务参数的仿真初始化读取；不能写成完全无需环境先验。
- 恢复器 `RecoveryController` 只接收 `target, floor_z, initial_state, enabled, config`，每步只接收 `x, proposal, previous_action`，无 `env`、reward、接触力、关节动力学或世界模型对象（`recovery_controller.py:61–79,305`）。
- 世界模型评分发生在恢复器之前；恢复器不消费评分。`evaluate.py:181–185` 是先 `propose(...)`，后 `supervisor.choose(...)`。**世界模型动作筛选和恢复触发是两个串联、可独立开关的功能**。
- 环境成功判据 `env._check_success()` 只在执行真实控制后由评估器判断，用于终止与计分（`evaluate.py:198–204`），不是恢复触发信号。
- 几何资格与余量只针对固定场景中的近似罐体、夹爪碰撞凸包；不能推出机械臂全链条无碰撞、接触稳定、真机安全性或跨场景安全保证。

## 2. 23D 状态定义与 canonical 修复

源码：`robot_can_v2/observations.py:1–26,50–107`；原打包键顺序 `common.py:14–18`。

| Python 半开切片 | 维数 | 定义 | 坐标/单位 |
|---|---:|---|---|
| `[0:3]` | 3 | 末端执行器世界位置 `p_e` | 世界系，m |
| `[3:7]` | 4 | 末端姿态四元数 `q_e` | xyzw |
| `[7:9]` | 2 | 两夹爪关节位置 | 源观测关节位置；当前任务按 m 差值计算夹口 |
| `[9:12]` | 3 | 物体相对末端位置 | 末端坐标系，m |
| `[12:16]` | 4 | 物体相对末端四元数 | xyzw |
| `[16:19]` | 3 | 物体世界位置 `p_o` | 世界系，m |
| `[19:23]` | 4 | 物体世界姿态 `q_o` | xyzw |

`canonicalize_state` 对同一帧绝对位姿重建相对量：

\[
p_{o/e}=R(q_e)^\top(p_o-p_e),\qquad
q_{o/e}=q_e^{-1}\otimes q_o.
\]

1. 输入支持 `(...,23)`，实数、非复数、所有条目有限；内部用 float64 计算，返回新的 float32 数组（`observations.py:50–70,93–96`）。
2. 两绝对四元数先归一化，范数小于 `1e-12` 拒绝。原始相对四元数无须非零，因为其内容被丢弃；但输入所有值仍必须有限。
3. 相对四元数再次归一化，采用 `w>0` 的确定符号；当 `w=0` 时按 x、y、z 首个非零值确定正号。绝对四元数只归一化，保留其原有正负号（`observations.py:34–47,67–92`）。因此不能写“对所有四元数统一消除了正负双解”。
4. `legacy` 直接使用 `common.pack_obs`，保留历史观测缓存语义和旧权重兼容性；`canonical` 在打包后修复（`observations.py:99–107`）。
5. 训练读取器只允许 train/dev，禁止 sealed test；读出的 x、xn、a、reward 在内存中标记只读，HDF5 原件用只读模式打开（`observations.py:110–129`，`common.py:21–31`）。
6. 该修复明确输入几何契约，不等于保证策略成功率提升；正式旧 Full 权重仍绑定 legacy，不得直接用 canonical 输入混搭（`evaluate.py:41–42,59–62,127–131`）。

## 3. 控制接口、20 Hz、D2、400 步和精确 FIFO 顺序

源码：`evaluate.py:24–51,68–86,153–205`。

### 3.1 控制命令

7D 命令为 `[u_x,u_y,u_z,u_rx,u_ry,u_rz,u_g]`。评估启动时强制验证 OSC 为世界系增量控制，20 Hz，平移 `output_max=[.05,.05,.05]` m，旋转 `[.5,.5,.5]` rad，负界对称（`evaluate.py:153–157`）。提交和实际应用动作均限幅到 `[-1,1]`。这些是控制器命令比例，不意味着一步后末端实际位移一定等于命令值。

- 20 Hz 对应名义控制周期 `0.05 s`。
- D2 表示命令队列中有两个将在新提交命令之前执行的旧命令，名义命令延迟 `0.1 s`。
- 400 是**最多 400 次控制执行**，对应名义最多 20 s 仿真控制时间，不是 400 次 MuJoCo 内部积分步，也不是 20 s 墙钟耗时。
- 初始队列为两个全零 7D 命令；初始 `previous_action` 为零；前两次执行均来自这两个初始化零命令（若启用噪声，平移零命令仍会被加噪）。

若以 0 为第一控制索引，每步顺序严格为：

1. 从上一环境返回的 `obs_t` 打包 `x_t`。
2. BC 输出完整动作块 `B_t∈R^{8×7}`。
3. 使用**当前实际 FIFO**构建世界模型候选；无世界模型时直接选 `B_t[2]`。
4. 对队列预算候选，先复制并验证当前队列，再调用恢复器；恢复器更新内部历史、计数和状态机，得到新提交动作 `u_t`。
5. 保存 `state_before`、BC 动作块、`queue_before`、候选编号和评分、proposal、submitted_action、恢复诊断、计算耗时。
6. `pending.append(u_t.copy())` 后 `executed=pending.popleft().copy()`，故本次执行的是最旧命令，新命令不会立即执行。
7. 只给出队命令的前 3 维加独立零均值高斯噪声 `N(0,noise²)`；再将 7 维命令限幅为 `applied`，并保存 `previous=applied.copy()`。
8. 调用 `env.step(applied)`，随后查询成功判据并保存 `state_after`、实际应用动作和 `success_after`。
9. 首次成功立即结束基准循环；否则至 400 次执行结束。

等价队列关系（忽略加噪和限幅符号）为 `a_t=Q_t[0]`，`Q_{t+1}=[Q_t[1],u_t]`，故 `u_t` 于 `t+2` 执行。`previous_action` 始终指上一次实际应用的含噪、限幅后动作，而非上一轮新提交动作。

噪声 0.05 的单位是**归一化动作单位**；按已验证平移比例，对应标准差 0.0025 m 的平移命令增量，但不是末端位移误差或传感器噪声。源 `evaluate.py:195–198`。

### 3.2 Full 世界模型筛选

`propose` 只使用 BC 动作块前四个候选 `B[0],B[1],B[2],B[3]`（`evaluate.py:77–86`）。每个模型输入动作序列长度 8，内容为“实际待执行的两个队列命令 + 同一候选动作重复 6 次”（代码先 repeat 8 再整体截断为 8）。这不是把剩余 BC 动作块直接铺进模型。

模型输出标签的 sigmoid 记为 progress `p_h` 和 risk `r_h`，评分（`world_model.py:47–51`）：

\[
S_k=p_{k,7}+0.5\frac1{8}\sum_{h=0}^{7}p_{k,h}-0.5\max_h r_{k,h}.
\]

先取 argmax（平局取先出现索引）；若最优不是默认索引 2、但评分提升小于 0.02，则仍选 2。无模型始终选 2。该评分不访问部署时环境 reward。**恢复器仍可覆盖筛选后的 proposal**。

### 3.3 时间统计

计算耗时计时从打包观测前开始，包含 BC、可选 WM、恢复器；不含 `env.step`、JSON 写入、成功检测、后续稳定诊断和评估环境创建（`evaluate.py:143,178–186`）。输出 p50/p95/p99/max、超过 50 ms 的次数和比例（`89–94`）。并行负载可影响统计，不能据此直接声称完整物理闭环严格实时。

## 4. v5 观测抓持与丢失识别

源码：`recovery_controller.py:102–116,305–323`。

记夹口 `g=|x_7-x_8|`，世界系末端物体差 `r=x[:3]-x[16:19]`，物体初始静置高度 `z_rest=initial_state[18]`。

`observable_hold=True` 同时要求：

1. `0.025<g<0.067` m。
2. `||r||<0.09` m。
3. 当前在内最近 5 帧历史均存在，且相对当前 `r` 最大偏差 `<0.012` m。
4. 物体高度比初始静置高度 `>0.02` m；在 TEST_LIFT 且 close 锚点存在时，比较基准改为进入 CLOSE 时的物体高度。
5. 上次实际执行动作的夹爪通道 `previous_action[6]>0`，或抓持锁存状态已为真。

注意最近 5 帧是**相对位置一致性窗口**，不能简化成“连续五次 hold 判定为真”；历史在本轮 hold 检查之前追加，窗口包含当前观测。

一旦 hold 真，`stable/ever_stable` 锁存为真、丢失计数清零；锁存后 hold 假，每帧丢失计数加一。计数 `>=5` 且非主动释放时，认定 loss。主动释放包括 RELEASE/RETREAT，或 NORMAL 中曾抓稳、物体在目标平面区域且 proposal 为开爪时设置的未来 25 步抑制窗口。

内部 `inside_goal` 只检查目标 XY 误差：`abs(dx)<.0875` 且 `abs(dy)<.1125` m（`102–105`）。它不是 benchmark 的环境完整成功判据，也不检查放稳或高度。

## 5. NORMAL 触发规则与优先级

源码：`recovery_controller.py:305–359`。每次 choose 先 `clock+=1,age+=1`，再追加历史；历史上限 41 帧。距离比历史 best 小 `>0.008 m` 才更新 progress_step。

恢复工作空间：`-.24<x_o<.45`，`-.57<y_o<.57`，`z_o>floor-.04`。活动恢复期间物体出界，记录失败并进入 EXHAUSTED；NORMAL 中不满足 reachable 不启动恢复。

NORMAL 按以下 if/elif 顺序，最多首先触发一类：

| 优先级 | 事件 reason | 条件 |
|---:|---|---|
| 1 | `observed_grasp_loss` | loss 且 reachable |
| 2 | `early_stall_preserve_budget` | time_budget 开启，非 hold、非主动释放、reachable、未在目标 XY、clock≥80、末端物体距离>.12 m、time_pressure、至少25帧历史、25帧窗口距离改善<.008 m且物体抬升<.02 m |
| 3 | `long_no_progress` / `recovery_budget_guard` | 非 hold、非主动释放、reachable、未在目标 XY、clock≥80，且距离 best 最后更新后≥65步或clock≥160；先满足65步则 reason 为前者 |
| 4 | `held_stall` / `near_object_stall` | NORMAL age>40，历史>40，非主动释放、reachable，且 holding 或clock≥80；40步进展<.005 m，并且距离<.12 m或曾抓稳 |

优先级4的进展：抓持时取“物体升高”和“到目标 XY 距离减少”的最大值；未抓持时取“末端物体距离减少”和“物体升高”的最大值。若 holding 触发，直接 LIFT；否则从 RETRACT 重新规划抓取。

`start_attempt` 在已有尝试数 `>=2` 时进入 EXHAUSTED，且不增加尝试数；否则加一并复位当前抓稳状态、lost_frames（`118–128`）。attempt 额度为整回合累计，不会在返回 NORMAL 后重置。

## 6. v5 状态机与逐阶段动作

源码：`recovery_controller.py:81–100,287–304,360–406`。状态集合为 NORMAL、RETRACT、ALIGN、DESCEND、CLOSE、TEST_LIFT、LIFT、CARRY、PLACE、RELEASE、RETREAT、EXHAUSTED。

| 当前状态 | 执行动作 / 目标 | 常规转换 |
|---|---|---|
| NORMAL | 透传 BC/WM proposal；监测失败 | 触发恢复后进入 RETRACT，已有稳定抓持的 held_stall 进入 LIFT |
| RETRACT | 开爪、保持起始姿态，目标 `[eef_x,eef_y,max(floor+.20,eef_z)]` | age>D+5 且 z≥waypoint_z−.012 → ALIGN |
| ALIGN | 开爪，对齐物体上方，目标XY为grasp_target，z=max(floor+.20,obj_z+.14)，姿态为计划抓取姿态 | age>D+5，位置误差<.014 m，旋转误差<.15 rad → DESCEND |
| DESCEND | 开爪下降到 grasp_target | age>D+5 且位置误差<.009 m → CLOSE |
| CLOSE | 夹爪+1，维持抓取位姿 | age≥16+D=18 → TEST_LIFT |
| TEST_LIFT | 夹爪+1，目标为进入本状态时 `[obj_x,obj_y,eef_z+.06]` | age≥D+5=7 且 observable_hold → LIFT，verified_regrasps加一；否则age≥35重试 |
| LIFT | 夹爪+1，记录进入时末端-物体偏移，目标物体z=floor+.28并加该偏移 | obj_z≥floor+.28−.016 且 holding → CARRY |
| CARRY | 夹爪+1，目标 `[target_x,target_y,floor+.28]+held_offset` | age>D+5 且物体目标XY距离<.045 m → PLACE |
| PLACE | 夹爪+1，目标 `[target_x,target_y,floor+.085]+held_offset` | age>D+5，XY距离<.05 m且物体z<floor+.10 → RELEASE |
| RELEASE | 在进入时末端位置开爪 | age≥14+D=16 → RETREAT |
| RETREAT | 开爪，目标为进入时末端位置上方 .12 m | age>D+5且位置误差<.02 m → NORMAL，清抓稳/丢失/历史 |
| EXHAUSTED | 透传 proposal，intervened=False | 没有自动回到恢复状态的转换；不是急停或零动作 |

阶段动作先按当前 phase 计算，再判断并更新下一 mode，因此本步日志 `mode` 表示动作所属阶段，`next_mode` 是更新后状态。不能把 transition 本步动作误归为 next_mode。

### 6.1 通用超时与异常转换

- 除 NORMAL、EXHAUSTED、CARRY 之外，age>阶段超时则 `failed_attempts++` 并 `start_attempt('phase_timeout')`。普通超时 70；ALIGN 使用 alignment_deadline，切换等价方向尝试时110；CARRY age>110 用 `transport_timeout`（`360–365`）。
- LIFT/CARRY/PLACE 中观测到 loss：若物体已在目标XY，则转 RELEASE；否则失败计数加一再重试（`348–352`）。
- TEST_LIFT age≥35且未确认抓持，`lift_probe_failed`。
- 活动恢复中物体离开约定恢复工作空间，EXHAUSTED。
- 最大尝试数用尽，EXHAUSTED。以上都是有界仿真回退策略，不能描述为对任意场景的保证恢复。
- 角落方案生效但物体已不属于原静置格时，重新 plan_grasp，必要时更新目标姿态（`353–355`）。

### 6.2 下降与对齐的专门监测

下降（`204–211`）：DESCEND 目标距离改善超过 .003 m 刷新进展时刻；age≥25、连续≥18步无此改善、误差≥.012 m、时间有压力、仍有尝试额度，触发 `descent_stalled_preserve_budget`。

对齐（`254–273`）：误差 `E=||p_goal-p_e||+.08*angle`；改善>.003 m刷新时刻。物体最近18帧距当前位置最大偏差<.008 m视为 settled；age≥24且18步无进展视为 stalled。

- age≥70但仍有进展、物体静止且 upright resting_cell 合格，将 ALIGN deadline 延长至110。
- stalled、settled、位置误差≥.014或旋转误差≥.15、未切换过方向、仍有额度且resting_cell合格：先失败计数加一并重新开始尝试，再把 planned_rotation 设置为原目标绕世界Z旋转π的等价抓取姿态，标记 direction_switched 和 alternate_attempt。
- 替代姿态 ALIGN 的单步旋转限幅为 .25 rad，普通 .15 rad。

### 6.3 位姿控制公式

源码 `287–304`：平移 `δp=.5(p_goal-p_e)`，范数限幅 .035 m；满足冻结 v5 时间压力、阶段 RETRACT/ALIGN/LIFT/CARRY，且后两者已 stable 时，可用 .05 m。旋转 `δr=.5*Log(R_goal R_e^T)`，范数限幅 .15 rad（替代 ALIGN 为 .25）。返回 `[δp/.05,δr/.5,grip]` 再逐维限幅 `[-1,1]`。

如果方案要求特定 approach_gap，在 RETRACT/ALIGN/DESCEND 阶段覆盖开爪命令为 `clip((当前gap-目标gap)*12,-.2,.2)`。这点应在详细设计中解释，不能笼统写这三个阶段恒为−1开爪。

## 7. 抓取几何资格与夹爪余量

### 7.1 直立、侧卧与初始 yaw

`plan_grasp`（`130–154`）由物体姿态矩阵第三列得到罐轴 a。`|a_z|<.6` 视为 sideways，初始 yaw 为轴的 atan2；否则 upright，根据固定内壁在 X/Y 上的最近距离选择 yaw=0或π/2。在 yaw、yaw+π 两个几何等价姿态中选择距当前姿态旋转角较小者，基本向下姿态的四元数为 `[1,0,0,0]`（xyzw）。

直立 grasp_target 为物体原点 z+.012；侧卧目标先沿罐轴减 .01 m，再在z加 .008，并约束不低于floor+.035；随后加角落平面偏移与侧卧三维偏移（`275–285`）。

### 7.2 Upright resting_cell 与角落方案

静置资格（`213–227`）：`|a_z|≥.90`，物体原点z在 `[floor+.045,floor+.075]`，且罐体圆 footprint 相对所有所在格内壁余量≥.027 m。固定格如下：

\[
(-.09,.29,-.49,-.01),\quad
\{(l,h,b,t): (l,h)\in\{(-.09,.09),(.11,.29)\},\ (b,t)\in\{(.04,.27),(.29,.52)\}\}.
\]

只有两个平面方向最近壁距都 `<.065` m 时才启用角落搜索（代码 `max(clear)<corner_distance`）。朝离开两近壁方向偏移各 .006 m，夹口 .060 m。候选 yaw 从−π/4到3π/4，间隔π/12；近似平面夹爪 extent 采用 .010 与 `gap/2+.017`。

候选同时要求壁余量≥.001 m，且考虑偏心的罐体能放入夹口：`.025+abs(offset·closure)≤gap/2−.001`。评分为壁余量减方向偏好与微小yaw罚项，取最大；再选 yaw+π 等价中较近的姿态。无候选时保留原方案。代码 `229–252`。

### 7.3 Side_cell 资格与侧卧搜索

`clearance.py:9–16`：轴向中心 `c=p_o-.01a`，罐体在各轴上包围盒半径

\[
e=.05|a|+.025\sqrt{\max(0,1-a^2)}.
\]

资格要求 `|a_z|≤.35`，下边界 `c_z−e_z∈[floor+.012,floor+.030]`，且完整XY包围盒相对某固定格每侧余量≥.001 m。侧卧分类阈值 .6 与侧卧静置资格 .35 不同，不能混用。

`plan_side_clearance`（`156–184`）：baseline 以 .080 m夹口调用余量；若余量≥.001则保留。否则搜索：高度 .008/.014/.020 m；轴向偏移 0/−.015/+.015 m；四个墙法向的离壁倾斜方向；角度0/10/20/30/35°；候选夹口 .056 m。合格余量≥.001 m；最小化 `.03*degree/35+2*(height−.008)+.5*|shift|`，再按负余量、高度、偏移、角度排序解决平局。无合格候选只记 eligible=True,feasible=False，不拒绝所有恢复、不保证可抓成功。

### 7.4 碰撞近似的明确边界

`clearance.py:18–36`：加载固定 `gripper_geometry.json` 中夹爪碰撞凸包顶点与边，按给定夹口移动两指；转到世界坐标；检查最低点相对 floor+.021 的间隙，并仅对 wall_top=floor+.103 以下部分检查格内XY signed margin。边穿越 wall_top 的交点也纳入，避免只查顶点漏检穿墙。

文件注释明确 **robot arm is not represented by this model**。罐体只有简化包围盒资格，夹爪使用固定姿态凸包；不是全机器人连续碰撞检测、接触动力学验证或真机安全认证。

## 8. 冻结 v5 时间预算

`recovery_controller.py:186–202`，记物体上方点 `p_above=[obj_xy,max(floor+.20,obj_z+.14)]`：

\[
A=\lceil\|p_{above}-p_e\|/.012\rceil+8,\quad
T=\lceil\|p_{o,xy}-p_{target,xy}\|/.010\rceil+10.
\]

其剩余步估计为：CARRY/PLACE/RELEASE/RETREAT：`T+40`；LIFT：`25+T+40`；其余：`A+70+T+40+50`。

`time_pressure` 使用严格 `<`：`400-clock < remaining_schedule(x)`。在未稳定持物运输的情况下，还要求 upright resting_cell 或 sideways side_cell 至少一种资格，否则返回False。稳定抓持且阶段为LIFT/CARRY/PLACE时不要求静置资格。该估计是启发式时间预留，不是成功oracle或严格可达性证明。

## 9. 队列预算候选精确算法（未替换正式 v5）

源码 `robot_can_v2/budget.py`。继承冻结v5，只替换时间估计相关判断；禁用时直接委派父类并保持输出。候选在本次实验中未获稳定净收益，设计书应标为“已实现并验证的探索候选”，不能作为已证实有效的改进。

### 9.1 固定参数和验证

窗口8个差分样本（最多9个位置点），最少4个同阶段差分样本；名义 .012 m/step，速度范围[.006,.020]；相比v5原估计最多减25或加50步。只允许D2/400步/最多2次尝试。`25–32,50–78`。

队列输入必须是有限 `(2,7)`，复制并限幅 `[-1,1]`。缺失队列视为 unavailable，不虚构投影。`81–91`。

### 9.2 路径阶段表

`93–134` 的阶段表是**计时近似**，不替代v5状态机：

| 阶段 | 计时waypoint | 强制停留步数 |
|---|---|---:|
| RETRACT | 当前XY、max(floor+.20,current_z) | D+6=8 |
| ALIGN | 物体XY、max(floor+.20,obj_z+.14) | 8 |
| DESCEND | grasp_target | 8 |
| CLOSE | grasp_target | close_steps+D=18 |
| TEST_LIFT | graspXY、grasp_z+.06 | D+stable_frames=7 |
| LIFT | 物体XY、floor+.28，再加抓持偏移 | 7 |
| CARRY | 目标XY、floor+.28，再加抓持偏移 | 8 |
| PLACE | 目标XY、floor+.085，再加抓持偏移 | 8 |
| RELEASE | RELEASE时取当前末端，否则place点 | 14+D=16 |
| RETREAT | release点上方 .12 | 8 |

从当前mode对应阶段切掉已过去的阶段；NORMAL/EXHAUSTED从第一个阶段估计。活动状态首阶段强制停留减去当前age并下限0；RETRACT/TEST_LIFT/LIFT/RELEASE/RETREAT有真实waypoint时首阶段使用该waypoint。未验证抓取时用默认grasp-obj作为抓持偏移。

### 9.3 观测速度估计

只取连续、与当前mode相同的最近末端位置；不跨阶段拼接（`136–170`）。位置差每维先clip到±.020再按范数限幅.020，以防极大有限输入溢出或夸大速度。设差分 `d_i`，指向下一个非平凡waypoint的单位方向 `n`：

\[
m=\operatorname{median}\|d_i\|,\quad
g=\operatorname{median}(d_i\cdot n),
\]

若waypoint距离≤1e-9，则g取m；无差分时m=g=0。

\[
v=\operatorname{clip}\left(\begin{cases}.012,&N<4\\.006+.5\max(0,g),&N\ge4\end{cases},.006,.020\right).
\]

队列单步投影上限 `v_cap=clip(m,.006,.020)`（有样本）或`clip(v,.006,.020)`（无样本）。这是每控制步位移启发式，不是m/s；换成名义m/s需乘20。

### 9.4 剩余步数和 FIFO 修正

设上述阶段目标为w_j、强制停留d_j，w_0为当前末端。先按统一当前估计速度v算：

\[
R_{raw}=\sum_j\max\{d_j,\|w_j-w_{j-1}\|/v\}+R_{retry}.
\]

mode属于 NORMAL/EXHAUSTED/RETRACT/ALIGN/DESCEND/CLOSE/TEST_LIFT 时 `R_retry=50`，否则0。

\[
R_b=\operatorname{clip}(R_{raw},\max(0,R_{v5}-25),R_{v5}+50).
\]

对实际队列动作u_i，平移命令 `d_i=.05*u_i[:3]`，逐条限范数到v_cap，再相加为q；忽略旋转/夹爪对平移投影的影响。设g_next为距当前末端> .001 m的第一个阶段目标（若无则首目标）：

\[
\Delta R_q={\|g_{next}-(p_e+q)\|-\|g_{next}-p_e\|\over v},\qquad
R_{wait}=\begin{cases}2,&队列可用\\0,&不可用\end{cases},
\]

\[
\widehat R=\operatorname{clip}(\lceil R_b+R_{wait}+\Delta R_q\rceil,0,400).
\]

源码 `172–221`。注意“−25/+50”限幅作用于 **加入队列等待和运动修正前的阶段估计**，并非最终估计对v5的严格偏差界。

恢复触发实际调用继承的 `time_pressure`，保留v5几何资格；`budget_diagnostics['pressure']` 只报告 `400-clock<estimate`，**本身不做几何资格过滤，不可等同实际干预事件**。

### 9.5 与 v5 的隔离

- `remaining_schedule` 在候选启用时用新估计。
- `pose_action` 临时设置 `_v5_transit_pressure=True`，确保父类选择 .035/.05 平移步长时仍使用原v5估计。因此候选调整触发时机，不同时改变快移规则（`223–236`）。
- 只对reason `early_stall_preserve_budget` 和 `descent_stalled_preserve_budget` 的成功新增恢复事件附上触发前queue_budget诊断；每步debug另有诊断（`238–254`）。
- `choose` 在父choose之前记录本次EEF位置及旧mode；父choose可能切mode，随后诊断如果新mode历史不足会回退名义速度。因此同一帧事件中触发前诊断和动作后的debug诊断可能对应不同phase，不得混作同一时刻估计。

## 10. 成功后稳定诊断：与抓持稳定、基准成功分开

源码 `evaluate.py:97–117,206–214`。

基准在第一次 `env._check_success=True` 时已终止。只对成功回合追加20次诊断控制：每次向现存FIFO追加 `[0,0,0,0,0,0,-1]`，先执行尚未完成的旧队列命令；同样在执行端加噪并限幅。不能写“成功后立即停止所有旧命令”。

记录每步物体位置、有限差分速度 `20*||p_k-p_{k-1}||` m/s、环境原成功判据。`stable=True` 当且仅当20次都满足原成功判据，且末10次最大物体速度 `<.02 m/s`。这是约1s的开爪保持/沉降观察，包含前2个在途命令；不改变原success、原steps、独立样本数量或成功率分母。失败回合不执行，置 observed=False、stable=False。

## 11. 异常处理与可复核日志

- 作业枚举、semantics、噪声、唯一非空demo、steps/delay整数、400/D2、max_attempts=2、stable_frames=5、settle_steps=20、Full固定seed17/23/41均在启动验证（`evaluate.py:24–51`）。
- 已存在输出目录直接 FileExistsError，避免覆盖；checkpoint语义、dim=23/horizon=8与dataset demo存在性启动校验（`120–139`）。
- Full权重强制full/23维/对应seed/future_only，legacy语义，模型horizon契约由validate_model_config检查（`55–65`）。
- 输出目录新建后先写running；每个完成回合追加record，flush trace与settling trace并保存running，全部完成后complete。异常则尽力记录 `error=repr(error)`、failed，再原样抛出；finally关闭已创建env（`140–150,213–222`）。原结果写入是普通write_text，若写入自身失败，这段exception记录也可能失败；后续基础设施重跑适配器属于独立留痕流程，不能写成本评估器原生已有原子提交。
- trace逐步保留状态前后、BC块、队列前值、筛选评分/索引、proposal、提交动作、出队加噪前动作、实际含噪动作、恢复诊断、成功标记。恢复 events 保留step/from/to/reason/attempt/object_position（`recovery_controller.py:81–84`）。
- 原始v5自身不全面验证所有finite/shape，使用SciPy的quat转换也可抛异常；上游canonical、评估契约与候选预算的验证提高入口约束，但不应声称所有legacy在线异常已被彻底自动恢复。

## 12. 设计文档中应避免的过度表述

1. 不把“observations-only”写成“无特权观测”或“只用摄像头”。
2. 不把模型risk标签称为经过校准的安全概率，也不把模型筛选称作恢复触发器。
3. 不把相对位置5帧一致性、首次环境成功、额外20步放置稳定诊断混为同一指标。
4. 不把EXHAUSTED描述为停止机器人或安全急停。
5. 不把固定几何余量筛选描述为完整机械臂无碰撞保证或真机安全保证。
6. 不把20Hz仿真控制频率描述为已验证全链路50ms硬实时。
7. 不把canonical工程修复与队列预算探索写成已经取得稳定成功率提升。
8. 不把队列预算的理论估计变化、debug pressure或每步日志帧数直接算作实际干预次数。

# 模型与数据事实提取记录

核对日期：2026-10-22。用途：详细设计写作的代码依据；本文不是论文正文。仅阅读代码和已有配置，未运行训练或仿真。除本文件外不修改任何文件。下文代码路径相对于 `C:/Users/Livio/Desktop/robot_can/`；引用时可加此前缀。实际网络定义位于根目录 `world_model.py`，不是 `models/world_model.py`（`models/` 主要存实验权重）。

## 1. 当前实现的输入范围与观测语义

- `common.py:13-18`：数据文件为 `data/can_ph_lowdim_v15.hdf5`；按 EEF 位置、EEF 四元数、夹爪关节、物体观测顺序拼接。采用仿真特权低维状态；不是图像感知，也没有视觉编码器。
- `robot_can_v2/observations.py:3-8,19-26,50-96`：完整 23 维定义如下（零基、左闭右开）。

| 索引 | 维数 | 物理含义 | 坐标/顺序 |
|---|---:|---|---|
| `0:3` | 3 | 末端执行器位置 | 世界坐标，米 |
| `3:7` | 4 | 末端执行器姿态 | 世界坐标，四元数 `xyzw` |
| `7:9` | 2 | 夹爪关节位置 | 原仿真关节量 |
| `9:12` | 3 | 物体相对 EEF 的位置 | EEF 局部坐标，米 |
| `12:16` | 4 | 物体相对 EEF 的姿态 | `xyzw` |
| `16:19` | 3 | 物体位置 | 世界坐标，米 |
| `19:23` | 4 | 物体姿态 | 世界坐标，`xyzw` |

新语义 `canonical` 从同一帧绝对位姿重算冗余相对量：

\[
p_{o/e}=R(q_e)^T(p_o-p_e),\qquad q_{o/e}=\bar q_e\otimes q_o.
\]

绝对四元数先归一化、保留原符号；相对四元数再归一化并选唯一符号（优先令 `w` 非负；`w=0` 时按 `x,y,z` 首个非零分量）。输入必须为实数、有限、末维 23，绝对四元数范数至少 `1e-12`，结果须可表示为有限 float32。支持批量 `(...,23)`，不原地修改输入（`observations.py:59-96`）。`legacy` 直接委托历史拼接函数（`observations.py:99-107`），不能把旧检查点无条件喂入 canonical 状态。

注意：这实现的是传感器语义一致性修复。它不增加信息维度，不等于新增视觉能力，也不能仅凭几何正确性宣称闭环成功率提升。正式冻结 Full 模型只接受 legacy（`robot_can_v2/evaluate.py:41-44,55-65`）。

## 2. 行为克隆策略（BC）

### 2.1 训练样本与网络

`common.py:143-151` 将每条演示长度 `T` 变为 `T-H+1` 个窗口；输入为窗口起点状态，目标为连续 `H` 个七维动作。不足 H 的轨迹不形成窗口。窗口跨演示边界不会拼接。

实际采用 `common.ChunkPolicy`：`common.py:78-108`。默认 H=8、K=5、width=512（`robot_can_v2/train_policy.py:185-194`；现有旧 BC 配置 `policy/bc_chunk_dev_v4/config.json` 亦如此）：

\[
\tilde x=(x-\mu_x)/s_x,\quad h=\mathrm{ReLU}(W_3\mathrm{ReLU}(W_2\mathrm{ReLU}(W_1\tilde x+b_1)+b_2)+b_3).
\]

结构为 `23→512→512→512` 三个 ReLU 隐层，线性输出头大小 `K(1+2H·7)=565`，输出混合权重 logits、每个分量的动作块均值、对角高斯标准差。均值经 tanh 限于 `[-1,1]`，标准差经 softplus 再截断到 `[0.02,1]`。可训练参数数目由层尺寸计算为 827,445（含线性偏置，不含 mean/scale 缓冲区）。这里没有循环历史；`common.py:111-140` 的 GRU 策略是另一可加载架构，当前正式 `architecture=mlp` 不走该分支。旧配置 `history=10` 对 MLP 无效。

### 2.2 目标函数与部署

\[
p_\theta(A\mid x)=\sum_{k=1}^{K}\pi_k(x)\prod_{h=0}^{H-1}\prod_{d=1}^{7}\mathcal N(A_{hd};m_{khd},\sigma_{khd}^2),
\quad
\mathcal L_{BC}=-\frac1{BH7}\sum_b\log p_\theta(A^{(b)}\mid x^{(b)}).
\]

代码 `common.py:100-103` 用 logsumexp 稳定计算，目标是负对数似然，不是普通动作 MSE。部署 `common.py:105-108` 选混合权重 logits 最大的分量均值块，不做随机采样，不保证选择混合密度的全局众数。`train_policy.py:65-74` 的 `dev_action_mse` 只度量选中分量的第一步动作，是辅助指标。

`robot_can_v2/train_policy.py:106-117,137-168`：训练窗口起点状态计算逐维 mean/std，std 下界 0.02；Adam、学习率 `3e-4`、批量 256、15,000 更新、梯度范数裁剪 10，种子 17；每批以独立 CPU `torch.Generator` 有放回采样，配对语义实验可保持相同可训练参数初始化和相同窗口索引。新入口默认每 250 步验证；这次实际配对训练配置为每 1,000 步（各运行 `config.json`），完整 dev 验证不随机抽样。

保存两类权重：`best.pt` 按最小完整 dev NLL；`final.pt` 为固定 15,000 步末状态（`train_policy.py:162-175`）。本次语义对照预注册使用 `final.pt`，不是按闭环成功率或确认结果挑选。已保存 `policies/paired_training_check.json` 表明两臂 best 均在 1,000 步，而最终 dev NLL 已变差；写作不能称最终权重“验证集最优”，未来可预注册早停/最佳权重对照，但不应替换当前已完成确认的固定权重。

## 3. 动作条件世界模型

### 3.1 递推和读出

`world_model.py:7-29`：模型在显式、标准化的 23 维物理状态空间递推，而不是图像潜变量生成器。隐藏宽度 128：

\[
z_0=(x_t-\mu)/s,\quad h_0=\tanh(W_i z_0+b_i),
\]
\[
h_{j+1}=\mathrm{GRUCell}([z_j,u_j],h_j),\qquad z_{j+1}=z_j+0.1\,f_\Delta(h_{j+1}),
\qquad \hat x_{t+j+1}=s\odot z_{j+1}+\mu.
\]

GRU 输入维数 `23+7=30`，delta MLP 为 `128→128(ReLU)→23`。标签 logits 为两维。`future_only=True` 时读出 `23→128(ReLU)→2`，只读预测物理状态；False 兼容旧模型，读入 `[z,h]` 的 151 维。正式 Full 为 True，总参数 87,321；False 架构参数由层尺寸计算为 103,705。此“future_only”只限制读出接口，不代表无隐藏状态，GRU 隐状态依然参与状态预测。

所有未来步均自由递推预测，不在每一步注入真值（没有 teacher forcing）。预测头没有硬性四元数单位化或几何一致性投影，因此未来预测未必严格满足刚体几何；canonical 输入校验不能推导出未来预测必然 canonical。模型也不输出经过校准的认知不确定性。

### 3.2 四个方法分支

| 方法 | 未来状态 | 进展/风险头 | 组内排序损失 |
|---|---|---|---|
| `plain` | 训练 | 训练 | 系数 0 |
| `full` | 训练 | 训练 | 系数 0.2 |
| `direct` | 无 | 当前状态及整块动作直接预测 | 系数 0.2 |
| `prediction_only` | 训练 | 参数冻结且新入口不执行 | 无 |

`robot_can_v2/train_world.py:46-76,219-247` 实现上表。`prediction_only` 为保持 state_dict 兼容仍含 labels 参数，但 `predict()` 自行递推并返回 `(future,None)`，不调用未训练标签头。不能把该头随机输出用于排序或选 checkpoint。

DirectScore (`world_model.py:32-44`) 以 `[标准化当前状态;flatten动作块]` 的 `23+8×7=79` 维输入，MLP `79→width→width→16`，reshape 为 `H×2` logits，无未来状态。新通用入口默认 width=128；**历史多种子参数匹配实验刻意取 width=251、87,364 参数**（`reproduction/multiseed_20261012_01/train_suite.py:79-101` 及对应 direct 配置）。这两个配置不要混淆；正式新确认仅使用冻结 Full 与 BC，不重新比较这四个分支。

### 3.3 分支标签与评分

标签来源 `collect_branches.py:12-17`：

- 进展标签 `p*`：若仿真成功则为 1，否则为 `env.staged_rewards()` 的最大值。它可为 `[0,1]` 连续软标签，不是“是否最终成功”的二分类概率。
- 风险标签 `r*`：分支初始已抓取，且当前抓取阶段奖励 `<0.3`，且尚未成功时记 1。它是特定的失持标签，不是完整碰撞/人体安全风险。
- 这些真值来自训练/验证分支仿真。部署排序通过神经网络预测标签，不查询未来仿真奖励；实验评价仍用环境成功判据。

`world_model.py:47-55` 的候选分数越大越好：

\[
S=\hat p_H+\frac{0.5}{H}\sum_{j=1}^H\hat p_j-\rho\max_j\hat r_j,\quad \rho=0.5,
\]

其中预测概率为 logits 的 sigmoid，真值分数用相同公式替换为软标签。默认分数理论界 `[-0.5,1.5]`，不是以米为单位的物理距离，也不是经校准的成功概率。

### 3.4 损失与训练细节

`robot_can_v2/train_world.py:235-253`：一个批次抽 B 个同起点候选组，并保留每组全部 K 个候选；展开后共 B·K 个动作序列。预测损失按训练起点状态 std 归一化：

\[
\mathcal L_{pred}=\operatorname{mean}_{b,k,j,d}[(\hat x_{bkjd}-x^*_{bkjd})/s_d]^2,
\quad \mathcal L_p=\mathrm{BCEWithLogits}(\ell_p,p^*),\quad \mathcal L_r=\mathrm{BCEWithLogits}(\ell_r,r^*).
\]

同一候选组内仅取 `i<j`、`|S_i^*-S_j^*|>0.025` 的信息对；没有信息对时排序损失为可求导的零（`world_model.py:58-65`）：

\[
\mathcal L_{rank}=\operatorname{mean}_{(i,j)\in M}\mathrm{softplus}\left(-\frac{\mathrm{sign}(S_i^*-S_j^*)(S_i-S_j)}{0.1}\right).
\]

总损失 `Lpred + Lp + 0.5 Lr + λrank Lrank`；Full/Direct 的 `λrank=0.2`，Plain 为 0，prediction_only 只有 Lpred。Direct 的 Lpred=0。

新入口默认 3,000 更新、B=16、width128、Adam lr3e-4、每250步验证、grad clip10、种子17（`train_world.py:224-274,278-296`）。Std 下界0.02；dev按完整分支集计算。Full/Plain/Direct按最小dev平均后悔值保存权重，prediction_only按最小dev归一化状态MSE；相等时保留较早步骤。新入口拒绝复用输出目录、记录数据/源码哈希，并独立 CPU 批次随机流。历史正式 Full 的训练由 `reproduction/multiseed_20261012_01/train_suite.py` 完成，与新通用入口区分。

## 4. 延迟执行与候选决策

### 4.1 真实 FIFO 语义

动作 `a∈[-1,1]^7`：前三维位置增量、接下来三维姿态控制、最后一维夹爪命令。正式环境控制器运行于20Hz，世界坐标 delta 输入；缩放上限平移0.05、姿态0.5（`robot_can_v2/evaluate.py:152-157`）。动作值不是直接以米/秒解释。D=2 相当于两个控制周期（0.1s）排队，而不是推理耗时测量。

对于队列 `Q_t=[q_t^0,...,q_t^{D-1}]` 和新命令 c，时刻t物理执行 `q_t^0`，队列变为 `[q_t^1,...,q_t^{D-1},c]`。模型展开输入为：

\[
U(c)=[q_t^0,\ldots,q_t^{D-1},c,\ldots,c]_{H\text{ steps}}.
\]

`robot_can_v2/evaluate_delay.py:35-54` 保证前缀含实际全部pending命令，H>D；每候选只提交第一条新命令。将 BC 块简单整体平移和真实 FIFO 前缀不是同一概念。

### 4.2 正式恢复因子实验的确定配置

`robot_can_v2/evaluate.py:68-86`：BC 块 `base[0:8]` 中前4个动作成为候选命令，各自重复补齐预测H=8，前缀为实际D2队列；**不是随机扰动6候选**。Full评分后取 `k*=argmax S_k`。fallback为BC块索引2；若 `k*≠2` 且 `S(k*)-S(2)<0.02` 则回退索引2。恰好0.02时不回退。无Full时直接取 `base[2]`。之后恢复控制器可改写proposal，所以模型选中不等于最终submitted。

最终 `submitted` 入队、最旧命令出队，若有扰动则仅执行命令前三维加独立 `N(0,noise²)` 后clip到[-1,1]（`evaluate.py:177-202`）；配对相同demo用 `70000+demo整数ID` 噪声种子。实验noise=0.05是归一化动作空间标准差，不能写“5厘米测量噪声”。20Hz和0.05缩放下，仅线性缩放量级为0.0025m命令增量标准差，但真实机械响应受控制器和动力学影响。

### 4.3 通用无恢复入口的额外分支

`robot_can_v2/evaluate_delay.py` 独立于恢复因子实验：

- `bc` 用 `base[0]`，`bc_shift` 默认用 `base[D]`（`211-215`）。
- WM候选支持 shift 或 random。random为 `collect_branches.py:20-30` 的声明式局部搜索：候选0保持BC块，其余候选位置加一次3维高斯偏移σ0.2、姿态加σ0.03，部分夹爪翻转/前三步打开，再clip；最终模型输入只重复其第一命令。
- 学习评分可score-only，或与物理代价混合：`J=Cphysical-wreadout S`，默认w=0.02。纯prediction_only必须w=0并禁用score-only；DirectScore必须score-only（`evaluate_delay.py:57-82`）。
- 物理代价仅对 `future[:,D:]` 计算，排除候选尚不生效的公共前缀（`26-32,77`）。holding hint由当前观测 `obj_z>0.88` 且EEF物体距离<0.10推断（`221`），不是未来真值抓取。
- 未持物：未来平均EEF物体欧氏距离；持物：未来平均物体目标欧氏距离加 `0.5·ReLU(0.86-最终物体z)`。此为启发式代价，不能与正式Full score-only混写。

## 5. 数据隔离与正式模型来源

`data/splits.json` 将200条演示划为train140、dev30、sealed_test30。`robot_can_v2/observations.py:110-130` 的训练读取器只允许train/dev并返回只读数组。BC当前有效训练窗口15,373、dev3,198（本轮两臂 `policies/*/config.json`）。

正式Full读取 `models/multiseed_20261012_01_full_s{17,23,41}/model.pt`（`evaluate.py:55-65`），由匹配数据多种子训练产生，dataset=`queuepair_20261012_01_real`。训练20个原train父轨迹×8点=160组，dev8个父轨迹×8点=64组；各组4候选、8未来步。单组总32个分支步不是32个独立起点。

重要采集协议 `collect_rollout_branches.py:41-89,123-162`：

1. 从演示起点进行真实D2 BC-shift闭环，保留执行动作前缀与pending队列；
2. 在各父轨迹选8个闭环锚点，取该点BC块前4命令；
3. 每个候选都从演示起点精确重放已执行前缀恢复锚点，验证物理/控制器签名、观测与阶段奖励误差≤1e-7；
4. real保留真实队列，zero将同锚点队列置零；每个候选命令重复执行H步，新命令经FIFO；写入模型的是实际出队执行动作；
5. 验证候选公共D步未来相同、real公共前缀匹配真实闭环、反序执行候选可复现，保留parent/point/queue/commands/状态/标签。

新 `train_world.py:93-125,190-203` 检查形状、有限性、动作范围、标签[0,1]、语义声明、canonical几何和父轨迹train/dev隔离。旧缺省语义按legacy解释；canonical必须显式metadata。注意：若人为提供的NPZ缺少parent元数据，新入口无法完成父轨迹隔离检验，仅检查文件不同；因此论文设计应规定parent必填，不能声称入口对任意NPZ都保证无泄漏。现有正式数据有parent且已检验。

`reproduction/multiseed_20261012_01/data_statistics.json`显示有效排序对较少：train96/960、dev39/384（阈值0.025），风险阳性分支步train287/5120、dev130/2048。这些样本稀疏性应作为局限，不能把所有动作步等价描述为独立风险案例。

## 6. 排序诊断边界

`robot_can_v2/metrics.py:5-48` 区分严格逆序、预测并列（差≤1e-8）、并列计半错、有效真值差>0.025的候选对、平均选择后悔值。分母只用信息对；无信息对返回None。历史 `world_model.rank_metrics` 只把乘积<0计错，会漏掉预测并列；新旧指标应明确命名、不可直接混当同一错误率。Argmax并列取首个候选。

## 7. 建议写入详细设计的已实现/未实现边界

- 已实现：显式状态BC、动作条件多步预测、进展与失持读出、同起点候选排序、真实pending前缀、保守回退、恢复控制器接口、语义版本契约、只读数据与新输出保护。
- 不应声称：视觉输入、真机迁移、在线学习/在线权重自适应、通用碰撞预测、概率校准、通用动力学保证、canonical必然提高成功率、世界模型一定优于BC。
- 论文级详细设计可解释尚未通过确认的候选机制，但必须和正式采用方案分开。当前正式新确认使用冻结legacy BC/Full及v5，canonical配对模型和queue_budget为已执行的候选实验，不应包装成已验证的优势。

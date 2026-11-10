# 19×19 Gomoku AlphaZero — 完整会话记录与接手文档

更新日期：2026-11-09，America/Vancouver。本文由当前会话整理，并在写入前重新读取源代码、checkpoint 和 history.jsonl，复跑测试。它既是整个 session 的报告，也是交给下一个 Codex 窗口的 handoff。

**事实优先级：磁盘上的当前代码和训练产物 > 本文对历史版本的描述 > 较早聊天中的计划。** 本文明确区分历史实验、内存诊断和当前实现，不要把它们混在一起。

## 0. 给下一个窗口：先读这些

1. 项目目录是 `/Users/ericl/Desktop/projects/wuziqi`。目标是理解并实现从随机权重、只知道游戏规则、通过自我对弈学习的 19×19 AlphaZero 风格系统。
2. 当前已经贯通规则、双头网络、PUCT MCTS、自我对弈、replay、优化器、checkpoint、独立评估和终端面板。**流程通了，不代表棋力已经提高。**
3. 当前路线已确定为 **AlphaZero + 标准 PUCT**。用户明确反对把 M1 Pro 误认为不能运行原版算法；不要再次仅因为低预算就劝用户换算法或缩小棋盘。
4. 新训练网络是 32 通道、3 个残差块、全局门控、3×3 Policy Head、全局汇总 Value Head，59,402 参数；保留了旧 CNN 加载分支。
5. 最近完成的真实实验是 `alphazero_5ec884c95696443f8aeab9ce14d667ca`：512 模拟/步，2 轮、4 局、20 次更新，轮 1 对初始模型 2/10 胜，轮 2 为 4/10 胜。它是在最新工程改动前启动的，**没有使用新树复用或单步进度**。
6. 最新工程改动已写入：单步 MCTS/当前步数进度、树复用、默认每 10 轮与最后一轮评估、评估前先保存及原子替换、节点内存与选点优化、可选但默认关闭的推理缓存。
7. 当前有 **13 项持久化回归测试，已重新运行并全部通过**。24 个现有 checkpoint 已逐个严格加载验证。
8. **默认参数仍是流程测试用的小预算。** `python train.py` 默认 1 轮、1 局、8 模拟/步、5 次更新；不是已推荐的 512 配置。`GomokuNetwork()` 默认也仍是旧 `cnn`，用于兼容旧配置；训练函数显式选择新架构。
9. 每次 `train()` 都随机初始化一个新实验。虽然 checkpoint 保存 optimizer，**目前没有恢复训练入口，也没有保存 replay/RNG 完整状态**。
10. 用户最新偏好：允许直接改代码，但要讲清原因，并给具体行号/修改报告。普通答复简短；报告和 handoff 要详细。训练命令通常由用户自己执行。不要无故重启、终止进程、删除实验或自动提交。

### 建议的第一步

- 读本文的“当前代码接口”“实验目录”“未完成事项”。
- 查看最新完成实验的 `history.jsonl`，不要仅根据截图或本会话早期推断还在运行。
- 运行下面的轻量检查，不启动训练：

```bash
conda activate ml
cd /Users/ericl/Desktop/projects/wuziqi
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest test_alphazero -v
```

然后向用户简短说明当前真实状态，并选一个小而具体的下一步。不要重新从头写项目或把已证实的事实重新当猜测。

## 1. 用户目标、限制和协作偏好

### 1.1 不可丢失的项目约束

- 从第一天就使用 **19×19**；不要推荐 7×7/更小棋盘作为先修项目。
- 无禁手、无 pass，连续 **五颗或以上** 同色棋获胜；没有 Renju 三三/四四/长连禁手。
- 随机初始化，只用自我对弈，没有人类棋谱、预训练权重或外部教师。
- 不使用 DQN、Double DQN、Q-learning，不回到旧项目调参。
- 不给活三、活四等手工评估分数。可以构造局面做正确性测试，但不要把手工好坏标签当训练数据。
- 源码直接放项目根目录，不建立 `az/` 包。
- M1 Pro、16 GB RAM 上先做可行的小实验；没有强棋力保证。
- 不自动 commit、push、部署或清理用户文件。
- 用户的重点是理解 RL、搜索、策略/价值学习，不是只拿成品。

### 1.2 本 session 中授权与偏好的演变

初始 handoff 要求用户自己写代码、一步一步教学、只读检查、不自动改文件、不自行训练。随后用户依次明确改变偏好：

1. “你帮我写代码，我自己复制粘贴”。
2. “一次给我一整个文档的 code，并且解释每个 func”。
3. 后来又要求只说具体修改位置，不要每次整文件替换。
4. 再明确要求：“你直接帮我更改，然后最后给一份报告改了哪里，linebyline”。
5. 最近的“改”授权了终端进度、树复用、评估频率和工程性能改进。

**最新授权允许直接改相关文件。** 不要机械复用开头的“只能读”限制来反复问权限。涉及新训练、重启正在跑的进程、删除 checkpoint 或重大算法改路时，仍要遵守当前用户意图和范围。

用户曾明确说“我自己来 run command”。之后实际训练主要由用户在终端执行；agent 做过只读检查、完整自我对弈、评估和诊断，以及明确授权范围内的内存反事实训练，没有把诊断模型保存成正式实验。

沟通上：

- 可以中文解释、保留英文术语。
- 普通回答要简单，用户问“简单回答”时不要写长论文。
- 用户不理解决策时，先讲机制和证据，不能只说“删这行就行”。
- 不要把设计假设说成已证明结论，更不要把“测试通过”说成“学会了五子棋”。
- 不长时间无状态反馈。用户曾因 agent 长时间测试而问“你在干什么”。
- 不需要每一步都询问确认；当前任务内已授权的工作可以继续。

## 2. 环境与文件布局

项目：`/Users/ericl/Desktop/projects/wuziqi`

用户提供环境：MacBook Pro M1 Pro，16 GB RAM，Python 3.11，Conda 环境 `ml`，PyTorch 已安装，VS Code + Codex。实际工具中使用的解释器：

```text
/Users/ericl/anaconda3/envs/ml/bin/python
```

已确认依赖包含 `torch`、`numpy`、`rich`、`tqdm`。当前训练创建 CPU 模型，没有 `.to('mps')`；没有启用 GPU。此前读取新 Python 进程默认 PyTorch CPU 线程数为 4；不可把它当所有启动方式的永久设置。

当前主要文件：

```text
wuziqi/
├── state.py                        游戏状态与规则
├── network.py                      旧 CNN 兼容 + 新全局残差双头网络
├── mcts.py                         PUCT、树复用、可选缓存、进度回调
├── self_play.py                    完整自我对弈与样本生成
├── train.py                        replay、D4 增强、更新、面板、保存、周期评估
├── evaluate.py                     初始/旧模型对战，MCTS 与 policy-only
├── test_alphazero.py                13 个 unittest 回归测试
├── gomoku_env.py                    原项目规则实现；已读，当前新流程用 state.py
├── test_web_app.py                  旧 web 测试，未纳入本次验收
├── web_app/                        旧 web 项目，可能依赖已删 DQN，不是当前重点
├── checkpoints/                    独立训练实验目录
├── NETWORK_UPGRADE_REPORT.md        网络升级逐行报告与完整 diff
├── PERFORMANCE_UPGRADE_REPORT.md    工程改进逐行报告、测试、微基准与完整 diff
├── SESSION_HANDOFF_2026-11-09.md     本文
├── .vscode/                        Conda/编辑器配置
└── __pycache__/                    用户正常执行 Python 产生的缓存
```

初始上下文提过旧 DQN `.pt`，当前根目录清单中未见；不要假定是本 agent 删除的，也不要尝试找回旧模型用于训练。没有本会话自动提交或创建 git 历史的记录。

原始 handoff 来源（仅供追溯）：
`/Users/ericl/.codex/attachments/c10cd4c8-bd99-4f63-bff6-404441a86fb4/Pasted text.txt`。

## 3. 会话整体时间线

### 3.1 从规则文件起步

初始新 AlphaZero 文件为空。先读了 `gomoku_env.py`：已有 NumPy 棋盘、合法动作、落子、四方向五连、满盘平局；但没有完整 clone、观察编码、独立 winner，且终局不切换玩家、终局 legal_actions 仍可能给空位。

随后按小步骤提供并检查 `state.py`：初始化 → 合法动作 → clone → 编码 → 用户要求整文件后补全落子、胜负、终局价值。

### 3.2 小 CNN 和梯度问题

第一版：三层 3×3/32 通道共享 CNN，1×1 Policy Head，卷积后压成单通道的 Value Head。

发现 Value Head 单通道卷积后紧接 ReLU，在某次初始化和真实编码输入下，361 个输出全部为负（约 -0.112～-0.082），经 ReLU 全部归零，价值损失无法传回共享主干。移除了这一处 ReLU，保留后续全连接层的 ReLU。

用户用随机输入测到非零梯度；解释了这与问题不矛盾：它依赖初始化和输入，不是每次触发。还解释了 `backward()` 累加梯度、测试前清空梯度，以及测试代码必须放 `if __name__ == '__main__'`，避免 import 时执行。用户曾把 Python 贴进 zsh 得到语法错误，后改为运行文件。

### 3.3 基础 MCTS、自我对弈、训练、评估贯通

实现并验证标准 PUCT、扩展、神经网络叶评估、真实终局评估、符号翻转回传、根访问分布；然后实现完整自我对弈和结果标签。

训练使用固定网络副本收集完整游戏，bounded replay，两个损失，Adam，checkpoint。用户运行了 1 局/5 次更新，随后实现独立评估。

早期自我对弈测试有 97 步黑胜；用户实际训练第一局为 170 步白胜。这不是矛盾：直接测试 `seed=0` 与训练 RNG 派生的 game seed 不同。

### 3.4 进度显示多次改进

最初只有 print，用户要求知道更新次数；加入 tqdm。每局和每轮 `tqdm.write` 造成滚屏，用户要求不换行；再讨论单行刷新。用户又希望同时看 loss/win rate 和进度，最终改为 Rich Live 固定面板。

明确区分自我对弈颜色胜率与对固定初始模型的评估胜率。训练 loss 不等于棋力。

### 3.5 发现训练误差下降但评估退步

旧 CNN/8 模拟的 50 局、100 更新实验中，总 loss 下降，末轮对初始模型却 0/20 胜。换种子复测仍差：MCTS 2/40 胜、policy-only 0/40 胜。

用户要求只 inspect、不改文件、其他可自由操作。进行了完整诊断和内存反事实实验，见后文。

### 3.6 感受野、全局信息与网络升级

确认旧策略感受野 7×7 看不到部分直接五连；先增加可选 3×3 Policy Head，使局部感受野达到 9×9，兼容旧权重。

用户进一步追问全局棋盘，指出 9×9 并非完整网络目标；讨论了 ResNet 与全局机制。用户随后授权直接改文件并给逐行报告，完成当前 `resnet_global`。

### 3.7 原版 AlphaZero 与 M1 的路线澄清

尝试不同 PUCT 探索系数、未访问节点估值、搜索预算，并查阅 Gumbel AlphaZero 论文。agent 曾倾向建议低预算 Gumbel，用户质疑“M1 不能用原版吗”。已明确纠正：**M1 可以运行原版算法，低预算问题不是硬件不支持**。

用户最终表示“反正就用最好的算法”，当前共识是保留标准 PUCT，建立可靠基线，以实际训练/对战检验，而不是声称存在一个处处最优算法或自动换路线。

### 3.8 512 模拟实验与慢速原因

完成 128/256/512 模拟的完整自我对弈基准，然后由用户启动 2 轮/4 局/20 更新的新网络实验。用户因第一局面板不动询问是否运行；只读确认 CPU 仍在计算，没有重启。

解释 512 是每次真实落子前的总搜索模拟预算，不是对局数、每个动作各 512 次或梯度更新次数。第一版 Rich 面板只在整局完成后刷新，因此不能显示局内工作。截图后来已处于评估阶段，而不是训练卡住。

用户随后要求工程优化，直接实施本次最新树复用与进度改动。

## 4. 当前规则、编码和价值约定

`GomokuState`：

- board：19×19，NumPy int8；0 空，1 黑，-1 白。
- current_player：即将落子的玩家；黑先。
- done：是否终局；winner：1 黑胜，-1 白胜，0 平或未结束（必须结合 done）。
- action 编号：`row * 19 + col`，合法范围 0～360；`divmod(action,19)` 解码。
- 非整数、bool、越界、占用位置、终局后落子会抛 ValueError。
- step 原地修改，没有返回 `(obs,reward,done)`。
- 每次成功落子后都切换 current_player，**包括终局**。
- legal_actions 在终局返回空数组。
- terminal_value 仅允许终局调用：`float(winner * current_player)`。

例：黑刚五连，终局 winner=1/current_player=-1，因此 terminal_value=-1；而最后落子前黑方样本的标签是 +1。不要因此误判符号错误。

encode 返回 float32 `(3,19,19)`：

1. 当前玩家棋子。
2. 对手棋子。
3. 当前玩家是黑棋则全 1，否则全 0。

clone 独立复制数组和 metadata。测试过四方向、边界、跨两端连线、长连、非法动作、clone、观察编码，以及一个完整 361 步合法平局。

## 5. 当前网络与兼容策略

`GomokuNetwork(channels=32, value_hidden=64, policy_kernel=1, architecture='cnn', residual_blocks=3)`。

默认旧 `cnn` 是兼容机制，**不是当前 train() 使用的架构**。旧配置没有 architecture/policy_kernel 字段时，仍严格重建旧网络。当前新训练显式保存：

```python
model_config = {
    'channels': 32,
    'value_hidden': 64,
    'policy_kernel': 3,
    'architecture': 'resnet_global',
    'residual_blocks': 3,
}
```

新结构：

```text
(B,3,19,19)
  → 输入 3×3 卷积 3→32 + ReLU
  → 3 个残差块，每块 Conv3×3 → ReLU → Conv3×3，输出 ReLU(x+F(x))
  → GlobalContext
      全盘平均池化 → 32→8 的 1×1 卷积 → ReLU → 8→32 → sigmoid
      每个通道乘以由整盘摘要决定的门控系数
  ├─ Policy：3×3 卷积 32→1 → flatten → (B,361) 原始 logits
  └─ Value：全盘平均池化 → flatten → Linear32→64 → ReLU → Linear64→1 → tanh
```

没有 BatchNorm；这是小型 ResNet-style + SE-style 门控，不要描述成完整复刻论文大型 AlphaZero 网络。

局部路径总共 8 层 3×3 卷积，理论局部感受野 17×17；全局依赖来自额外门控通路。整盘摘要不保留所有精确位置关系，也不保证 D4 不变性；保留训练数据增强。

网络能力验证：

- 3 个随机种子下，两头都有有限梯度传回主干与 gate。
- 固定诊断权重，左上两个动作的 logit 差对右下输入的梯度约 -4.06247e-8；禁用 gate 后为 0。检查相对 logit 是为了排除“只增加会被 softmax 抵消的公共常数”。
- 原先 22 个历史 checkpoint 的初始化/推理与旧实现逐项完全一致；本报告生成前所有现存 24 个 checkpoint 都严格加载成功。
- 新配置+state_dict 在内存序列化往返成功。
- 这些是结构/正确性验证，不是已经学会远程防守的证据。

masked_policy 检查形状、合法 mask；非法 logits 设 -inf 后 softmax，没有合法动作时报错。网络 forward 本身返回 raw logits。

## 6. 当前 MCTS 接口与重要语义

```python
MCTS(model, simulations=32, c_puct=1.5,
     dirichlet_alpha=0.03, noise_fraction=0.25,
     seed=None, cache_size=0)

policy, root = searcher.search(state, add_noise=False,
                               progress_callback=None)
searcher.advance(action, state_after_real_move)
searcher.reset()
```

TreeNode 用 slots，字段为 base_prior、prior、visit_count、value_sum、children。q 为该节点当前玩家视角的 value_sum/N；未访问返回 0。

选择仍为：

```text
score(parent→child) = -child.q
                     + c_puct * child.prior * sqrt(parent.N+1)/(1+child.N)
```

负号是父子玩家视角转换，backup 每向上一层再翻转 value。已检查终局/标签，没有发现颠倒。

### 最新加入的树复用

- searcher 保存匹配当前棋盘的 root；不匹配就新建。
- 真正落子后 advance 用 clone+step 重放核对状态，保留对应子树；错位、不合法或未展开分支则丢弃树。
- 自我对弈单网络使用一个 searcher；双模型评估双方各有独立 searcher，每次真实落子后双方都推进。
- 每步仍执行严格指定数量的 **新模拟**。复用树的 N 可能已大于 0，故 root.N、子节点累计访问次数之和不再必须等于本步 budget。
- 目标策略按保留后的累计子节点访问次数归一化。不能把这点改回“累计 N 必须等于 512”的旧测试。
- 每次根搜索都先恢复子节点的 base_prior，再混入新噪声；不会把前一次 root noise 反复叠加。
- last_search_stats 包含 reused_visits、network_calls、cache_hits；这些是当前 search 的统计，不是整局累计。

### 缓存和模型生命周期

- 有界 LRU 可以通过 cache_size 开启，但默认 **0，关闭**；真实小对照没有缓存命中，未据此宣称加速。
- key 包含棋盘大小、完整棋盘、当前玩家、done、winner。
- 缓存和树要求模型权重固定；外部改权重后要 reset()。当前生产流程每局使用固定快照/评估网络，没有跨 optimizer 更新复用同一 searcher。
- 没有自动检测任意外部权重更新；这是 API 使用前提，后续修改不能忽略。

### 进度回调

search 的 callback 接收 `(completed,total)`，报告 0 和结束，中间约每 0.2 秒更新。不是每次 simulation 都刷新 Rich。

play_game/play_match 的 callback 接收事件字典：move、player、simulations_done、simulations_total、elapsed_seconds、reused_visits、network_calls、cache_hits。

回调不参与随机抽样或选择分数；已验证不改变对应搜索结果。

## 7. 自我对弈、数据增强、损失和训练

### self_play.py

`play_game(model, simulations=8, exploration_moves=30, seed=None, progress_callback=None)` 返回 `(samples, final_state)`。

- 同一模型控制两色，每步都加根 Dirichlet 噪声。
- 前 30 步按 root 访问分布抽样；之后取最大访问数，并列随机。
- 保存落子前 observation、完整访问分布、当时 player。
- 完整游戏后赋标签 `winner * recorded_player`。
- select_action 会以 float64 再归一化，避免 numpy 抽样因浮点误差失败。
- 不截断棋局、不执行梯度、不保存数据文件。

### ReplayBuffer 与 D4 增强

- deque(maxlen=10000)，超容量丢旧样本。
- sample 无放回抽 batch。
- 每个样本随机旋转 0/90/180/270 度，再可选水平镜像。
- observation 的空间轴和 policy.reshape(19,19) 严格同步变换；outcome 不变，身份平面不变。
- copy 保证不改 replay 原数据，也避免负 stride 传入 torch。
- augmentation_rng = default_rng(seed+200000)，与对局/抽样 rng 分离；评估 rng 使用 seed+100000。
- 所有八种变换曾用独立坐标公式验证，合法概率、总和、标签、输入不变性均通过。该详细增强测试当时在内存执行，尚未全部固化进当前 13 个 unittest。

### train_step

```python
log_probabilities = torch.log_softmax(logits, dim=1)
policy_loss = -(target_policy * log_probabilities).sum(1).mean()
value_loss = mse(predicted_value, final_outcome)
loss = policy_loss + value_loss
```

zero_grad → forward → backward → Adam.step。两种损失权重均为 1；没有 DQN bootstrapping、target Q 网络或手工奖励。训练用未屏蔽的 log_softmax，非法动作 target=0；实际落子推理仍屏蔽非法动作。不要轻率把它改为 0×-inf 的不稳定实现。

### train() 当前默认值

| 参数 | 默认值 |
|---|---:|
| iterations | 1 |
| games_per_iteration | 1 |
| simulations | 8 |
| updates_per_iteration | 5 |
| batch_size | 32 |
| buffer_capacity | 10000 |
| learning_rate | 0.001 |
| seed | 0 |
| eval_every | 10 |
| eval_games | 20 |
| cpu_threads | None，保留 PyTorch 设置 |

每轮固定 deepcopy snapshot 收集完整对局，样本加入 replay，然后更新原网络。batch 不足 32 时使用现有样本数量。

每轮更新后先保存，再按周期/最后一轮评估。checkpoint 先写 .pt.tmp，再 replace；写入失败时保留原文件并清理 tmp。评估完成后原子补充结果。history.jsonl 记录完整完成的轮次；若评估中断，可能已有 checkpoint，但该轮 history 尚未写完。

checkpoint 保存 model_config、model_state_dict、optimizer_state_dict、iteration、updates_completed、metrics、evaluation、seed、simulations，新版本还保存 runtime_config（线程、树复用、缓存、频率）。

**不包含完整 replay、RNG 状态、训练恢复入口。每次 train() 都是新实验。** 不要仅因为保存了 optimizer 就声称支持断点续训。

## 8. 评估协议和面板

evaluate.py 支持加载 candidate checkpoint，默认对手是用 checkpoint.seed 与相同 model_config 重新初始化的网络；可用 --opponent 指定旧 checkpoint。

- MCTS 模式：双方相同新模拟预算，关闭 root noise，选择最大 root 访问动作，并列随机。
- policy 模式：直接取合法策略最大概率，并列随机；Value 不参与。
- 交替 candidate 执黑/执白，games 必须是至少 2 的偶数。
- result = winner * candidate_color，胜+1、负-1、平0。
- win_rate=W/N，score=(W+0.5D)/N，按总体和颜色分开统计。
- 最新默认启用树复用；`--no-tree-reuse` 可恢复每步新建树的旧协议。
- 小局数和重复棋路不能给可靠棋力结论。相同模型的 policy-only 常只有两条确定性路线，即使 seed 不同也可能完全重复。

Rich Live 面板当前显示：Round、Self-play、Updates、当前手 MCTS、最近评估进度；当前 move/玩家、局内耗时、CPU 线程、复用/NN/缓存统计、batch loss、round mean loss、replay、自我对弈颜色结果、最近评估的轮次和样本量、最后 checkpoint。

自我对弈黑胜率不是“模型对外胜率”。最近一次评估结果在其他轮保留原轮次标签，不是假装实时重新评估。Round 进度表示收集/更新/评估/保存整轮完成，因此中途 Updates=50% 而 Round=0% 可以正常。

## 9. 实验目录清单：当前 5 个目录、24 个 checkpoint

写报告时目录总产物约 13,150,594 字节（含历史日志）。以下均未删除。

| 完整目录名 | 内容与用途 |
|---|---|
| alphazero_2c886fca6482420ebac3db943510cdb7 | 旧 CNN，1 个 checkpoint，5 次更新；早版没有 updates_completed 字段，但 Adam step=5。 |
| alphazero_590518dcdb0243ea85c278ae84289115 | 旧 CNN，1 个 checkpoint，5 次更新；曾做早期独立评估。权重与 2c886… 完全相同。 |
| alphazero_8fa2de7d9a7349d3863c2fd5797a17f8 | 旧 CNN，10 个 checkpoint，50 局/100 更新；没有 history.jsonl 的早期日志版。 |
| alphazero_0475761549914b0d8203a890363e7db4 | 同样旧 CNN 的 50 局/100 更新，有完整 history、R5/R10 评估；每一轮权重与 8fa… 对应轮逐项相同，已比较确认。 |
| alphazero_5ec884c95696443f8aeab9ce14d667ca | 新 resnet_global + policy_kernel3 + D4，512 模拟/步，2 个 checkpoint，4 局/20 更新；最新完成实验。 |

每个 iteration_NNN.pt 都是独立可加载的模型快照，不依赖前面 iteration 文件。文件夹名随机 UUID 只是一次运行的唯一标识，没有棋力含义。最新不一定最强。

用户问过是否全有用：建议保留早期对照和带日志的新实验，重复目录可作为未来清理候选；**未获明确删除授权，不要删除**。

## 10. 真实训练与对战结果

### 10.1 早期 1 局、5 次更新

旧架构、8 模拟/步、170 步白胜、170 样本。

- mean total loss：6.8870168686
- mean policy loss：5.8879321098
- mean value loss：0.9990847111
- 用户面板最后 batch loss 6.8288，与平均值不同并不矛盾。
- 检查过 CNN 主干、Policy、Value 参数确实变动，Adam step=5，checkpoint 可加载。

对应 590518… 的评估：

| 协议 | 结果 |
|---|---|
| 4 局 MCTS，8 模拟，seed12345 | 3胜1负，75%；样本太少。 |
| 4 局 policy-only | 2胜2负；四局全部黑棋获胜。 |
| 40 局 MCTS，8 模拟，seed54321 | 19胜21负，47.5%；candidate 黑13/20，白6/20。未观察到明确改善。 |

### 10.2 旧 CNN 的 50 局、100 更新

命令：

```bash
python -c "from train import train; train(iterations=10, games_per_iteration=5, simulations=8, updates_per_iteration=10)"
```

当时运行版本不同于现在，周期评估在 R5/R10，搜索每步新建树，没有 D4 或全局残差网络。相同 seed 的两次实验产生完全相同对应权重。

累计样本 6262，黑26胜、白24胜，无平局。

| 轮次 | 样本池 | Policy loss | Value loss | Total loss | 当轮评估 |
|---:|---:|---:|---:|---:|---|
| 1 | 636 | 5.8878 | 1.0125 | 6.9003 | — |
| 2 | 1199 | 5.8844 | 0.9923 | 6.8767 | — |
| 3 | 1837 | 5.8811 | 0.9535 | 6.8346 | — |
| 4 | 2431 | 5.8786 | 1.0007 | 6.8794 | — |
| 5 | 3106 | 5.8696 | 0.9232 | 6.7928 | 9胜11负，45% |
| 6 | 3784 | 5.8578 | 0.8447 | 6.7024 | — |
| 7 | 4251 | 5.8438 | 0.6145 | 6.4583 | — |
| 8 | 5011 | 5.8320 | 0.5332 | 6.3652 | — |
| 9 | 5599 | 5.8065 | 0.3222 | 6.1287 | — |
| 10 | 6262 | 5.7890 | 0.3162 | 6.1052 | 0胜20负，0% |

末轮换种子 seed54321，40 局/模式复测：

| 模式 | 总体 | 执黑 | 执白 |
|---|---|---|---|
| MCTS，8 模拟 | 2胜38负，5% | 2/20 胜 | 0/20 胜 |
| policy-only | 0胜40负 | 0/20 胜 | 0/20 胜 |

实际棋谱验证：policy-only 执黑重复同一条 12 步路线，执白重复另一条 11 步路线。因此不是 40 条不同路线的独立失败证据，但当前确定性对战确实差。

### 10.3 最新已完成的新网络、512 模拟实验

用户执行：

```bash
python -c "from train import train; train(iterations=2, games_per_iteration=2, simulations=512, updates_per_iteration=10, eval_every=1, eval_games=10)"
```

目录 5ec884…，本报告生成时磁盘已有两轮完整记录：

| 轮次 | 完成对局 | 样本池 | 更新数 | Policy loss | Value loss | Total loss | 对同架构初始模型 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2 | 177 | 10 | 5.88246865 | 0.98211936 | 6.86458807 | 2胜8负，20% |
| 2 | 4 | 326 | 20 | 5.87653689 | 0.95345011 | 6.82998705 | 4胜6负，40% |

两轮累计自我对弈黑2胜、白2胜。没有足够证据宣布更强，也不能直接把40%与旧网络5%相减当作改善幅度：架构、样本数量、预算、数据增强、评估种子/对手架构都不同。

**该实验在最新工程优化之前启动。** checkpoint 含 resnet_global 配置，但没有 runtime_config；结合启动时序可确认它没有新树复用和单步回调。两个 checkpoint 已存在、两行 history 已完成，不能继续仅凭老截图声称它还卡在第 1 轮；本次未专门确认终端进程是否已退出。

## 11. 诊断证据：已证实与不能过度推断的内容

### 11.1 价值符号没有发现错误

检查了两色获胜、终局返回、父子 Q 转换、逐层 backup、历史玩家标签。黑刚胜时 child 当前玩家白，terminal=-1，parent 黑得+1，逻辑一致。不要为了修退步直接再翻一次符号。

### 11.2 感受野的可证明限制

三层 3×3 共享 CNN + 1×1 Policy 的局部感受野是7×7，半径3；端点补五连的第四颗关键棋子可在距离4。

两个合法构造局面中，把距候选(9,9)四格的黑子从(9,5)挪到(0,0)，其他近邻不变：一处落子成五，一处仅成四。旧网络对应 logit 完全一样，为0.1920738816，相对另一个远落点的 logit 差也相同。这是网络信息通路限制，不是多训练几次能解决。

3×3 Policy Head 使局部感受野9×9，固定诊断权重可让该变化影响分数；**不代表随机/已训练网络一定学会判断**。

如果仅靠普通步长1、无 dilation 的3×3卷积，总9层得到19×19窗口，但角落仍看不到对角；总18层的半径18才保证任意落点能覆盖最远角。当前没有堆18层，而是局部17×17+全局门控。

### 11.3 PUCT 覆盖不足是真实的，但不是公式实现 bug 的证明

旧训练末轮模型，在唯一立即获胜(9,8)，action179 的局面中，其策略排名51；另一必要防守局面，关键点排名67。

旧公式下立即获胜测试：8、64、128模拟都只访问一个错误根动作。128时仍未试必胜点。64模拟末期分数约为：

| 根动作 | 父视角 Q | U | 总分 |
|---|---:|---:|---:|
| 错误动作 | 0.502083 | 0.000790 | 0.502873 |
| 未访问必胜动作 | 0 | 0.038028 | 0.038028 |

低先验乘以探索项，在小预算和不可靠价值下可被乐观 Q 压住。旧训练 R10 收集过程平均每手只访问3.745个根动作，约31.4%位置只访问一个。

尝试过父节点均值作为未访问估值、更大的 c_puct 和128/512/2048预算：并非稳定修复。例：c=25、512次能在特定立即获胜测试优先正确动作，但必要防守到2048次仍未优先正确动作。父均值方案还会恶化某些防守覆盖。**这些方案都没有写入生产选择公式。**

新网络随机 seed0 某个立即获胜测试8模拟就成功，是该初始化恰好搜索到了它，不能当作已学习。必要防守仍可能在512次中仅访问一次并不优先。

### 11.4 策略/价值来源交换：内存实验，不是保存的新模型

固定原始初始对手、同40个seed/颜色、8模拟，结果如下：

| Candidate 组合 | 胜/平/负 |
|---|---|
| 初始P + 初始V | 21/0/19 |
| 训练后P + 训练后V | 2/0/38 |
| 训练后P + 初始V | 3/0/37 |
| 初始P + 训练后V | 26/0/14 |
| 训练后P + 恒0 V | 0/0/40 |
| 初始P + 恒0 V | 14/0/26 |

结论限于这些模型/协议：只换回初始 Value 救不回训练后策略，不能把问题全怪 Value；也不能根据26/40声称某种混合结构普遍最强。

### 11.5 原训练完全复现、梯度反事实

用户明确允许只读 inspect、其他操作自由时，在内存重建了每轮原快照生成的50局与随机批次，同时训练反事实副本。原版 both 分支的每一轮参数与磁盘 checkpoint 最大差为0，确认重建精确。

固定原始数据/批次、不让各反事实生成自己的新训练分布：

| 损失路径 | 两条固定 policy-only 路线（黑/白） | 40局 MCTS 胜/平/负 |
|---|---|---|
| 原 policy+value | 负/负 | 2/0/38 |
| 只训练 policy | 胜/胜 | 17/0/23 |
| 只训练 value（共享主干仍变） | 负/负 | 19/0/21 |
| value 对主干 detach，policy 正常更新主干 | 胜/胜 | 11/0/29 |

这支持共享主干训练改变策略表现的解释，但并不证明共享架构本身错误，或应该删 Value/永久 detach。当前生产代码仍让两头共同反传。

一个最后 batch 的主干梯度范数：policy≈0.5893，value≈0.5930，cosine≈-0.0163；单个 batch 不足以断言两任务一直冲突或某损失压倒一切。

### 11.6 方向/位置泛化差

旧最终网络在抽查的原 replay 样本上，Value MSE约0.2822；只旋转输入、胜负标签不变：90°约1.4532，180°约1.3279，270°约1.7871。180°前后平均绝对预测差约0.7936。

这不是独立外部验证集；它揭示相同棋局不同方向的敏感性。最终网络对第10轮刚收集样本的MSE约1.2147，对部分早轮样本低至0.07～0.13；混合训练均值会掩盖新局面的差表现。

因此引入同步D4增强、全局信息通路，并减少固定坐标专用 Value FC 的自由度。它们是有依据的设计改进，效果仍需新训练验证。

## 12. 性能基准和慢速原因

### 12.1 工程优化前的完整局基准

新全局网络、同seed初始化/对局、CPU2线程、每档一局，无参数更新：

| 新模拟/手 | 步数 | 整局秒 | 秒/手 | 平均根动作覆盖数 |
|---:|---:|---:|---:|---:|
| 128 | 86 | 15.877 | 0.1846 | 15.49 |
| 256 | 86 | 30.931 | 0.3597 | 21.08 |
| 512 | 123 | 86.862 | 0.7062 | 79.25 |

三档该局均未出现只访问一个根动作；这不是多局强度评估。不同预算会改变棋路和棋长，不能只比较总时间。

### 12.2 用户看到“没跑”的真实原因

第一版面板在完整一局结束前不改变对局计数，2局采集完才开始更新，因此 loss=--、buffer=0 可以正常。历史只读CPU采样从累计1:58升到2:09，使用率约118～127%；后来4:57墙钟时累计CPU5:53，约122%，确认在算。历史 PID86080 仅用于当时检查，**不要据旧PID对新进程发信号**。

截图中后来已经2/4自我对弈、10/20更新、评估4/10。慢的主要部分是每步512次串行搜索、通常一次一个叶子batch1推理、Python树操作，而非那10次backprop。

那组配置仅4局收集，却20局评估，计算占比偏重。这是为什么现在降低默认周期，并让终局评估仍可保留。

### 12.3 最新工程优化的实测与限制

- 361个空节点（含各自children空字典）测量：80,864→49,096字节，约减少39.3%。不是整个进程内存降39%。
- 同一个固定选点微基准：123.378→80.926微秒，约减少34.4%。不是整个训练提速34%。
- slots/平方根优化前，六个相同位置每手新增512模拟：原fresh树4.2577秒/3078次NN；复用无缓存4.5387秒/3073次NN/保留2290访问；复用有缓存4.5982秒/0cache hit。没有据此宣称树复用让墙钟更快。
- 缓存开关下策略逐项相同，但该小对照没收益，因此默认关闭。
- CPU线程1/2/4的batch1前向分别约0.769/0.825/0.758毫秒，batch32约30.707/30.455/31.790毫秒，没有稳定大优势；默认不强行设置。
- no_grad/inference_mode的另一次小测试约0.7409/0.7353毫秒，差异小，未更换推理模式或宣称提速。
- MPS/GPU、批量叶子推理、分布式/多进程均未加入。不要假定GPU一定更快；应单独测batch1开销和搜索调度正确性。

## 13. 最新测试清单和验证边界

当前 test_alphazero.py 有13项：

1. 选点优化与参考PUCT公式一致。
2. 树复用保留统计并精确增加新预算。
3. advance状态错位时丢树。
4. 没有advance但位置变化时也丢树。
5. 根噪声从base重新混合，关闭后恢复。
6. cache按玩家区分、容量受限、reset失效。
7. cache开关不改变策略。
8. callback不改变策略，首尾进度正确。
9. 两色终局符号、终局不调用网络。
10. 旧评估模式可关闭树复用。
11. 完整self-play/eval的callback、样本和标签。
12. 默认评估频率与先保存后评估顺序。
13. 模拟保存失败后旧文件保留、tmp清理。

生成本报告时重新运行：**13/13通过，约0.374秒**。此外24个当前checkpoint全部strict load通过。

注意：持久化测试大量使用CountingNetwork；训练组织测试mock了train_step/optimizer，没有执行真实参数更新。原子保存测试只写自动清理的临时目录。它们不等同于新工程版本已完成训练或证明棋力。

最新真实网络工程集成检查完成118步白胜、约1.397秒、236个进度事件（8模拟/手），逐条标签/概率/输入检查通过，权重不变。更早网络升级检查114步白胜、约0.978秒是树复用前的另一次测试，不要把不同路线时长当同条件加速证据。

新代码fresh树与旧代码在四个真实固定位置、16模拟、同噪声seed的策略与统计逐项相同。当前测试未覆盖旧web_app，也未做最新工程版本的正式训练/大样本独立强度评估。

## 14. 下一个窗口的具体接手顺序

### A. 先确认状态，不重复或中断现有实验

读取5ec884…的history和两个checkpoint。当前已完成计划中的两轮产物。若用户说仍有进程，只读确认新PID；不要按本文旧PID操作，不要擅自kill或重启。

### B. 明确用户接下来想学习什么

优先一个小步骤：解释512新模拟与保留访问的区别；读最新326样本/40%评估的含义；或复测最新checkpoint。不要在用户仅问概念时启动长实验。

### C. 若需要复测旧512运行，先保持旧协议

可由用户执行：

```bash
python evaluate.py checkpoints/alphazero_5ec884c95696443f8aeab9ce14d667ca/iteration_002.pt --games 10 --simulations 512 --mode mcts --seed 100000 --no-tree-reuse
```

训练内评估使用seed+100000生成游戏种子；上面尽量匹配历史seed和fresh-tree协议。跨硬件/线程/浮点不承诺逐步完全复现。之后用新seed和更多局数作独立复测；启用/关闭树复用应分开记录。

### D. 若开始新工程版本的小训练

先明确不续训旧模型，而是新随机初始化实验。等用户准备好，由用户运行或在明确授权后运行：

```bash
python -c "from train import train; train(iterations=2, games_per_iteration=2, simulations=512, updates_per_iteration=10, eval_games=10)"
```

省略eval_every=1，当前默认每10轮加最后一轮，所以这个2轮实验仅最后评估10局。显式传1仍会每轮评估；代码不会覆盖显式选择。不要保证多少分钟或棋力改善。

### E. 后续值得做，但尚未完成

- 新树复用版本真正训练后的独立棋力与计算预算对照。
- 数据收集量/优化步数/搜索预算的受控实验，避免一次同时改变所有参数。
- 分色结果、置信区间/局面多样性、避免确定性路线伪造样本量。
- 固化真实网络的全局通路、D4同步变换、旧架构兼容等此前内存检查为持久化回归测试（按需要，不机械写重复测试）。
- 网络推理与Python树操作的真实profile；必要时测试批量叶子推理或MPS，而非仅凭GPU名义启用。
- 推理缓存仅在可证明输出一致且实测有效的工作负载上启用。
- 若用户要求，设计明确的完整恢复训练协议：replay、RNG、snapshot/iteration状态、optimizer和架构版本。当前未实现。
- 更强对手/旧checkpoint之间的评估。不要只看输赢初始模型或训练loss。
- web集成仍是后续事项，不要现在挪用旧DQN代码。

## 15. 本 session 中需要记住的纠正

- “MCTS通过基础测试”只证明机械流程，并不保证访问分布提供更好的学习目标。
- 旧7×7感受野是实际漏查的结构限制；9×9只是局部最低要求，当前已升级全局路径。
- 没有证据表明奖励符号写反，不应靠乱翻符号解决退步。
- 不应该把低模拟困难说成M1不能使用原版AlphaZero。用户已要求继续原路线。
- 不能把更大模拟数、换Gumbel、ResNet、global pooling或D4增强称为保证强棋力的开关。
- 没有证据支持缓存或更多CPU线程必定更快，默认选择反映了实测而非直觉。
- “旧 checkpoint 有optimizer”不等于“可以续训”；“每轮最新”不等于“最强”。
- 用户希望agent直接做已授权修改、给清晰证据和报告，避免反复索取同一权限；也不希望agent擅自替他启动训练。

## 16. 已有详细报告与参考论文

两个现有报告含对应版本的逐行diff和语义说明，适合追溯；其行号是各自写作时快照，后续文件增长会使部分旧行号漂移，接手时用函数名搜索：

- [网络升级报告](/Users/ericl/Desktop/projects/wuziqi/NETWORK_UPGRADE_REPORT.md)
- [工程改进报告](/Users/ericl/Desktop/projects/wuziqi/PERFORMANCE_UPGRADE_REPORT.md)

曾查阅的一手资料（当前没有引入Gumbel）：

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)：残差学习帮助深网络优化，不自动提供全局空间覆盖。
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)：全局汇总与通道调制的参考。
- [Policy Improvement by Planning with Gumbel](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf)：小模拟相对大动作空间时的策略改进讨论；其保证有条件，不能简化成“换算法必胜”。

## 17. 一段可直接贴给下一个窗口的开场白

> 请先读取 `/Users/ericl/Desktop/projects/wuziqi/SESSION_HANDOFF_2026-11-09.md`，再检查当前代码和最新实验记录。我们做的是19×19、随机初始化、纯自我对弈的AlphaZero+PUCT，已完成规则、全局残差双头网络、D4增强、训练/评估和树复用/逐步面板。13项回归通过。最新完成的5ec884…实验为512模拟、4局、20更新，最后对初始模型4胜6负，使用的是工程优化前的fresh-tree协议，不能宣布变强。最新代码还没有正式训练验证。可以直接做相关代码改进，但先讲清机制，普通回复简短，修改后给行号报告；训练通常由我自己执行，不要擅自启动、终止进程或删checkpoint。先向我说明当前状态，再选一个具体下一步。

---

下面的环境版本与源文件指纹由最终只读检查补充，用于判断交接后代码是否发生变化；不是固定未来工具环境的要求。

## 18. 最终检查指纹

解释器：`/Users/ericl/anaconda3/envs/ml/bin/python`；Python `3.11.17`。

| 包 | 本次检查版本 |
|---|---|
| torch | 2.14.1 |
| numpy | 2.4.6 |
| rich | 15.0.0 |
| tqdm | 4.70.1 |

| 源文件 | 行数 | SHA-256 |
|---|---:|---|
| state.py | 106 | `0286956fb257adab5d5d869dad72be75b54a61d8758e2d9a6e8222a31ce5a559` |
| network.py | 151 | `ffd53c39d47f68f184511dae8e64274a86c51d89fa7d4955b1a0a2aecfe63be1` |
| mcts.py | 251 | `da167810e423d63d452c4bc6e66e9c7716eef2a013081b547ba46102fe3b5714` |
| self_play.py | 135 | `09f80d5447a92fef6a7177dc88d43ee02354dbc1199882a65a9f631f258b7a0d` |
| train.py | 503 | `0a91292ba7b3858a62fd9b7e1c8f838d1960f85d8334c4aae1de3dce4cc113d5` |
| evaluate.py | 265 | `ef2259d830cf30db3dc077c0b29ab47938cb1805a11c01185871e402c9bca1f5` |
| test_alphazero.py | 289 | `a8130c22f1f2bc9f278c5379938b2a85ee5de69f2b3d012c88dd5684610af833` |

本文生成过程只读取项目/模型并复跑已有轻量测试，没有启动训练、修改生产源码、删除产物或操作运行进程；本次新写入文件仅为本 handoff。

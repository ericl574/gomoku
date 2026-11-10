# 训练进度与搜索工程改进报告

本次修改四个生产文件：mcts.py、self_play.py、evaluate.py、train.py；在原来的空文件 test_alphazero.py 中加入回归测试。完整逐行 diff 见本文末尾。

**当前已经运行的 Python 进程没有被停止、重启或重新配置。它使用启动时加载的旧代码；以下改动下次运行生效。没有启动实际训练，没有更新现有模型权重或覆盖现有 checkpoint。**

## 改了哪里

| 文件和修改后行号 | 改动 |
|---|---|
| [mcts.py:12](/Users/ericl/Desktop/projects/wuziqi/mcts.py:12) | TreeNode 使用 slots，减少大量节点对象的内存开销。base_prior 保存原始网络先验，用来防止根噪声被反复混合。 |
| [mcts.py:85](/Users/ericl/Desktop/projects/wuziqi/mcts.py:85) | reset() 清除树和缓存；如果外部改变该 searcher 使用的模型权重，需要调用它。当前生产流程每局使用固定模型，不跨参数更新复用 searcher。 |
| [mcts.py:93](/Users/ericl/Desktop/projects/wuziqi/mcts.py:93) | advance(action, state) 在真实落子后保留对应子树，并用规则重放核对棋盘；不匹配时丢弃树。 |
| [mcts.py:111](/Users/ericl/Desktop/projects/wuziqi/mcts.py:111) | search() 复用匹配位置的树，每步仍添加指定数量的新模拟；加入每约 0.2 秒一次的进度回调，始终报告起点和终点。每次根搜索从 base_prior 重新加噪声。 |
| [mcts.py:187](/Users/ericl/Desktop/projects/wuziqi/mcts.py:187) | 每次选择只计算一次父节点访问数的平方根，避免对每个候选动作重复计算；PUCT 公式及乘法顺序保持不变。 |
| [mcts.py:204](/Users/ericl/Desktop/projects/wuziqi/mcts.py:204) | 实现按完整局面和当前玩家区分的有界 LRU 推理缓存。真实对照没有发现收益，因此默认 cache_size=0；可用 MCTS(..., cache_size=4096) 单独实验。 |
| [self_play.py:31](/Users/ericl/Desktop/projects/wuziqi/self_play.py:31) | 自我对弈可通过 progress_callback 报告当前落子编号、玩家、搜索进度、整局耗时、复用和网络调用统计。 |
| [self_play.py:84](/Users/ericl/Desktop/projects/wuziqi/self_play.py:84) | 每次真实落子后推进搜索树。 |
| [evaluate.py:57](/Users/ericl/Desktop/projects/wuziqi/evaluate.py:57) | 评估对局也支持逐步进度，并对双方应用相同的树复用规则。 |
| [evaluate.py:109](/Users/ericl/Desktop/projects/wuziqi/evaluate.py:109) | 双方的独立 searcher 都随每一步真实落子推进；每方的缓存和网络仍相互独立。 |
| [evaluate.py:214](/Users/ericl/Desktop/projects/wuziqi/evaluate.py:214) | 新增 --no-tree-reuse，可重建每步搜索树，复现旧评估协议。终端会明确打印是否复用。 |
| [train.py:130](/Users/ericl/Desktop/projects/wuziqi/train.py:130) | checkpoint 先写临时文件，再原子替换；写入失败时保留已有文件并清理临时文件。 |
| [train.py:164](/Users/ericl/Desktop/projects/wuziqi/train.py:164) | 默认改为每 10 轮评估一次，最后一轮仍评估。用户显式传入 eval_every=1 时，仍会按每轮执行，不会偷偷覆盖参数。 |
| [train.py:166](/Users/ericl/Desktop/projects/wuziqi/train.py:166) | 增加可选 cpu_threads；默认 None，保留当前 PyTorch 设置。实测没有稳定的最佳线程数，不强行指定。 |
| [train.py:253](/Users/ericl/Desktop/projects/wuziqi/train.py:253) | 面板新增当前步数、当前玩家、对局耗时、线程数、已有访问统计和推理统计。 |
| [train.py:288](/Users/ericl/Desktop/projects/wuziqi/train.py:288) | 接收自我对弈和评估的事件，原地刷新单步 MCTS 进度条；当前步数没有伪装成已知的整局总进度。 |
| [train.py:304](/Users/ericl/Desktop/projects/wuziqi/train.py:304) | 更新参数时隐藏单步搜索条，避免把上一手搜索当成正在计算。 |
| [train.py:401](/Users/ericl/Desktop/projects/wuziqi/train.py:401) | checkpoint 保存线程数、树复用、缓存与评估频率等运行配置，便于解释不同实验结果。 |
| [train.py:412](/Users/ericl/Desktop/projects/wuziqi/train.py:412) | 参数更新后、评估开始前先保存 checkpoint；长评估中断也能保留本轮已更新模型。 |
| [train.py:440](/Users/ericl/Desktop/projects/wuziqi/train.py:440) | 评估期间也显示当前步数和 0～512 等实际搜索进度。 |
| [train.py:473](/Users/ericl/Desktop/projects/wuziqi/train.py:473) | 评估完成后原子替换 checkpoint，补充结果；训练日志仍记录完整轮次信息。 |

## 重要语义

- 原算法仍为 PUCT，没有引入 Gumbel、手工棋形评分或新的奖励。
- 512 始终表示该步新增 512 次模拟。复用树可能已经包含更早的访问统计，所以根节点累计访问次数可大于 512；训练目标按保留后的子节点累计访问次数归一化。
- 树复用会改变搜索行为，不能将新协议的胜率与旧协议直接混为同一实验。评估双方使用相同设置；需要旧协议时使用 --no-tree-reuse。
- 缓存、树统计仅适用于权重固定的模型。外部更新模型后必须 reset()。现有自我对弈快照和评估流程满足这个条件。
- 树复用保留已有工作，不保证在固定“新增模拟次数”下每步更快，也不保证棋力提高。

## 实测与取舍

本次测量受当前机器负载影响；没有把小基准结果当作完整训练加速保证。

### 确认有帮助的局部优化

- 361 个空子节点对象（含各自的空 children 字典）的测量：80,864 → 49,096 字节，减少约 39.3%；不代表整个进程内存减少同样比例。
- 固定节点的选点微基准中位数：123.378 → 80.926 微秒，约减少 34.4%；不代表整局或训练时间减少同样比例。
- 默认评估频率从每 5 轮改为每 10 轮，最后仍评估。显式指定频率优先。

### 没有据此宣称加速的部分

在相同六个真实位置、每步新增 512 次模拟的一个对照中（在 slots/平方根局部优化前）：

| 设置 | 耗时 | 网络调用次数 | 复用访问统计 | 缓存命中 |
|---|---:|---:|---:|---:|
| 原来的每步新建树 | 4.2577 秒 | 3078 | 0 | — |
| 树复用，缓存关闭 | 4.5387 秒 | 3073 | 2290 | 0 |
| 树复用，缓存开启 | 4.5982 秒 | 3073 | 2290 | 0 |

因此缓存默认关闭，树复用的收益应从保留统计与后续对战验证，不把这张表描述成速度提升。缓存开关下的策略输出逐项完全一致。

线程测试（每次前向毫秒，中位数）：

| CPU 线程 | batch=1 | batch=32 |
|---|---:|---:|
| 1 | 0.769 | 30.707 |
| 2 | 0.825 | 30.455 |
| 4 | 0.758 | 31.790 |

没有稳定的大幅优势，故保留系统默认，允许传入 cpu_threads。

另一小测试中 no_grad 与 inference_mode 分别约 0.7409 / 0.7353 毫秒，差异很小，没有据此改动推理模式或宣称提速。GPU 和批量叶子推理本次没有加入，需要另外验证执行开销、调度和搜索语义。

## 验证

运行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest test_alphazero -v
```

13 项回归检查通过，涵盖：

- 优化后的选择结果与参考 PUCT 公式一致。
- 复用后的原统计保留，指定的新模拟次数精确加入。
- 不匹配的动作／局面不复用旧树。
- 根噪声每次从原始先验重新混合，也能关闭。
- 缓存区分当前玩家、容量受限、reset 后失效。
- 缓存不改变搜索结果；进度回调不改变搜索结果。
- 两种颜色的终局符号正确，终局跳过神经网络。
- 完整自我对弈与评估的事件和训练标签正确。
- 默认评估频率生效，评估前保存模型。
- 模拟保存失败时保留已有 checkpoint。
- 旧评估协议可以关闭树复用。

另外完成了真实残差网络的完整对局检查：118 步、白胜，样本、进度事件、结果视角通过验证，权重未变。面板在 80 和 100 列宽度下均完整显示约 20 行。新树的四个固定局面搜索与旧版本的对应新建树搜索逐项一致。

测试中的训练组织检查使用 mock，没有执行真正的 optimizer.step；原子保存测试只在自动清理的临时目录写入诊断文件，没有覆盖现有训练产物。

## 下次运行

先让当前训练自然结束。之后可以运行：

```bash
python -c "from train import train; train(iterations=2, games_per_iteration=2, simulations=512, updates_per_iteration=10, eval_games=10)"
```

这次省略 eval_every=1，使用默认频率；2 轮实验只在最后一轮评估 10 局。新面板会显示当前步数和单步搜索进度。每次调用 train 仍会创建新的训练实验，不是恢复旧 checkpoint。

## 完整逐行 diff

以下为四个生产文件和测试文件的完整修改。减号为删除行，加号为新增行，@@ 标明行号。

```diff
--- before/mcts.py
+++ after/mcts.py
@@ -1,4 +1,6 @@
 import math
+import time
+from collections import OrderedDict
 
 import numpy as np
 import torch
@@ -7,8 +9,11 @@
 
 
 class TreeNode:
+    __slots__ = ("base_prior", "prior", "visit_count", "value_sum", "children")
+
     def __init__(self, prior=1.0):
-        self.prior = float(prior)
+        self.base_prior = float(prior)
+        self.prior = self.base_prior
         self.visit_count = 0
         self.value_sum = 0.0
         self.children = {}
@@ -30,6 +35,7 @@
         dirichlet_alpha=0.03,
         noise_fraction=0.25,
         seed=None,
+        cache_size=0,
     ):
         if (
             isinstance(simulations, bool)
@@ -46,6 +52,12 @@
 
         if not 0 <= noise_fraction <= 1:
             raise ValueError("Noise fraction must be between 0 and 1")
+        if (
+            isinstance(cache_size, bool)
+            or not isinstance(cache_size, int)
+            or cache_size < 0
+        ):
+            raise ValueError("Cache size must be a nonnegative integer")
 
         self.model = model
         self.simulations = int(simulations)
@@ -53,9 +65,50 @@
         self.dirichlet_alpha = dirichlet_alpha
         self.noise_fraction = noise_fraction
         self.rng = np.random.default_rng(seed)
+        self.cache_size = cache_size
+        self._cache = OrderedDict()
+        self._root = None
+        self._root_key = None
+        self._root_state = None
+        self.last_search_stats = {}
+
+    @staticmethod
+    def _state_key(state):
+        return (
+            state.size,
+            state.board.tobytes(),
+            state.current_player,
+            state.done,
+            state.winner,
+        )
+
+    def reset(self):
+        """Discard cached results; required if this searcher's model changes."""
+        self._root = None
+        self._root_key = None
+        self._root_state = None
+        self._cache.clear()
+        self.last_search_stats = {}
+
+    def advance(self, action, state):
+        """Retain the chosen subtree after a real move, checking the position."""
+        child = None
+        if self._root is not None and self._root_state is not None:
+            expected = self._root_state.clone()
+            try:
+                expected.step(action)
+            except ValueError:
+                pass
+            else:
+                if self._state_key(expected) == self._state_key(state):
+                    child = self._root.children.get(int(action))
+
+        self._root = child
+        self._root_key = self._state_key(state) if child is not None else None
+        self._root_state = state.clone() if child is not None else None
 
     @torch.no_grad()
-    def search(self, state, add_noise=False):
+    def search(self, state, add_noise=False, progress_callback=None):
         if state.done:
             raise ValueError("Cannot search a terminal state")
 
@@ -63,8 +116,25 @@
         self.model.eval()
 
         try:
-            root = TreeNode()
-            self._evaluate(state, root)
+            key = self._state_key(state)
+            if self._root is None or self._root_key != key:
+                self._root = TreeNode()
+                self._root_key = key
+                self._root_state = state.clone()
+            root = self._root
+            self.last_search_stats = {
+                "reused_visits": root.visit_count,
+                "network_calls": 0,
+                "cache_hits": 0,
+            }
+            if progress_callback is not None:
+                progress_callback(0, self.simulations)
+            if not root.children:
+                self._evaluate(state, root)
+
+            # Never mix fresh root noise with noise left over from an earlier search.
+            for child in root.children.values():
+                child.prior = child.base_prior
 
             if add_noise:
                 children = list(root.children.values())
@@ -78,7 +148,8 @@
                         + self.noise_fraction * noise_prior
                     )
 
-            for _ in range(self.simulations):
+            next_report = time.perf_counter() + 0.2
+            for simulation in range(self.simulations):
                 simulation_state = state.clone()
                 node = root
                 path = [node]
@@ -91,11 +162,21 @@
                 value = self._evaluate(simulation_state, node)
                 self._backup(path, value)
 
+                completed = simulation + 1
+                if progress_callback is not None and (
+                    completed == self.simulations
+                    or time.perf_counter() >= next_report
+                ):
+                    progress_callback(completed, self.simulations)
+                    next_report = time.perf_counter() + 0.2
+
             policy = np.zeros(state.size * state.size, dtype=np.float32)
 
             for action, child in root.children.items():
                 policy[action] = child.visit_count
 
+            # Retained visits remain part of the search policy; the loop above
+            # always adds exactly self.simulations NEW simulations.
             policy /= policy.sum()
 
             return policy, root
@@ -103,6 +184,8 @@
             self.model.train(was_training)
 
     def _select(self, node):
+        sqrt_visits = math.sqrt(node.visit_count + 1)
+
         def score(item):
             _, child = item
 
@@ -110,7 +193,7 @@
             exploration = (
                 self.c_puct
                 * child.prior
-                * math.sqrt(node.visit_count + 1)
+                * sqrt_visits
                 / (1 + child.visit_count)
             )
 
@@ -122,32 +205,47 @@
         if state.done:
             return state.terminal_value()
 
-        device = next(self.model.parameters()).device
-        observation = torch.from_numpy(state.encode())
-        observation = observation.unsqueeze(0).to(device)
-
-        logits, value = self.model(observation)
-
-        legal_actions = state.legal_actions()
-        legal_indices = torch.as_tensor(
-            legal_actions,
-            dtype=torch.long,
-            device=device,
-        )
-
-        legal_mask = torch.zeros_like(logits, dtype=torch.bool)
-        legal_mask[0, legal_indices] = True
-        probabilities = masked_policy(logits, legal_mask)
-        probabilities = probabilities[0].cpu().numpy()
-
-        for action in legal_actions:
+        key = self._state_key(state) if self.cache_size else None
+        cached = self._cache.get(key) if self.cache_size else None
+        if cached is not None:
+            self._cache.move_to_end(key)
+            probabilities, value = cached
+            self.last_search_stats["cache_hits"] = (
+                self.last_search_stats.get("cache_hits", 0) + 1
+            )
+        else:
+            device = next(self.model.parameters()).device
+            observation = torch.from_numpy(state.encode())
+            observation = observation.unsqueeze(0).to(device)
+            logits, predicted_value = self.model(observation)
+            legal_indices = torch.as_tensor(
+                state.legal_actions(),
+                dtype=torch.long,
+                device=device,
+            )
+            legal_mask = torch.zeros_like(logits, dtype=torch.bool)
+            legal_mask[0, legal_indices] = True
+            probabilities = masked_policy(logits, legal_mask)
+            probabilities = probabilities[0].cpu().numpy()
+            value = predicted_value.item()
+            self.last_search_stats["network_calls"] = (
+                self.last_search_stats.get("network_calls", 0) + 1
+            )
+            if self.cache_size:
+                probabilities = probabilities.copy()
+                probabilities.setflags(write=False)
+                self._cache[key] = (probabilities, value)
+                if len(self._cache) > self.cache_size:
+                    self._cache.popitem(last=False)
+
+        for action in state.legal_actions():
             action = int(action)
             node.children[action] = TreeNode(probabilities[action])
 
-        return value.item()
+        return value
 
     def _backup(self, path, value):
         for node in reversed(path):
             node.visit_count += 1
             node.value_sum += value
-            value = -value+            value = -value
--- before/self_play.py
+++ after/self_play.py
@@ -28,6 +28,7 @@
     simulations=8,
     exploration_moves=30,
     seed=None,
+    progress_callback=None,
 ):
     if (
         isinstance(exploration_moves, bool)
@@ -46,9 +47,25 @@
 
     state = GomokuState()
     history = []
+    started = time.perf_counter()
 
     while not state.done:
-        policy, _ = searcher.search(state, add_noise=True)
+        def report_search(completed, total):
+            if progress_callback is not None:
+                progress_callback({
+                    "move": len(history) + 1,
+                    "player": state.current_player,
+                    "simulations_done": completed,
+                    "simulations_total": total,
+                    "elapsed_seconds": time.perf_counter() - started,
+                    **searcher.last_search_stats,
+                })
+
+        policy, _ = searcher.search(
+            state,
+            add_noise=True,
+            progress_callback=report_search if progress_callback else None,
+        )
 
         history.append(
             (
@@ -64,6 +81,7 @@
             sample=len(history) <= exploration_moves,
         )
         state.step(action)
+        searcher.advance(action, state)
 
     samples = []
 
--- before/evaluate.py
+++ after/evaluate.py
@@ -1,4 +1,5 @@
 import argparse
+import time
 from pathlib import Path
 
 import numpy as np
@@ -53,6 +54,8 @@
     mode,
     simulations,
     seed,
+    progress_callback=None,
+    reuse_tree=True,
 ):
     state = GomokuState()
     rng = np.random.default_rng(seed)
@@ -71,14 +74,29 @@
         }
 
     moves = 0
+    started = time.perf_counter()
 
     while not state.done:
         player = state.current_player
 
         if mode == "mcts":
+            searcher = searchers[player]
+
+            def report_search(completed, total):
+                if progress_callback is not None:
+                    progress_callback({
+                        "move": moves + 1,
+                        "player": player,
+                        "simulations_done": completed,
+                        "simulations_total": total,
+                        "elapsed_seconds": time.perf_counter() - started,
+                        **searcher.last_search_stats,
+                    })
+
             policy, _ = searchers[player].search(
                 state,
                 add_noise=False,
+                progress_callback=report_search if progress_callback else None,
             )
         elif mode == "policy":
             policy = policy_for_state(models[player], state)
@@ -87,9 +105,14 @@
 
         action = select_action(policy, rng, sample=False)
         state.step(action)
+        for searcher in searchers.values():
+            if reuse_tree:
+                searcher.advance(action, state)
+            else:
+                searcher.reset()
         moves += 1
 
-    result = final_result = state.winner * candidate_color
+    result = state.winner * candidate_color
 
     return int(result), moves
 
@@ -101,6 +124,7 @@
     games=4,
     simulations=8,
     seed=12345,
+    reuse_tree=True,
 ):
     if games < 2 or games % 2 != 0:
         raise ValueError("Games must be a positive even number")
@@ -126,6 +150,7 @@
             mode,
             simulations,
             game_seed,
+            reuse_tree=reuse_tree,
         )
 
         records.append(
@@ -186,6 +211,11 @@
     parser.add_argument("--simulations", type=int, default=8)
     parser.add_argument("--seed", type=int, default=12345)
     parser.add_argument(
+        "--no-tree-reuse",
+        action="store_true",
+        help="Rebuild the search tree each move for comparison with older runs",
+    )
+    parser.add_argument(
         "--mode",
         choices=("mcts", "policy", "both"),
         default="both",
@@ -215,6 +245,7 @@
     print("Games per mode:", args.games)
     print("MCTS simulations per move:", args.simulations)
     print("Evaluation seed:", args.seed)
+    print("Tree reuse:", not args.no_tree_reuse)
 
     modes = ("mcts", "policy") if args.mode == "both" else (args.mode,)
 
@@ -226,8 +257,9 @@
             games=args.games,
             simulations=args.simulations,
             seed=args.seed,
+            reuse_tree=not args.no_tree_reuse,
         )
 
 
 if __name__ == "__main__":
-    main()+    main()
--- before/train.py
+++ after/train.py
@@ -127,6 +127,16 @@
     }
 
 
+def save_checkpoint(checkpoint, path):
+    """Replace a checkpoint only after the new file is fully written."""
+    temporary = path.with_suffix(path.suffix + ".tmp")
+    try:
+        torch.save(checkpoint, temporary)
+        temporary.replace(path)
+    finally:
+        temporary.unlink(missing_ok=True)
+
+
 def render_dashboard(progress, status):
     table = Table.grid(padding=(0, 2))
     table.add_column(style="cyan", no_wrap=True)
@@ -151,8 +161,9 @@
     buffer_capacity=10000,
     learning_rate=1e-3,
     seed=0,
-    eval_every=5,
+    eval_every=10,
     eval_games=20,
+    cpu_threads=None,
 ):
     counts = (
         iterations,
@@ -178,6 +189,14 @@
 
     if not np.isfinite(learning_rate) or learning_rate <= 0:
         raise ValueError("Learning rate must be positive and finite")
+    if cpu_threads is not None:
+        if (
+            isinstance(cpu_threads, bool)
+            or not isinstance(cpu_threads, int)
+            or cpu_threads < 1
+        ):
+            raise ValueError("CPU threads must be a positive integer")
+        torch.set_num_threads(cpu_threads)
 
     torch.manual_seed(seed)
     rng = np.random.default_rng(seed)
@@ -230,6 +249,10 @@
     status = {
         "Stage": "Starting",
         "Search / batch": f"{simulations} simulations / up to {batch_size}",
+        "CPU threads": str(torch.get_num_threads()),
+        "Current move": "--",
+        "Game elapsed": "--",
+        "Last search": "--",
         "Loss: last batch": "--",
         "Loss: round mean": "--",
         "Replay buffer": f"0 / {buffer_capacity}",
@@ -249,6 +272,7 @@
     round_task = progress.add_task("Round", total=iterations)
     game_task = progress.add_task("Self-play", total=total_games)
     update_task = progress.add_task("Updates", total=total_updates)
+    search_task = progress.add_task("MCTS", total=simulations, visible=False)
     eval_task = progress.add_task(
         "Evaluation",
         total=eval_games,
@@ -261,6 +285,27 @@
         render_dashboard(progress, status),
         refresh_per_second=4,
     ) as live:
+        def report_search(event):
+            player = "Black" if event["player"] == 1 else "White"
+            status["Current move"] = f"#{event['move']} {player}"
+            status["Game elapsed"] = f"{event['elapsed_seconds']:.1f}s"
+            status["Last search"] = (
+                f"retained={event['reused_visits']} "
+                f"NN={event['network_calls']} cache={event['cache_hits']}"
+            )
+            progress.update(
+                search_task,
+                completed=event["simulations_done"],
+                total=event["simulations_total"],
+                visible=True,
+            )
+            live.update(render_dashboard(progress, status))
+
+        def hide_search():
+            progress.update(search_task, visible=False)
+            status["Current move"] = "--"
+            status["Game elapsed"] = "--"
+
         for iteration in range(1, iterations + 1):
             snapshot = deepcopy(model)
             snapshot.eval()
@@ -277,6 +322,7 @@
                     snapshot,
                     simulations=simulations,
                     seed=int(rng.integers(0, 2**32)),
+                    progress_callback=report_search,
                 )
 
                 replay.add_game(samples)
@@ -302,6 +348,7 @@
 
             actual_batch_size = min(batch_size, len(replay))
             metrics = []
+            hide_search()
 
             for update in range(1, updates_per_iteration + 1):
                 status["Stage"] = (
@@ -339,6 +386,31 @@
             )
 
             evaluation = None
+
+            checkpoint_path = run_dir / f"iteration_{iteration:03d}.pt"
+            checkpoint = {
+                "iteration": iteration,
+                "updates_completed": completed_updates,
+                "model_config": model_config,
+                "model_state_dict": model.state_dict(),
+                "optimizer_state_dict": optimizer.state_dict(),
+                "metrics": averages,
+                "evaluation": None,
+                "seed": seed,
+                "simulations": simulations,
+                "runtime_config": {
+                    "cpu_threads": torch.get_num_threads(),
+                    "tree_reuse": True,
+                    "cache_size": 0,
+                    "eval_every": eval_every,
+                    "eval_games": eval_games,
+                },
+            }
+            # Preserve completed updates before potentially lengthy evaluation.
+            status["Stage"] = f"Round {iteration}: saving"
+            live.update(render_dashboard(progress, status))
+            save_checkpoint(checkpoint, checkpoint_path)
+            status["Last checkpoint"] = checkpoint_path.name
 
             if iteration % eval_every == 0 or iteration == iterations:
                 model.eval()
@@ -365,6 +437,7 @@
                         mode="mcts",
                         simulations=simulations,
                         seed=int(eval_seeds[game]),
+                        progress_callback=report_search,
                     )
                     results.append(result)
 
@@ -393,28 +466,11 @@
                     f"W/D/L={wins}/{draws}/{losses}"
                 )
                 model.train()
-
-            checkpoint_path = (
-                run_dir / f"iteration_{iteration:03d}.pt"
-            )
-
-            status["Stage"] = f"Round {iteration}: saving"
-            live.update(render_dashboard(progress, status))
-
-            torch.save(
-                {
-                    "iteration": iteration,
-                    "updates_completed": completed_updates,
-                    "model_config": model_config,
-                    "model_state_dict": model.state_dict(),
-                    "optimizer_state_dict": optimizer.state_dict(),
-                    "metrics": averages,
-                    "evaluation": evaluation,
-                    "seed": seed,
-                    "simulations": simulations,
-                },
-                checkpoint_path,
-            )
+                hide_search()
+                status["Stage"] = f"Round {iteration}: saving evaluation"
+                live.update(render_dashboard(progress, status))
+                checkpoint["evaluation"] = evaluation
+                save_checkpoint(checkpoint, checkpoint_path)
 
             record = {
                 "iteration": iteration,
--- before/test_alphazero.py
+++ after/test_alphazero.py
@@ -0,0 +1,289 @@
+import tempfile
+import unittest
+from pathlib import Path
+from unittest.mock import mock_open, patch
+
+import numpy as np
+import torch
+from torch import nn
+
+import train as training
+from evaluate import play_match
+from mcts import MCTS, TreeNode
+from self_play import play_game
+from state import GomokuState
+
+
+class CountingNetwork(nn.Module):
+    """A deterministic diagnostic network; never used as training data."""
+
+    def __init__(self):
+        super().__init__()
+        self.anchor = nn.Parameter(torch.tensor(0.0))
+        self.calls = 0
+
+    def forward(self, x):
+        self.calls += 1
+        logits = torch.zeros(len(x), 361, device=x.device)
+        logits[:, 180] = 10.0
+        values = x[:, 2, 0, 0].unsqueeze(1) * 0.2 - 0.3 + self.anchor
+        return logits, values
+
+
+class SearchTests(unittest.TestCase):
+    def test_optimized_selection_matches_reference_formula(self):
+        rng = np.random.default_rng(19)
+        searcher = MCTS(CountingNetwork())
+        for _ in range(20):
+            root = TreeNode()
+            root.visit_count = 512
+            priors = rng.dirichlet(np.ones(50))
+            for action, prior in enumerate(priors):
+                child = TreeNode(prior)
+                child.visit_count = int(rng.integers(0, 20))
+                child.value_sum = float(rng.uniform(-1, 1) * child.visit_count)
+                root.children[action] = child
+            expected = max(
+                root.children.items(),
+                key=lambda item: (
+                    -item[1].q
+                    + searcher.c_puct * item[1].prior
+                    * np.sqrt(root.visit_count + 1)
+                    / (1 + item[1].visit_count)
+                ),
+            )
+            self.assertEqual(searcher._select(root)[0], expected[0])
+
+    def test_reuse_retains_statistics_and_adds_exact_budget(self):
+        state = GomokuState()
+        searcher = MCTS(CountingNetwork(), simulations=8)
+        _, root = searcher.search(state)
+        child = root.children[180]
+        prior_visits = child.visit_count
+        prior_child_visits = sum(c.visit_count for c in child.children.values())
+        board = state.board.copy()
+        state.step(180)
+        searcher.advance(180, state)
+        policy, reused = searcher.search(state)
+        self.assertIs(reused, child)
+        self.assertEqual(reused.visit_count, prior_visits + 8)
+        self.assertEqual(
+            sum(c.visit_count for c in reused.children.values()),
+            prior_child_visits + 8,
+        )
+        self.assertEqual(searcher.last_search_stats["reused_visits"], prior_visits)
+        self.assertAlmostEqual(float(policy.sum()), 1.0)
+        self.assertEqual(policy[180], 0)
+        board[9, 9] = 1
+        np.testing.assert_array_equal(state.board, board)
+
+    def test_mismatched_advance_discards_tree(self):
+        state = GomokuState()
+        searcher = MCTS(CountingNetwork(), simulations=4)
+        _, old = searcher.search(state)
+        state.step(1)
+        searcher.advance(180, state)
+        _, root = searcher.search(state)
+        self.assertIsNot(root, old.children[180])
+        self.assertEqual(searcher.last_search_stats["reused_visits"], 0)
+        self.assertEqual(root.visit_count, 4)
+
+    def test_unannounced_position_change_discards_tree(self):
+        state = GomokuState()
+        searcher = MCTS(CountingNetwork(), simulations=4)
+        searcher.search(state)
+        state.step(0)
+        _, root = searcher.search(state)
+        self.assertEqual(searcher.last_search_stats["reused_visits"], 0)
+        self.assertEqual(root.visit_count, 4)
+
+    def test_root_noise_is_fresh_and_can_be_disabled(self):
+        state = GomokuState()
+        searcher = MCTS(CountingNetwork(), simulations=1, seed=17)
+        rng = np.random.default_rng(17)
+        for _ in range(2):
+            _, root = searcher.search(state, add_noise=True)
+            noise = rng.dirichlet(np.full(361, 0.03))
+            actual = [child.prior for child in root.children.values()]
+            expected = [
+                0.75 * child.base_prior + 0.25 * n
+                for child, n in zip(root.children.values(), noise)
+            ]
+            np.testing.assert_allclose(actual, expected)
+        _, root = searcher.search(state, add_noise=False)
+        self.assertTrue(all(c.prior == c.base_prior for c in root.children.values()))
+
+    @torch.no_grad()
+    def test_cache_keys_include_player_and_reset_invalidates_cache(self):
+        model = CountingNetwork().eval()
+        searcher = MCTS(model, simulations=1, cache_size=2)
+        state = GomokuState()
+        first = searcher._evaluate(state, TreeNode())
+        self.assertEqual(searcher._evaluate(state, TreeNode()), first)
+        self.assertEqual(model.calls, 1)
+        state.current_player = -1
+        other = searcher._evaluate(state, TreeNode())
+        self.assertNotEqual(first, other)
+        self.assertEqual(model.calls, 2)
+        state.step(0)
+        searcher._evaluate(state, TreeNode())
+        self.assertEqual(len(searcher._cache), 2)
+        model.anchor.add_(1)
+        searcher.reset()
+        self.assertEqual(len(searcher._cache), 0)
+        self.assertNotEqual(searcher._evaluate(state, TreeNode()), other)
+
+    def test_cache_does_not_change_search_results(self):
+        state = GomokuState()
+        cached = MCTS(CountingNetwork(), simulations=8, seed=5, cache_size=64)
+        uncached = MCTS(CountingNetwork(), simulations=8, seed=5, cache_size=0)
+        for _ in range(4):
+            a, ra = cached.search(state, add_noise=True)
+            b, rb = uncached.search(state, add_noise=True)
+            np.testing.assert_array_equal(a, b)
+            self.assertEqual(ra.visit_count, rb.visit_count)
+            self.assertEqual(ra.value_sum, rb.value_sum)
+            action = int(np.argmax(a))
+            state.step(action)
+            cached.advance(action, state)
+            uncached.advance(action, state)
+
+    def test_callbacks_preserve_search_and_report_endpoints(self):
+        events = []
+        state = GomokuState()
+        plain = MCTS(CountingNetwork(), simulations=8, seed=3)
+        observed = MCTS(CountingNetwork(), simulations=8, seed=3)
+        a, _ = plain.search(state, add_noise=True)
+        b, _ = observed.search(
+            state, add_noise=True,
+            progress_callback=lambda done, total: events.append((done, total)),
+        )
+        np.testing.assert_array_equal(a, b)
+        self.assertEqual(events[0], (0, 8))
+        self.assertEqual(events[-1], (8, 8))
+        self.assertEqual(events, sorted(events))
+
+    def test_terminal_sign_and_network_bypass_for_both_colors(self):
+        for color in (1, -1):
+            state = GomokuState()
+            state.current_player = color
+            state.board[9, 5:9] = color
+            model = CountingNetwork()
+            policy, root = MCTS(model, simulations=8).search(state)
+            self.assertEqual(policy[180], 1)
+            self.assertEqual(root.q, 1)
+            self.assertEqual(root.children[180].q, -1)
+            self.assertEqual(model.calls, 1)
+            self.assertTrue(model.training)
+
+
+class IntegrationTests(unittest.TestCase):
+    def test_legacy_evaluation_can_disable_tree_reuse(self):
+        import evaluate as evaluation
+        retained = []
+
+        class ObservedMCTS(MCTS):
+            def search(self, *args, **kwargs):
+                result = super().search(*args, **kwargs)
+                retained.append(self.last_search_stats["reused_visits"])
+                return result
+
+        with patch.object(evaluation, "MCTS", ObservedMCTS):
+            play_match(
+                CountingNetwork().eval(), CountingNetwork().eval(),
+                1, "mcts", 2, 0, reuse_tree=False,
+            )
+        self.assertTrue(retained)
+        self.assertTrue(all(count == 0 for count in retained))
+
+    def test_complete_self_play_and_evaluation_callbacks(self):
+        model = CountingNetwork().eval()
+        events = []
+        samples, final = play_game(
+            model, simulations=2, seed=0, progress_callback=events.append,
+        )
+        self.assertTrue(final.done)
+        self.assertEqual(len(samples), np.count_nonzero(final.board))
+        for i, (obs, policy, value) in enumerate(samples):
+            self.assertEqual(obs[:2].sum(), i)
+            self.assertAlmostEqual(float(policy.sum()), 1)
+            self.assertFalse(policy[(obs[0] + obs[1]).ravel() != 0].any())
+            self.assertEqual(value, final.winner * (1 if i % 2 == 0 else -1))
+        self.assertEqual(events[0]["move"], 1)
+        self.assertEqual(events[-1]["move"], len(samples))
+        self.assertEqual(events[-1]["simulations_done"], 2)
+        self.assertTrue(any(e["reused_visits"] > 0 for e in events))
+        self.assertTrue(all(p.grad is None for p in model.parameters()))
+        events.clear()
+        result, moves = play_match(
+            model, CountingNetwork().eval(), 1, "mcts", 2, 7,
+            progress_callback=events.append,
+        )
+        self.assertIn(result, (-1, 0, 1))
+        self.assertEqual(events[-1]["move"], moves)
+
+    def test_default_evaluation_frequency_and_save_before_eval(self):
+        event_order = []
+        state = GomokuState()
+        state.done = True
+        state.winner = 1
+        policy = np.zeros(361, dtype=np.float32)
+        policy[180] = 1
+        sample = (GomokuState().encode(), policy, 1.0)
+
+        def notify(callback):
+            callback({
+                "move": 1, "player": 1, "simulations_done": 2,
+                "simulations_total": 2, "elapsed_seconds": 0.1,
+                "reused_visits": 0, "network_calls": 2, "cache_hits": 0,
+            })
+
+        def fake_game(*args, **kwargs):
+            notify(kwargs["progress_callback"])
+            return [sample], state
+
+        def fake_match(*args, **kwargs):
+            event_order.append("eval")
+            notify(kwargs["progress_callback"])
+            return 1, 9
+
+        def capture(checkpoint, path):
+            event_order.append(("save", checkpoint["iteration"], checkpoint["evaluation"] is None))
+
+        with (
+            patch.object(training.Path, "mkdir"),
+            patch.object(training.Path, "open", mock_open()),
+            patch.object(training, "Live"),
+            patch.object(training, "play_game", side_effect=fake_game),
+            patch.object(training, "play_match", side_effect=fake_match) as matches,
+            patch.object(training, "train_step", return_value={"loss": 2., "policy_loss": 1., "value_loss": 1.}),
+            patch.object(training, "save_checkpoint", side_effect=capture),
+            patch.object(training.torch.optim, "Adam"),
+            patch("builtins.print"),
+        ):
+            training.train(iterations=2, simulations=2, updates_per_iteration=1, eval_games=2)
+        self.assertEqual(matches.call_count, 2)
+        self.assertEqual(event_order, [
+            ("save", 1, True), ("save", 2, True),
+            "eval", "eval", ("save", 2, False),
+        ])
+
+    def test_failed_checkpoint_write_keeps_previous_file(self):
+        with tempfile.TemporaryDirectory() as directory:
+            path = Path(directory) / "model.pt"
+            training.save_checkpoint({"value": 1}, path)
+            before = path.read_bytes()
+
+            def interrupted(checkpoint, temporary):
+                temporary.write_bytes(b"incomplete")
+                raise RuntimeError("simulated failed write")
+
+            with patch.object(training.torch, "save", side_effect=interrupted):
+                with self.assertRaises(RuntimeError):
+                    training.save_checkpoint({"value": 2}, path)
+            self.assertEqual(path.read_bytes(), before)
+            self.assertFalse(path.with_suffix(".pt.tmp").exists())
+
+
+if __name__ == "__main__":
+    unittest.main()

```

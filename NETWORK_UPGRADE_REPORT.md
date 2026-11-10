# 全局残差网络修改报告

本次直接修改 `network.py`、`train.py`、`self_play.py`，并新增本报告。下面的行号以修改后的文件为准；末尾附完整逐行 diff。没有启动训练，没有更新或覆盖 checkpoint、训练日志。

## 1. 最终网络结构

```text
输入 (B, 3, 19, 19)
  → 3×3 卷积：3 → 32 通道，ReLU
  → 3 个 ResidualBlock：每块两层 3×3 卷积，保留 19×19
  → GlobalContext：整盘平均汇总 → 通道门控 → 反馈到所有位置
  ├─ Policy：3×3 卷积 → 展平 → (B, 361) logits
  └─ Value：整盘平均汇总 → 32→64→1 全连接 → tanh → (B, 1)
```

新配置为 `architecture="resnet_global"`、`residual_blocks=3`、`channels=32`、`value_hidden=64`、`policy_kernel=3`。总参数量 **59,402**。

仅计算局部卷积通路，策略分支共有 8 层 3×3 卷积，理论局部感受野为 17×17。全局信息由另外的门控通路提供，不依靠这个局部窗口覆盖整盘。

全局平均汇总是一个压缩摘要，不能等同于保留整盘所有精确位置关系；残差结构也不自动保证棋力或旋转不变性。

## 2. 按行解释

### network.py

| 修改后行号 | 修改及原因 |
|---|---|
| [5–6](/Users/ericl/Desktop/projects/wuziqi/network.py:5) | 新增 ResidualBlock 类，表示“在已有特征上学习修正”。 |
| [8–10](/Users/ericl/Desktop/projects/wuziqi/network.py:8) | 初始化模块并注册残差分支，PyTorch 可以收集它的参数。 |
| [11](/Users/ericl/Desktop/projects/wuziqi/network.py:11) | 第一层 3×3 卷积，通道数和棋盘尺寸保持不变。 |
| [12](/Users/ericl/Desktop/projects/wuziqi/network.py:12) | ReLU 为残差分支提供非线性。 |
| [13–15](/Users/ericl/Desktop/projects/wuziqi/network.py:13) | 第二层卷积生成修正量，另外定义相加后的激活函数。 |
| [17–18](/Users/ericl/Desktop/projects/wuziqi/network.py:17) | 计算 ReLU(x + F(x))；原特征与学习到的修正相加。 |
| [21–22](/Users/ericl/Desktop/projects/wuziqi/network.py:21) | 新增 GlobalContext 类，将全局摘要用于通道门控。 |
| [24–26](/Users/ericl/Desktop/projects/wuziqi/network.py:24) | 门控隐藏层宽度为 channels/4，最少为 1；32 通道时为 8。 |
| [27–28](/Users/ericl/Desktop/projects/wuziqi/network.py:27) | 对全部 19×19 位置取平均，输出 (B, C, 1, 1)。 |
| [29](/Users/ericl/Desktop/projects/wuziqi/network.py:29) | 将通道摘要从 C 压缩到隐藏宽度。 |
| [30](/Users/ericl/Desktop/projects/wuziqi/network.py:30) | 在门控网络中加入非线性。 |
| [31–33](/Users/ericl/Desktop/projects/wuziqi/network.py:31) | 恢复 C 个通道，并用 sigmoid 产生 0 到 1 的门控系数。 |
| [35–36](/Users/ericl/Desktop/projects/wuziqi/network.py:35) | 用全局系数乘以每个位置的局部特征，使远处棋子能够影响当前位置的通道组合。不是简单给所有动作加同一个常数。 |
| [45](/Users/ericl/Desktop/projects/wuziqi/network.py:45) | 新增 architecture 参数，默认 cnn，以兼容没有这个字段的旧 checkpoint。 |
| [46](/Users/ericl/Desktop/projects/wuziqi/network.py:46) | 新增 residual_blocks 参数，默认 3，允许后续控制规模。 |
| [52–59](/Users/ericl/Desktop/projects/wuziqi/network.py:52) | 检查架构名称和残差块数量，防止配置错误。 |
| [61–70](/Users/ericl/Desktop/projects/wuziqi/network.py:61) | 旧 cnn 分支保留原模块名称、参数形状及初始化顺序；旧模型的计算结果保持一致。 |
| [71–77](/Users/ericl/Desktop/projects/wuziqi/network.py:71) | 新分支使用输入卷积、指定数量的残差块、全局门控。 |
| [79–87](/Users/ericl/Desktop/projects/wuziqi/network.py:79) | 策略头逻辑保留；由配置选择卷积核大小，新训练使用 3×3。 |
| [89–97](/Users/ericl/Desktop/projects/wuziqi/network.py:89) | 旧价值头完整保留，确保旧 checkpoint 可加载。 |
| [98–106](/Users/ericl/Desktop/projects/wuziqi/network.py:98) | 新价值头先全局平均汇总，再用通道特征预测价值；不再为 361 个固定坐标分配独立的第一层全连接权重。仍保留 ReLU 与 tanh。 |
| [108–113](/Users/ericl/Desktop/projects/wuziqi/network.py:108) | forward 的输入输出接口不变，MCTS 与自我对弈无需适配输出形状。 |
| [116–136](/Users/ericl/Desktop/projects/wuziqi/network.py:116) | masked_policy 未修改：非法动作屏蔽与概率归一化保持原样。 |
| [140–144](/Users/ericl/Desktop/projects/wuziqi/network.py:140) | 直接运行 network.py 的形状检查改用新架构。 |

### train.py

| 修改后行号 | 修改及原因 |
|---|---|
| [194–197](/Users/ericl/Desktop/projects/wuziqi/train.py:194) | 整理配置字典缩进，保留 channels=32、value_hidden=64、policy_kernel=3。 |
| [198](/Users/ericl/Desktop/projects/wuziqi/train.py:198) | architecture 改为 resnet_global，后续新训练实际启用新结构。 |
| [199](/Users/ericl/Desktop/projects/wuziqi/train.py:199) | 将 residual_blocks=3 纳入配置，也会随 checkpoint 保存。 |
| [202](/Users/ericl/Desktop/projects/wuziqi/train.py:202) | 原有配置构造逻辑继续生效；自我对弈快照、固定初始对手和 checkpoint 的重建配置都保持一致。 |

数据增强、两种损失、优化器和训练次数均未修改。评估代码已经通过 model_config 重建网络，因此无须修改 evaluate.py。

### self_play.py

| 修改后行号 | 修改及原因 |
|---|---|
| [80–84](/Users/ericl/Desktop/projects/wuziqi/self_play.py:80) | 独立示例显式使用新架构，避免直接运行 self_play.py 时仍使用默认的旧 cnn。 |
| [85](/Users/ericl/Desktop/projects/wuziqi/self_play.py:85) | eval 模式继续保留。 |

play_game 函数、自我对弈标签和动作选择规则未修改。

## 3. 已完成验证

以下均为本次实际执行的检查：

- 三个修改文件的语法检查通过。
- 新结构 batch=1、2、3 的输出形状正确，价值有限且处于 [-1,1]。
- 三个随机初始化种子下，Policy 与 Value 均能将有限的非零梯度传到残差主干和全局门控。只计算梯度，没有 optimizer.step。
- 将残差分支置零后，非负输入通过该块保持不变。
- 固定诊断权重，检查左上角两个动作的 logit 差对右下角输入的梯度：启用全局门控时约为 -4.06247e-8，禁用门控时为 0。这验证的是相对动作分数，不只是会被 softmax 抵消的公共偏移。
- 22 个现有 checkpoint 全部严格加载成功，推理输出与修改前逐项完全一致；旧 cnn 的随机初始化结果也一致。
- 新网络配置和权重在内存中序列化、重新加载成功。
- 实际新网络运行了 8 次模拟的 MCTS 检查，合法性、访问次数和原棋盘保持正确。
- 完成一局新网络自我对弈：114 步，白棋获胜，约 0.978 秒（本次 CPU、2 线程）。逐条检查样本形状、策略合法性和结果视角，网络权重未变。这不是棋力评估或完整性能基准。
- 没有启动训练，没有写入新的 checkpoint 或 history.jsonl。验证脚本没有新增持久化测试文件。

## 4. 当前边界与后续工作

- 新模型拥有全局信息通路，不等于已经学会五子棋。
- 原先的旋转/镜像训练增强保留；当前网络不保证精确的旋转不变性。
- mcts.py 没有修改。此前发现的低预算探索不足仍然存在，不能把本次网络升级当作搜索问题已经解决。
- 当前新训练配置会从头初始化新架构，不会把旧 cnn 权重直接当成新网络权重恢复。
- 暂未进行新的训练或棋力对比。下一步应单独验收 MCTS。

## 5. 完整逐行 diff

`-` 表示删除的旧行，`+` 表示新增的行。`@@` 给出修改前后的起始行及行数，空白行和缩进修改也包含在内。

```diff
--- before/network.py
+++ after/network.py
@@ -1,5 +1,39 @@
 import torch
 from torch import nn
+
+
+class ResidualBlock(nn.Module):
+    """Learn a correction to the incoming board features."""
+
+    def __init__(self, channels):
+        super().__init__()
+        self.layers = nn.Sequential(
+            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
+            nn.ReLU(),
+            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
+        )
+        self.activation = nn.ReLU()
+
+    def forward(self, x):
+        return self.activation(x + self.layers(x))
+
+
+class GlobalContext(nn.Module):
+    """Use a whole-board summary to gate each feature channel."""
+
+    def __init__(self, channels):
+        super().__init__()
+        hidden = max(1, channels // 4)
+        self.gate = nn.Sequential(
+            nn.AdaptiveAvgPool2d(1),
+            nn.Conv2d(channels, hidden, kernel_size=1),
+            nn.ReLU(),
+            nn.Conv2d(hidden, channels, kernel_size=1),
+            nn.Sigmoid(),
+        )
+
+    def forward(self, x):
+        return x * self.gate(x)
 
 
 class GomokuNetwork(nn.Module):
@@ -8,20 +42,39 @@
         channels=32,
         value_hidden=64,
         policy_kernel=1,
+        architecture="cnn",
+        residual_blocks=3,
     ):
         super().__init__()
 
         if policy_kernel not in (1, 3):
             raise ValueError("Policy kernel must be 1 or 3")
+        if architecture not in ("cnn", "resnet_global"):
+            raise ValueError("Unknown network architecture")
+        if (
+            isinstance(residual_blocks, bool)
+            or not isinstance(residual_blocks, int)
+            or residual_blocks < 1
+        ):
+            raise ValueError("Residual blocks must be a positive integer")
 
-        self.trunk = nn.Sequential(
-            nn.Conv2d(3, channels, kernel_size=3, padding=1),
-            nn.ReLU(),
-            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
-            nn.ReLU(),
-            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
-            nn.ReLU(),
-        )
+        # Preserve the original module names and shapes for old checkpoints.
+        if architecture == "cnn":
+            self.trunk = nn.Sequential(
+                nn.Conv2d(3, channels, kernel_size=3, padding=1),
+                nn.ReLU(),
+                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
+                nn.ReLU(),
+                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
+                nn.ReLU(),
+            )
+        else:
+            self.trunk = nn.Sequential(
+                nn.Conv2d(3, channels, kernel_size=3, padding=1),
+                nn.ReLU(),
+                *(ResidualBlock(channels) for _ in range(residual_blocks)),
+                GlobalContext(channels),
+            )
 
         self.policy_head = nn.Sequential(
             nn.Conv2d(
@@ -33,14 +86,24 @@
             nn.Flatten(start_dim=1),
         )
 
-        self.value_head = nn.Sequential(
-            nn.Conv2d(channels, 1, kernel_size=1),
-            nn.Flatten(start_dim=1),
-            nn.Linear(19 * 19, value_hidden),
-            nn.ReLU(),
-            nn.Linear(value_hidden, 1),
-            nn.Tanh(),
-        )
+        if architecture == "cnn":
+            self.value_head = nn.Sequential(
+                nn.Conv2d(channels, 1, kernel_size=1),
+                nn.Flatten(start_dim=1),
+                nn.Linear(19 * 19, value_hidden),
+                nn.ReLU(),
+                nn.Linear(value_hidden, 1),
+                nn.Tanh(),
+            )
+        else:
+            self.value_head = nn.Sequential(
+                nn.AdaptiveAvgPool2d(1),
+                nn.Flatten(start_dim=1),
+                nn.Linear(channels, value_hidden),
+                nn.ReLU(),
+                nn.Linear(value_hidden, 1),
+                nn.Tanh(),
+            )
 
     def forward(self, x):
         features = self.trunk(x)
@@ -74,11 +137,15 @@
 
 
 if __name__ == "__main__":
-    model = GomokuNetwork(policy_kernel=3)
+    model = GomokuNetwork(
+        policy_kernel=3,
+        architecture="resnet_global",
+        residual_blocks=3,
+    )
     x = torch.randn(1, 3, 19, 19)
 
     with torch.no_grad():
         policy, value = model(x)
 
     print("Policy shape:", policy.shape)
-    print("Value shape:", value.shape)+    print("Value shape:", value.shape)
--- before/train.py
+++ after/train.py
@@ -192,11 +192,13 @@
     )
 
     model_config = {
-    "channels": 32,
-    "value_hidden": 64,
-    "policy_kernel": 3,
+        "channels": 32,
+        "value_hidden": 64,
+        "policy_kernel": 3,
+        "architecture": "resnet_global",
+        "residual_blocks": 3,
     }
-    
+
     model = GomokuNetwork(**model_config)
 
     # This opponent stays unchanged throughout the run.
@@ -442,4 +444,4 @@
 
 
 if __name__ == "__main__":
-    train()+    train()
--- before/self_play.py
+++ after/self_play.py
@@ -77,7 +77,11 @@
 def main():
     torch.manual_seed(0)
 
-    model = GomokuNetwork()
+    model = GomokuNetwork(
+        policy_kernel=3,
+        architecture="resnet_global",
+        residual_blocks=3,
+    )
     model.eval()
 
     started = time.perf_counter()
@@ -110,4 +114,4 @@
 
 
 if __name__ == "__main__":
-    main()+    main()
```


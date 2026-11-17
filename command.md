# Gomoku AlphaZero：常用命令（按重要程度排序）

每个终端先运行：

```bash
conda activate ml
cd /Users/ericl/Desktop/projects/gomoku
```

---

## 1. 让它自己下（自我对弈训练）

```bash
caffeinate -i python train.py --rounds 300
```

- **会自动接着上次的训练**：
  - 上次是自我对弈 → 精确续训同一个 run（经验池、优化器、随机数都恢复）。
  - 上次是和你下（网页学习）→ 从网页学到的模型开新 run，**并带上上次自我对弈的经验池**。
- **复习你赢它的棋**（自动）：
  1. **复盘**：开始训练前，新的输棋会用 4096 次深度搜索重新分析它下的每一步，学深度搜索找到的正确走法，而不是当时的错棋。每局只复盘一次，几十局约几分钟。
  2. **复习**：每批训练数据里约 10% 来自你和它下过的棋（复盘过的用复盘结果）。
  3. **重下**：约 20% 的自我对弈（每轮约 1 局）直接从它输给你的某个局面开始（随机旋转/翻转），让它在出错的地方自己反复下。这些对局不计入黑白胜率。
- `--rounds 300` = **再跑** 300 轮，约一整夜（估计值）。
- 每轮结束都会保存。**随时可以 Ctrl+C**，最多丢掉正在进行的那一轮；之后的 worker 报错可以忽略。再运行同一条命令就会继续。
- 想从某个模型开新 run：加 `--from <file.pt>`。
- 单独复盘（一般不需要，`train.py` 会自动做）：`python review.py`

## 2. 跟它下（网页学习，http://127.0.0.1:8000）

```bash
python -m web_app.server --learn --imitate 0.7
```

- **自动接着最新的训练结果**（自我对弈或上次网页学习）。
- 每下完一局，马上用这局训练并自动保存（成为最新模型）。下次「让它自己下」会从这里接着。
- `--imitate 0.7`：同时学你的走法（你比它强时很有用）。
- 你所有的对局都会保存，以后每次网页学习和自我对弈都会复习（输棋还会复盘、重下，见第 1 节）。
- **不要和自我对弈同时运行**（它们会各自训练一份模型）。先 Ctrl+C 停掉一个，再开另一个。

## 3. 检查训练是否健康

```bash
RUN=$(ls -td checkpoints/*_sims512 | grep -v _web_ | head -1 | xargs basename); python -c "
import json,re,sys
run=sys.argv[1]; h=[json.loads(l) for l in open(f'checkpoints/{run}/history.jsonl')]
a=h[max(0,len(h)-51)] if len(h)>50 else None; b=h[-1]['self_play']
bw=b['black_wins']-(a['self_play']['black_wins'] if a else 0); ww=b['white_wins']-(a['self_play']['white_wins'] if a else 0)
L=[int(m) for m in re.findall(r'in (\d+) moves',open(f'game_logs/selfplay_{run}.txt').read())][-300:]
p=[x['metrics']['policy_loss'] for x in h[-50:]]; v=[x['metrics']['value_loss'] for x in h[-50:]]
print(f'rounds {len(h)} | last 50 rounds: black wins {bw/(bw+ww):.0%} | policy loss {sum(p)/len(p):.2f} | value loss {sum(v)/len(v):.2f} | last 300 games: mean length {sum(L)/len(L):.1f}, <=11 moves {sum(x<=11 for x in L)/len(L):.0%}')
" $RUN
```

自动检查最新的自我对弈 run（要查别的 run，把 `RUN=$(...)` 换成 `RUN=<run 名字>`）。和修复前的 1000 轮 run 对比：

| 指标 | 修复前（旧搜索） | 希望看到 |
|---|---|---|
| 黑棋胜率 | 86% | 往下降（五子棋先手本来占优，不必到 50%） |
| 平均对局长度 | 24 步 | 明显更长（试验中约 40 步） |
| 11 步内结束的对局 | 33% | 很少（试验中约 4%） |
| policy loss | 3.02，卡住不降 | 慢慢降到 3.0 以下 |
| value loss | 0.41 | 不要接近 0（接近 0 = 一方总赢） |

## 4. 训练后：新模型和 best 比赛

```bash
python promote.py
```

- 最新模型对 best 下 40 局，得分 **≥ 60%** 才会成为新的 best，否则 best 不变。
- 每次结果都记在 `checkpoints/promotions.jsonl`。
- 指定某个模型：`python promote.py --candidate checkpoints/<RUN>/round_NNN_gamesNNNN.pt`

## 5. 只下棋、不马上学习（网页）

| 命令 | 用哪个模型 |
|---|---|
| `python -m web_app.server` | **best**（每局新游戏重新检查） |
| `python -m web_app.server --latest` | **最新**存档 |

- 不会马上训练，训练时可以同时开。
- **每局下完都会保存**（`checkpoints/<时间>_web_play/`），下次 `python train.py` 会复盘、复习、重下它输掉的棋。不想保存：加 `--no-save`。
- 加 `--imitate 0.7` 会把你的走法也存下来，让它以后学。
- AI 默认执黑。加 `--ai-color white` 让 AI 执白（你先下），`--ai-color alternate` 每局轮换。
- 都可以加 `--checkpoint <file.pt>` 指定模型。

## 6. 战术诊断（只读：会不会堵四、会不会一步胜）

```bash
python diagnose_defense.py --checkpoint checkpoints/<RUN>/round_NNN_gamesNNNN.pt
```

每行 `PASS=True` 才算对。默认用 128 和 512 次模拟各测一次。

## 7. 改代码后跑测试

```bash
python -B -m unittest test_alphazero test_maintenance test_web_app test_review
```

---

## 参考

- **`python train.py` 内部做什么**：精确续训用 `resume_from`（参数必须和原 run 一样，自动读取）；从别的模型开新 run 用 `init_checkpoint`（并用 `init_replay_from` 带上经验池）。如果有更新的存档（比如网页学习），程序会拒绝直接 `resume_from`，以免丢掉它。
- **run 文件夹**可以随项目移动，但**不能改名**。
- **你的对局**存在 `checkpoints/*_web_*/game_*.npz`（复盘结果在旁边的 `game_*.review.npz`，原文件不变；删掉 `.review.npz` 就会重新复盘）。`checkpoints/web_buffer_start.txt` 之前的对局不会被使用（由 `--learn --fresh` 设置）。
- **训练设置改了**（比如代码更新）：`python train.py` 不再精确续训旧 run，而是从它最新的模型开新 run，并带上经验池。
- **latest vs best**：latest 是最新的 `.pt` 存档（自我对弈或网页学习都算）；best 只能由 `promote.py` 改变（记录在 `checkpoints/best.json`）。最新不等于最强。
- **`evaluate.py`**：对从存档开始训练的 run 或网页学习的模型，必须加 `--opponent <file.pt>`。
- **重设 best（一般不需要）**：`python promote.py --set-baseline <file.pt>`

# Gomoku AlphaZero：常用命令（按重要程度排序）

每个终端先运行：

```bash
conda activate ml
cd /Users/ericl/Desktop/projects/gomoku
```

---

## 1. 开始新训练（用修好 FPU 的搜索）

从 1000 轮自我对弈的模型出发（它通过了全部战术诊断，best 模型没有）：

```bash
caffeinate -i python -c "from train import train; train(iterations=300, games_per_iteration=6, simulations=512, updates_per_iteration=50, batch_size=64, eval_games=2, eval_every=1000, self_play_workers=6, init_checkpoint='checkpoints/2026-11-11_0023_sims512/round_1000_games6000.pt')"
```

- 开始时会打印 `Run directory: .../checkpoints/<RUN>`，**记下 `<RUN>` 名字**，续训和检查都要用。
- 修好 FPU 后对局大约长了一倍，每轮大约慢 3 倍：300 轮大约要一整夜（估计值）。
- 每轮结束都会保存。**随时可以 Ctrl+C**，最多丢掉正在进行的那一轮。之后的 worker 报错可以忽略。

## 2. 中断后续训（接着同一个 run）

```bash
caffeinate -i python -c "from train import train; train(iterations=300, games_per_iteration=6, simulations=512, updates_per_iteration=50, batch_size=64, eval_games=2, eval_every=1000, self_play_workers=6, resume_from='checkpoints/<RUN>/resume_state.bin')"
```

- 除 `iterations` 外，**参数必须和第 1 步完全一样**，否则会报错。不要加 `init_checkpoint`。
- `iterations=300` 表示「**再跑** 300 轮」。
- 经验池、优化器、随机数都会恢复，和没中断过的结果完全一样。
- run 文件夹可以随项目移动，但**不能改名**。
- 忘了 `<RUN>` 名字：`ls -t checkpoints | head -3`

## 3. 检查训练是否健康

```bash
RUN=<RUN>; python -c "
import json,re,sys
run=sys.argv[1]; h=[json.loads(l) for l in open(f'checkpoints/{run}/history.jsonl')]
a=h[max(0,len(h)-51)] if len(h)>50 else None; b=h[-1]['self_play']
bw=b['black_wins']-(a['self_play']['black_wins'] if a else 0); ww=b['white_wins']-(a['self_play']['white_wins'] if a else 0)
L=[int(m) for m in re.findall(r'in (\d+) moves',open(f'game_logs/selfplay_{run}.txt').read())][-300:]
p=[x['metrics']['policy_loss'] for x in h[-50:]]; v=[x['metrics']['value_loss'] for x in h[-50:]]
print(f'rounds {len(h)} | last 50 rounds: black wins {bw/(bw+ww):.0%} | policy loss {sum(p)/len(p):.2f} | value loss {sum(v)/len(v):.2f} | last 300 games: mean length {sum(L)/len(L):.1f}, <=11 moves {sum(x<=11 for x in L)/len(L):.0%}')
" $RUN
```

和修复前的 1000 轮 run 对比：

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

## 5. 网页对战 (http://127.0.0.1:8000)

| 命令 | 用哪个模型 | 会从你的棋里学习吗？ |
|---|---|---|
| `python -m web_app.server` | **best**（每局新游戏重新检查） | 不会；训练时可以同时开 |
| `python -m web_app.server --latest` | **最新**存档 | 不会；训练时可以同时开 |
| `python -m web_app.server --learn --imitate 0.7` | 从 best 开始 | **会**；它的存档会变成最新模型 |

- **不要同时运行 `--learn` 和自我对弈训练**，它们各自训练自己的一份模型。
- 任何一个命令都可以加 `--checkpoint <file.pt>` 指定模型。

## 6. 战术诊断（只读：会不会堵四、会不会一步胜）

```bash
python diagnose_defense.py --checkpoint checkpoints/<RUN>/round_NNN_gamesNNNN.pt
```

每行 `PASS=True` 才算对。默认用 128 和 512 次模拟各测一次。

## 7. 改代码后跑测试

```bash
python -B -m unittest test_alphazero test_maintenance test_web_app
```

---

## 参考

- **`init_checkpoint` vs `resume_from`**：`init_checkpoint` = 新开一个 run（经验池清空，随机数重新开始）；`resume_from` = 接着同一个 run。
- **latest vs best**：latest 是最新的 `.pt` 存档（自我对弈或网页学习都算）；best 只能由 `promote.py` 改变（记录在 `checkpoints/best.json`）。最新不等于最强。
- **`evaluate.py`**：对从存档开始训练的 run 或网页学习的模型，必须加 `--opponent <file.pt>`。
- **重设 best（一般不需要）**：`python promote.py --set-baseline <file.pt>`

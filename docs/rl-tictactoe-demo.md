# 井字棋 Q-learning Demo

`src/bin/rl_tictactoe` 是第 12 章配套的**表格式 Q-learning** 示例：智能体执 X，通过与环境反复对弈，把 Q 值从终局倒灌回开局。

## 运行

工作目录必须是 `src/`：

```bash
cd src
./bin/rl_tictactoe
```

默认配置：训练 **30000** 局（对手为随机 O）、`α=0.5`、`γ=0.99`、ε 从 1.0 衰减到 0.05。结束后分别对随机、最优两种对手各评估 1000 局（greedy、不探索）。

典型输出（`rand-seed=0`）——**注意：跟谁练就擅长打谁**：

- 默认（对随机训练）：对随机对手 **不败率 ~99.7%**（约 984 胜 / 13 和 / 3 负），但对最优 minimax 对手**几乎全输**——它没见过高手的严密下法。
- 加 `--opponent optimal`（对最优训练）：对最优对手 **稳守和棋**（0 胜 0 负、全和，不败率 100%；井字棋双方无失误必和），但对随机对手只剩 **~59% 胜率**（学的是「稳不输」而非「逮失误猛攻」）。

> 默认 3 万局时确实是「跟谁练就擅长打谁」；不过把局数加到 30 万（`--episodes 300000`）后，同一默认配置能同时做到对随机 100% 不败、对最优全部守和——训练分布决定天花板，训练量决定你能不能摸到它（电子书第 12 章实验台加载的 `q_table.json` 就是这份 30 万局导出）。

## 常用参数

```bash
./bin/rl_tictactoe --episodes 50000 --eval-games 1000 --rand-seed 0
./bin/rl_tictactoe --opponent optimal          # 训练时也对抗最优对手（更难，收敛慢）
./bin/rl_tictactoe --show-sample               # 打印一局 greedy 对局
./bin/rl_tictactoe --play                      # 训练后与你下一盘（你是 O，stdin 输入 0–8）
```

棋盘位置编号：

```
0|1|2
-+-+-
3|4|5
-+-+-
6|7|8
```

## 代码对应

| 书里概念 | 源码 |
|---------|------|
| 环境 MDP | `src/deeplearning/rl/tic_tac_toe_env.*` |
| Q 表 + TD 更新 | `src/deeplearning/rl/tabular_q_learning.*` |
| 训练循环 | `src/demo/rl_tictactoe/main.cpp` |
| 单元测试 | `src/test/rl/rl_test.h` |

状态用 9 格棋盘的三进制整数编码（每格空/X/O → 0/1/2）。奖励：赢 +1、输 −1、和 0。更新式与书中 Q-learning 一致：

```
Q(s,a) ← Q(s,a) + α · [ r + γ·max Q(s′,·) − Q(s,a) ]
```

## 与第 12 章 / 第 25 章的关系

- 概念讲解：[第 12 章](https://chenxuan520.github.io/deeplearning/chapter-12.html)（强化学习骨架）
- **逐行代码导读**：[第 25 章](https://chenxuan520.github.io/deeplearning/chapter-25.html)（与第 23、24 章同结构的实战章）

- 老鼠走廊 = 一维、奖励稀疏；井字棋 = 二维动作空间（9 个落子位），但状态仍有限，适合 **Q 表**。
- 状态再多（如围棋、雅达利画面）就要上 **DQN**；动作再大（词表）就要 **策略梯度 / PPO**（见 RLHF）。

全书实战部分三份程序对照：第 23 章 MNIST（监督）、第 24 章 mini-LM（生成）、**第 25 章本 demo（强化学习）**。

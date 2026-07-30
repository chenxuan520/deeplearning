# MNIST Demo

`src/demo/mnist/main.cpp` 是用 `deeplearning::NeuralNetwork` 在 MNIST 手写数字
数据集上做 10 类分类的最小可运行示例。本文记录当前 demo 的配置、运行方法,
以及从 baseline 升级到当前版本的设计决策。

## 1. 网络与训练配置

| 维度 | 值 |
|---|---|
| 网络结构 | 784 → 128 → 64 → 10 (~109K 参数) |
| 激活 | ReLU (隐层) |
| 输出 | Softmax + Cross-Entropy |
| 参数初始化 | He (配 ReLU) |
| 优化器 | Adam (β1=0.9, β2=0.999) |
| 学习率 | `WarmupCosineLR(base=1e-3, warmup=1 epoch, t_max=5 epochs, min=1e-5)` |
| Batch size | 64 |
| 训练轮数 | 5 epochs (≈ 4690 step) |
| 随机种子 | 42 (训练全程可复现) |
| 输入预处理 | `pixel / 255.0` (归一化到 [0,1]) |

输入预处理在 `src/demo/mnist/mnist_data.h::ReadMnistImages` 内完成,
`optimizer_bench` demo 也共用同一份预处理。

## 2. 运行

### 构建
```bash
cd src
./build.sh                         # 或 mkdir -p build && cd build && cmake .. && make
```

### 训练 + 评估
工作目录必须是 `src/` (因为 main.cpp 使用相对路径读取 MNIST 文件):
```bash
cd src
./bin/mnist
```

### 产出文件
- `src/demo/mnist/mnist/demo.v2.param`: 训完后保存的模型参数。下一次运行会
  自动加载这个文件继续训练 (lr 自动降为 `1e-4` 做微调)。
- 如果想冷启动重新训练, 删除该文件即可:
  ```bash
  rm src/demo/mnist/mnist/demo.v2.param
  ```

### Dropout 消融开关
- `./bin/mnist --dropout 0.2`: 给两个隐藏层加 inverted dropout (训练时按
  概率丢单元 + 幸存输出乘 `1/keep` 补量, 推理自动关闭), 取值 `[0, 1)`。
- 开启后使用独立的参数文件 `demo.dropout.param` (同样支持自动加载续训),
  不会覆盖基线的 `demo.v2.param`, 便于和基线做消融对比。
- 同一 seed=42 下实测 5 epochs: 基线 train 98.99% / test 97.82% (裂口
  1.17 个点); `--dropout 0.2` train 98.35% / test 97.51% (裂口 0.84 个点)。
  dropout 在全量 MNIST 上不提 test 精度, 但明显收窄 train/test 裂口。
- 过拟合场景 (`--train-limit 4000 --epochs 30`, 同 seed): 基线 train 冲到
  100.00% / test 93.62% / test_loss 0.0410; `--dropout 0.2` train 99.82% /
  test **94.21%** / test_loss **0.0362** —— 双指标全胜并阻止 100% 死记;
  `--dropout 0.5` 则因过量欠拟合 (train 98.38% / test 92.91%)。

### 其他参数
- `--epochs <int>`: 训练轮数, 默认 5。
- `--train-limit <int>`: 只用前 N 张训练图 (默认 0 = 全部 60000 张),
  用于构造小数据/过拟合场景。

### 输出示例
```
train_data: 60000  test_data: 10000
Init new model: 784 -> 128 -> 64 -> 10
Train: batch=64 epochs=5 steps_per_epoch=938 total_steps=4690 base_lr=0.001 (warmup-cosine)
epoch 1/5  train_loss=0.0281  test_loss=0.0317  test_acc=0.9458  lr=1.00e-03
epoch 2/5  train_loss=0.0159  test_loss=0.0202  test_acc=0.9670  lr=8.55e-04
epoch 3/5  train_loss=0.0103  test_loss=0.0150  test_acc=0.9734  lr=5.05e-04
epoch 4/5  train_loss=0.0074  test_loss=0.0132  test_acc=0.9774  lr=1.55e-04
epoch 5/5  train_loss=0.0068  test_loss=0.0128  test_acc=0.9782  lr=1.00e-05
Train finished in 418.0 sec
Final accuracy:  train=0.9899  test=0.9782
Saved model to ./demo/mnist/mnist/demo.v2.param
```

> 训练时间在 Apple Silicon 单线程 Debug 构建下测得。这套实现是纯
> `std::vector` 三重循环, 没有 BLAS/SIMD, 评估阶段又会全量过一次
> train probe (5000) + test (10000) 单样本前向, 所以 wall-clock 偏慢。

## 3. 升级历程: 从 92.67% → 97.82%

| 指标 | Baseline (v1) | 当前 (v2) | 变化 |
|---|---|---|---|
| 测试准确率 | 92.67% | **97.82%** | **+5.15pp** |
| 训练准确率 | — | 98.99% | — |
| 错误样本数 | 733 / 10000 | 218 / 10000 | **-70%** |
| 训练耗时 | ~138 s | ~418 s | ↑ (含全量 eval) |

v1 (旧版本) 配置: 784 → 20 → 10, sigmoid + uniform random, SGD, batch=1,
lr=0.2/0.02, 训 1.5 个 epoch (`Train(..., 1.5*N, 1, 0.2)`)。

### 关键改动 (按影响排序)

1. **像素归一化代替二值化** (`mnist_data.h`)
   - v1 把像素 `>0` 的全部置为 1, 丢掉了灰度信息。
   - v2 改为 `pixel / 255.0`, 保留 0~1 之间的连续值。这一项是单项收益最大的修正,
     在 v2 网络结构下也是上 97% 的前提。

2. **网络扩容: 20 → 128/64** (`main.cpp`)
   - v1 隐藏层只有 20 个神经元 (~15.9K 参数), 容量不够装下 MNIST 的模式。
   - v2 改为 784→128→64→10 (~109K 参数), 给优化器留出充足表达空间。

3. **ReLU + He init 代替 Sigmoid + Uniform** (`main.cpp`)
   - Sigmoid 在深层 + 大输入下饱和, 梯度消失。
   - ReLU 无饱和, 但要配 He 初始化保证方差稳定: `set_param_init_function(PARAM_INIT_HE)`。

4. **Softmax + Cross-Entropy 闭环** (`main.cpp`)
   - v1 用了 `LOSS_CROSS_ENTROPY` 但 softmax 是 `SOFTMAX_NONE` (默认), 输出层
     没归一化, 与 CE 的"概率分布"假设不匹配。
   - v2 显式 `set_softmax_function(SOFTMAX_STD)`, 配上分类任务标配。

5. **Adam 代替 SGD** (`main.cpp`)
   - 自适应步长, 对学习率不敏感, 收敛快。
   - 直接 `set_optimizer_function(OPTIMIZER_ADAM)`, 配 `lr=1e-3`。

6. **WarmupCosineLR** (`main.cpp`)
   - 1 epoch 线性 warmup 防止 Adam 早期方差估计不稳, 之后 cosine 退火到 1e-5。
   - 比固定 lr 在尾段精度高 ~0.5pp。

7. **Batch size 1 → 64** (`main.cpp`)
   - 单样本 SGD 梯度噪声大, 对 Adam 这种带二阶矩估计的优化器也不友好。
   - batch=64 同时让 forward / backward 走批量化代码路径, 单 step 慢但每个样本
     的等效时间短。

8. **`PredictBatch` 加速评估** (`main.cpp`)
   - v1 评估时一个个 `Predict` 调 10000 次。
   - v2 用 `PredictBatch` 一次过完, 时间约为原来的一半左右。

### 代码骨架
```c++
NeuralNetwork net;
net.Init({784, 128, 64, 10});
net.set_random_seed(42);
net.set_param_init_function(PARAM_INIT_HE);   // 必须在 set_random_seed 之后
net.set_activate_function(ACTIVATE_RELU);
net.set_softmax_function(SOFTMAX_STD);
net.set_loss_function(LOSS_CROSS_ENTROPY);
net.set_optimizer_function(OPTIMIZER_ADAM);

int total_steps = 5 * (60000 + 63) / 64;
net.set_lr_scheduler(std::make_shared<WarmupCosineLR>(
    1e-3, /*warmup=*/938, /*t_max=*/total_steps, /*min=*/1e-5));

net.Train(train_data, train_target, callback, total_steps, /*batch=*/64, 1e-3);
```

## 4. 还能往哪儿涨

97.8% 对纯 MLP 基本是天花板, 想继续提升需要超出当前库的能力或加更多训练:

- **训更长 + Cosine 重启**: 当前 epoch=5 已被 cosine 压到 lr=1e-5 收尾,
  再训也涨不动。可以试 `epoch=10` 重新跑 cosine。
- **加 weight decay**: `net.optimizer_function()->set_weight_decay(1e-4)`,
  缩小 train(98.99%) 和 test(97.82%) 之间的 ~1pp 过拟合差距。
- **数据增强**: 随机平移 1~2 像素 / 小角度旋转, MLP 也能再涨 0.3~0.5pp。
- **Dropout**: 库已支持 `set_dropout_rate` (隐藏层 inverted dropout), demo 自带
  `--dropout` 开关 (见上文「Dropout 消融开关」)。实测它能把 train/test 裂口
  从 ~1.2 个点收到 ~0.7, 但在全量 MNIST 上不提 test 精度——适合数据更少、
  过拟合更严重的场景。
- **换 CNN**: 仓库现在已有最小 `conv + pool` 实现（见 `src/deeplearning/cnn/` 与 `src/demo/cnn_mnist/`），但要稳定到 99%+ 仍通常需要更深的 CNN、更多通道和更完整的数据增强。

## 5. 与 Baseline 的兼容性

升级后的 demo 使用 `demo.v2.param` 作为模型缓存文件名, **与历史的
`demo.param` 完全独立**:
- `demo.v2.param`: 当前 v2 模型 (784-128-64-10, 归一化输入, Adam)。
- `demo.param`: 旧 v1 模型 (784-20-10, 二值化输入, SGD)。已不再被本 demo
  读写, 仅作历史参考。

如果以后想清掉旧文件:
```bash
rm src/demo/mnist/mnist/demo.param
```

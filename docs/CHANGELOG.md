# CHANGELOG

## Unreleased — 训练设施 + CNN/RNN 扩展

本次主要补齐 "现代神经网络训练" 中常见的几个组件, 并把书里提到的最小 CNN /
RNN 代码链路也落到仓库里, 让库不仅能跑通 MLP / CNN / RNN / Transformer,
还能比较合理地工程化使用. 没有引入新的依赖, 全部为 C++17 实现.

### 新增

#### CNN / RNN
- 新模块 `src/deeplearning/cnn/`:
  - `Conv2D` — 最小二维卷积层, 支持 stride / padding / 手写 forward/backward.
  - `MaxPool2D` — 最小最大池化层, forward 记录 argmax, backward 按路由回传.
  - `MiniCNNClassifier` — `conv + ReLU + max-pool + linear` 教学级分类器.
- 新模块 `src/deeplearning/rnn/`:
  - `SimpleRNN` — tanh 循环单元, 显式缓存整条序列并做 BPTT.
  - `MiniRNNLM` — 最小字符级 RNN 语言模型, 支持 next-token loss / perplexity /
    greedy 生成 / 梯度裁剪 / scheduler.
- 新 demo:
  - `src/demo/cnn_mnist` — 用最小 CNN 在 MNIST 子集上训练和评估.
  - `src/demo/rnn_char` — 用最小 RNN 在字符语料上做 next-token 训练与生成.
- 新测试:
  - `src/test/cnn/cnn_test.h` — 卷积、池化前反向与 toy 图像分类收敛.
  - `src/test/rnn/rnn_test.h` — RNN 已知权重前向与 toy 字符语料收敛 / 生成.

#### 优化器
- `OPTIMIZER_ADAM` — Adam (Kingma & Ba 2014), 带 bias correction.
- `OPTIMIZER_ADAMW` — AdamW (Loshchilov & Hutter 2017), 解耦 weight decay.
- `OPTIMIZER_RMSPROP` — RMSProp, 自适应步长但不带一阶矩.

所有优化器 (含已有的 SGD / Momentum) 都新增 `set_weight_decay(double)`. SGD /
Momentum / Adam / RMSProp 走 `L2-in-gradient` (`g += wd*w`); AdamW 走解耦的
`update += lr * wd * w`. Bias 始终不被衰减.

`OptimizerFunction` 接口扩展:
- 新增 `BeforeStep()` 虚函数 (默认 no-op, Adam/AdamW 推进 step 计数).
- `CalcChangeValue` 新增 `param_value` 形参 (默认 0), 用于实现 weight decay.

`NeuralNetwork::ApplyGradient` 相应改造:
1. 调 `optimizer->BeforeStep()` 一次.
2. 给每个参数传入它当前的值 (用于解耦 weight decay).
3. 在调度优化器之前可选地 `ClipGradients(B)`.

`NeuralNetwork` 新增重载:
- `set_optimizer_function(std::shared_ptr<OptimizerFunction>)`: 注入自定义实例,
  可设置 beta1 / beta2 / momentum 等超参.
- `optimizer_function()`: 读 / 调超参 (例如 `set_weight_decay`).

#### 学习率调度器 (新模块 `lr_scheduler/`)
- `StepDecayLR` — `lr = base * gamma^(step / step_size)`
- `ExponentialDecayLR` — `lr = base * gamma^step`
- `CosineAnnealingLR` — `lr = min + 0.5 * (base - min) * (1 + cos(π * step / T_max))`
- `WarmupCosineLR` — linear warmup + cosine, 现代 Transformer 训练标配

`NeuralNetwork::set_lr_scheduler(std::shared_ptr<LRScheduler>)` 之后, Train 会
在每个 minibatch step 调用 `scheduler->GetLR(step)` 并覆盖 `learning_rate_`.

#### 梯度裁剪
- `NeuralNetwork::set_gradient_clip_norm(double)` — 全局 L2 范数阈值, 超阈值整体缩放
- `NeuralNetwork::set_gradient_clip_value(double)` — 逐分量绝对值裁剪
- 两者可叠加 (先 by-value, 再 by-norm). 默认 ≤ 0 表示关闭.

#### 激活函数
- `ACTIVATE_LEAKY_RELU` — 默认 slope=0.01, 改善 dead ReLU 问题.
- `ACTIVATE_GELU` — 用 `std::erf` 的精确版本; Transformer / BERT / GPT 类常用.

`ActivateFunction` 接口扩展:
- 新增 `DerivActivate(input, output)` 2-arg 版本. 默认实现回退到老的
  `DerivActivate(output)`, 不影响已有激活. GELU 等需要 input 的激活重载这个版本.
- `NeuralNetwork` 新增 `neuron_preact_` 缓冲, 在 forward 时记录 z = b + W·x,
  backward 时把它喂给 `DerivActivate(input, output)`. 现有激活 (sigmoid / tanh /
  relu) 行为完全不变, 因为它们只看 output.

#### 可复现性
- `ParamInitFunction` 新增 `set_seed(int)`. seed=-1 (默认) 时仍走 `random_device`;
  seed≥0 时用确定性 mt19937.
- `NeuralNetwork::set_random_seed(n)` 会把同一个 seed 透传给 param_init, 让训练
  全流程跨次复现. `set_param_init_function(...)` 会读当前的 `rand_seed_` 透传.
- 历史上的 `MiniBatch BenchmarkSameSampleCount` 测试由于 Xavier init 用了
  非确定性 `random_device`, 在边界 acc 上偶现 flake. 修复后测试稳定通过.

#### 新 demo
- `src/demo/optimizer_bench`: 同一 MLP (`[input, 64, 32, output]`, ReLU,
  He init, cross-entropy + softmax) 下对比 SGD / Momentum / RMSProp / Adam /
  AdamW / Adam+CosineLR 的 wall-clock 时间和 train/test acc.
- 支持 MNIST 和合成数据 (`--skip-mnist`).

#### MNIST demo 升级 (v1 → v2)
`src/demo/mnist/main.cpp` 重写, 利用上述新增设施做完整升级:
- 网络: `784→20→10` (sigmoid + uniform random + SGD) → `784→128→64→10`
  (ReLU + He + Softmax + Cross-Entropy + Adam).
- 训练: `batch=1`, 固定 lr, 1.5 epoch → `batch=64`, `WarmupCosineLR`
  (1 epoch warmup + cosine 退火到 `1e-5`), 5 epochs.
- 评估: 逐样本 `Predict` → `PredictBatch` 一次批量前向.
- 输入: `mnist_data.h` 的像素从二值化 `>0 → 1` 修正为归一化 `pixel / 255.0`,
  保留灰度信息. `optimizer_bench` 也跟着受益.
- 模型缓存: `demo.param` → `demo.v2.param` (与历史 v1 完全隔离, 续训检测同名文件
  自动以 `lr=1e-4` 微调).
- 结果: 测试准确率 92.67% → **97.82%** (错误样本数 -70%).
- 升级历程 / 配置 / 调优思路写在 `docs/mnist-demo.md`.

#### 新测试
- `optimizer_test.h`: Adam / AdamW / RMSProp 收敛性, SGD with weight decay,
  Adam 步计数器, Adam vs AdamW 在带 wd 时的参数差异.
- `lr_scheduler_test.h`: 每种 scheduler 的关键步数值断言, 以及 Train 端到端集成.
- `grad_clip_test.h`: 故意大 lr + 离群点制造爆炸, 验证 clip_norm / clip_value
  能拉回 finite.
- `activate_test.h`: LeakyReLU / GELU 已知值断言, GELU 解析导数对比中心差分,
  factory 创建, 端到端收敛.

### 改动

- `OptimizerFunction::CalcChangeValue` 多了一个 `param_value` 参数 (默认 0,
  向后兼容). 自定义子类如果继承自旧接口, 编译会因为 override 签名不匹配报错,
  需要更新签名.
- `ActivateFunction` 多了一个 2-arg `DerivActivate(input, output)`, 默认 fallback,
  不需要改子类.
- `ParamInitFunction` 多了 `set_seed/seed/seed_`, 子类不需要改但建议读 `seed_`
  以支持可复现.

### 测试

52/52 tests pass, 且新增的 CNN / RNN 定向 smoke demo 可编译运行.
新增测试覆盖 optimizer / scheduler / grad clip / 激活函数 / mini-batch /
Transformer / CNN / RNN 等核心接口.

## 后续 roadmap (尚未实现, 仅记录方向)

- ~~Dropout 层 (训练 / eval 模式切换)~~ 已实现: `set_dropout_rate` 隐藏层
  inverted dropout, 训练时采样掩码 / 推理自动关闭, 见 `src/test/dropout_test.h`
- BatchNorm / LayerNorm 已部分存在于 transformer 里, 但还不是 NeuralNetwork 一等公民
- DataLoader (多线程读 + 自动 shuffle)
- Mixed precision (float vs double)
- SIMD / OpenBLAS 后端加速矩阵乘
- 更完整的 CNN 栈 (多层 conv / 多层池化 / 参数保存加载)
- LSTM / GRU
- 自动微分图 (autograd) — 目前是手动反向

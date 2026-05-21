# CHANGELOG

## Unreleased — 优化器与训练设施扩展

本次主要补齐 "现代神经网络训练" 中常见的几个组件, 让库不仅能跑通 MLP /
Transformer, 还能比较合理地工程化使用. 没有引入新的依赖, 全部为 C++17 实现.

### 新增

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

47/47 tests pass, 跨多次运行结果一致 (修了之前的 flaky benchmark).
新增的 13 个 test 覆盖了上面所有新接口.

## 后续 roadmap (尚未实现, 仅记录方向)

- Dropout 层 (训练 / eval 模式切换)
- BatchNorm / LayerNorm 已部分存在于 transformer 里, 但还不是 NeuralNetwork 一等公民
- DataLoader (多线程读 + 自动 shuffle)
- Mixed precision (float vs double)
- SIMD / OpenBLAS 后端加速矩阵乘
- CNN 算子 (conv + pool)
- 自动微分图 (autograd) — 目前是手动反向

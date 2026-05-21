# deeplearning
- 简单的 C++ 深度学习框架
## Author
-  **chenxuan**
## 项目结构
-  `src/deeplearning` 为库头文件与实现源码
-  `src/test` 为测试代码
-  `src/deeplearning/transformer` 为最小 Transformer 基础模块
-  `src/deeplearning/optimizer` 为优化器实现 (SGD / Momentum / Adam / AdamW / RMSProp)
-  `src/deeplearning/lr_scheduler` 为学习率调度器
-  `src/deeplearning/activate` 为激活函数 (含 Sigmoid / ReLU / Tanh / LeakyReLU / GELU)
-  `src/deeplearning/param_init` 为参数初始化器 (含确定性 seed 支持)
## 使用 demo
-  `src/demo` 中有 demo 代码,可以参考
    - mnist 为 mnist 数据集,使用代码 demo 默认配置下识别率约为 91%
    - optimizer_bench 为不同 optimizer 在 MNIST 上的对比 (SGD / Momentum / Adam / AdamW / RMSProp / Adam+CosineLR)
    - transformer_char 为最小字符级语言模型

## 核心特性
### 训练
- 真 mini-batch (向量化梯度累加 + 一次 ApplyGradient)
- 多种优化器: `OPTIMIZER_SGD` / `OPTIMIZER_MOMENTUM` / `OPTIMIZER_ADAM` / `OPTIMIZER_ADAMW` / `OPTIMIZER_RMSPROP`
- L2 weight decay (Adam 走耦合; AdamW 走解耦, bias 不衰减)
- 梯度裁剪: `set_gradient_clip_norm` / `set_gradient_clip_value`, 可叠加
- LR 调度: `StepDecayLR` / `ExponentialDecayLR` / `CosineAnnealingLR` / `WarmupCosineLR`,
  通过 `set_lr_scheduler(...)` 注入, Train 每个 step 自动应用
- 参数初始化: `ZERO` / `UNIFORM` / `NORMAL` / `XAVIER` / `HE`, 支持 `set_seed` 复现
- `set_random_seed(n)` 会自动把同样的 seed 透传给 param_init, 让训练结果跨次复现

### 推理
- `Predict(data, result)` 单样本
- `PredictBatch(batch, results)` 一次前向多个样本 (比循环 Predict 快 ~2x)

### 激活函数
- `ACTIVATE_SIGMOID` / `ACTIVATE_RELU` / `ACTIVATE_TANH`
- `ACTIVATE_LEAKY_RELU` (默认 slope=0.01, 改善 dead ReLU)
- `ACTIVATE_GELU` (Transformer / BERT 类常用; 使用 erf 精确版本)

### 模型序列化
-  `NeuralNetworkLoader::ExportParamToFile` / `ImportParamFromFile` 用二进制格式保存/读取参数

## Transformer 基础模块
-  `transformer/positional_encoding.h` 提供正弦位置编码
-  `transformer/token_embedding.h` 提供 token id 到向量的映射
-  `transformer/layer_norm.h` 提供按 token 归一化的 LayerNorm
-  `transformer/self_attention.h` 提供最小 multi-head self-attention
-  `transformer/transformer_block.h` 提供 attention + FFN + residual + layer norm 的最小 block
-  `transformer/transformer_encoder.h` 提供 block 堆叠后的 encoder
-  `transformer/mini_transformer_lm.h` 提供最小语言模型前向链路
-  `transformer/mini_transformer_lm_loader.h` 提供最小语言模型的保存与加载
-  `transformer/character_tokenizer.h` 提供字符级 tokenizer
-  `transformer/character_dataset.h` 提供 next-token 样本切片工具
-  `transformer/transformer_decoder.h` 提供带 causal mask 的 decoder-only 堆叠
-  `MiniTransformerLM` 支持 encoder / decoder 两种主干选择，以及 block 学习率比例配置
-  `MiniTransformerLM::Config` 可集中描述模型结构与主干配置，减少散落的初始化参数
-  当前提供一个可训练的小型字符级语言模型闭环：数据切片、训练、保存/加载、生成

## Transformer Demo
-  `src/demo/transformer_char` 提供一个最小字符级 demo
-  该 demo 重点演示 `CharacterDataset -> 训练 -> 保存/加载 -> Generate` 这条最小可用链路
-  当前 demo 默认走 decoder-only 路线，并演示多 block 训练
-  常用参数：`--prompt` `--generate-num` `--temperature` `--top-k` `--top-p` `--epochs` `--learning-rate` `--rand-seed` `--backbone` `--model-dim` `--head-num` `--feed-forward-dim` `--block-num` `--context-size` `--block-learning-rate-scale` `--model-file` `--config-file` `--attention-export-file` `--corpus` `--corpus-file` `--save-model` `--no-save-model` `--eval-only` `--force-train`
-  示例：`./bin/transformer_char --prompt ab --generate-num 8 --temperature 0.7 --top-k 2 --top-p 0.9 --backbone decoder --block-num 2 --force-train`
-  文件语料示例：`./bin/transformer_char --corpus-file ./demo/transformer_char/corpus.txt --prompt ab --force-train`
-  仅评估已有模型：`./bin/transformer_char --eval-only --model-file transformer_char_demo.param --prompt ab`
-  导出真实 attention 权重：`./bin/transformer_char --attention-export-file ./docs/attention-guide/attention-sample.json --force-train`
-  项目讲解静态页：`docs/attention-guide/index.html`

## Optimizer Benchmark Demo
-  `src/demo/optimizer_bench` 在同一 MLP (`[input, 64, 32, output]`, ReLU, He init, cross-entropy)
   下对比不同优化器的训练时间和准确率
-  支持 MNIST 数据 (默认从 `./demo/mnist/mnist/` 读取) 和合成 4 类数据 (`--skip-mnist`)
-  示例: `./bin/optimizer_bench --steps 3000 --batch 64`
-  示例 (无 MNIST): `./bin/optimizer_bench --skip-mnist --steps 2000 --batch 16`

### MNIST 上的实测对比 (3000 step, batch=64, lr 各自最佳)
| optimizer            | test acc | train acc | time   |
|----------------------|---------:|----------:|-------:|
| SGD lr=0.05          |   95.77% |    96.10% |  ~79s  |
| Momentum lr=0.05     |   95.44% |    96.71% |  ~81s  |
| RMSProp lr=0.001     | **96.52%** | 97.43% |  ~80s  |
| Adam lr=0.001        |   96.26% |    97.47% |  ~85s  |
| AdamW lr=0.001 wd=1e-4 | 96.17% |    97.40% |  ~86s  |
| Adam+CosineLR        |   95.90% |    96.57% |  ~83s  |

> 说明: 这是教学库的 CPU 实现 (纯 std::vector + 三重循环), 数据未做 BLAS / SIMD 加速.
> Adam 系列在相同 step 下比 SGD 平均高 0.5~1pp 测试准确率.

## Quick Start
1. `mkdir build;cmake ..;sudo make install` 安装头文件与库文件
```c++
#include "deeplearning/neural_network.h"
using namespace deeplearning;

int main() {
  NeuralNetwork network((std::vector<int>() = {2, 1, 1}));
  std::vector<std::vector<double>> data = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  std::vector<std::vector<double>> target = {{0}, {0}, {1}, {1}};

  auto print_func = [](const NeuralNetwork &network, double loss_sum) {
    std::cout << loss_sum << std::endl;
  };
  auto rc = network.Train(data, target, print_func);
  if (rc != NeuralNetwork::SUCCESS) {
    std::cout << "Train failed" << std::endl;
    return -1;
  }
  std::cout << "Train success" << std::endl;
  std::vector<double> test_data = {1, 1.2}, result;
  rc = network.Predict(test_data, result);
  if (rc != NeuralNetwork::SUCCESS) {
    std::cout << "Predict failed" << std::endl;
    return -1;
  }
  return 0;
  std::cout << "Predict: " << result[0] << std::endl;
}
```

## Advanced Usage
### 使用 Adam + 学习率调度 + 梯度裁剪
```c++
#include "deeplearning/neural_network.h"
#include "deeplearning/lr_scheduler/warmup_cosine_lr.h"

using namespace deeplearning;

NeuralNetwork net;
net.Init({784, 128, 10});
net.set_random_seed(42);                                // 训练全程可复现
net.set_param_init_function(ParamInitType::PARAM_INIT_HE);
net.set_activate_function(ActivateType::ACTIVATE_GELU); // BERT 风格
net.set_loss_function(LossType::LOSS_CROSS_ENTROPY);
net.set_softmax_function(SoftmaxType::SOFTMAX_STD);
net.set_optimizer_function(OptimizerType::OPTIMIZER_ADAMW);
net.optimizer_function()->set_weight_decay(1e-4);       // AdamW 解耦衰减
net.set_gradient_clip_norm(1.0);                        // 防梯度爆炸
net.set_lr_scheduler(                                   // warmup + cosine
    std::make_shared<WarmupCosineLR>(/*base_lr=*/1e-3,
                                      /*warmup_steps=*/500,
                                      /*t_max=*/10000));
net.Train(data, target, nullptr, /*steps=*/10000, /*batch=*/64);
```

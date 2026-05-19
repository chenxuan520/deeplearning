# deeplearning
- 简单的C++深度学习框架
## Author
-  **chenxuan**
## 项目结构
-  `src/deeplearning` 为库头文件与实现源码
-  `src/test` 为测试代码
-  `src/deeplearning/transformer` 为最小 Transformer 基础模块
## 使用demo
-  `src/demo` 中有demo代码,可以参考
    - mnist 为 mnist 数据集,使用代码demo默认配置下识别率约为91%

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

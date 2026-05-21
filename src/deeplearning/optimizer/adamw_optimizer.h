#pragma once

#include "adam_optimizer.h"

namespace deeplearning {

// AdamW: Adam with Decoupled Weight Decay
// 参考: Loshchilov & Hutter 2017, https://arxiv.org/abs/1711.05101
//
// 与经典 Adam 的区别: weight decay 不进入 m/v 的累加 (`g += wd*w` 被移除),
// 而是直接在参数更新上加一项 `lr * wd * w`:
//
//   update = lr * (m_hat / (sqrt(v_hat) + eps)) + lr * wd * w
//
// 在 Transformer / BERT / 现代 CV 训练里通常优于经典 Adam.
class AdamWOptimizer : public AdamOptimizer {
public:
  AdamWOptimizer(const std::vector<int> &layer, double beta1 = 0.9,
                 double beta2 = 0.999, double epsilon = 1e-8);

  double CalcChangeValue(double delta, double learning_rate,
                         const std::pair<int, int> &pos,
                         int weight_pos = -1,
                         double param_value = 0.0) override;

  OptimizerType GetOptimizerType() override;
};

} // namespace deeplearning

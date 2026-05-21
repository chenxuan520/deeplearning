#pragma once

#include "optimizer_base.h"
#include <vector>

namespace deeplearning {

// RMSProp: Hinton 在 Coursera 课上提出的自适应优化器
//
//   v_t = decay * v_{t-1} + (1 - decay) * g_t^2
//   update = lr * g_t / (sqrt(v_t) + eps)
//
// 比 Adam 简单 (没有一阶矩 / bias correction), 在 RNN 上历史口碑较好.
class RMSPropOptimizer : public OptimizerFunction {
public:
  RMSPropOptimizer(const std::vector<int> &layer, double decay = 0.9,
                   double epsilon = 1e-8);

  double CalcChangeValue(double delta, double learning_rate,
                         const std::pair<int, int> &pos,
                         int weight_pos = -1,
                         double param_value = 0.0) override;

  OptimizerType GetOptimizerType() override;

  void set_decay(double v) { decay_ = v; }
  void set_epsilon(double v) { epsilon_ = v; }
  double decay() const { return decay_; }
  double epsilon() const { return epsilon_; }

private:
  std::vector<std::vector<std::vector<double>>> weight_v_;
  std::vector<std::vector<double>> bias_v_;
  double decay_;
  double epsilon_;
};

} // namespace deeplearning

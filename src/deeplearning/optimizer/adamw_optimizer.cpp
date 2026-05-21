#include "adamw_optimizer.h"

#include <cmath>

namespace deeplearning {

AdamWOptimizer::AdamWOptimizer(const std::vector<int> &layer, double beta1,
                               double beta2, double epsilon)
    : AdamOptimizer(layer, beta1, beta2, epsilon) {}

double AdamWOptimizer::CalcChangeValue(double delta, double learning_rate,
                                       const std::pair<int, int> &pos,
                                       int weight_pos, double param_value) {
  auto [x, y] = pos;

  // AdamW: 梯度本身不加 wd, 直接更新自适应项即可.
  double g = delta;

  double m, v;
  if (weight_pos == -1) {
    bias_m_[x][y] = beta1_ * bias_m_[x][y] + (1.0 - beta1_) * g;
    bias_v_[x][y] = beta2_ * bias_v_[x][y] + (1.0 - beta2_) * g * g;
    m = bias_m_[x][y];
    v = bias_v_[x][y];
  } else {
    auto &mref = weight_m_[x][y][weight_pos];
    auto &vref = weight_v_[x][y][weight_pos];
    mref = beta1_ * mref + (1.0 - beta1_) * g;
    vref = beta2_ * vref + (1.0 - beta2_) * g * g;
    m = mref;
    v = vref;
  }

  double m_hat = m / bias_correction1_;
  double v_hat = v / bias_correction2_;
  double adaptive = learning_rate * m_hat / (std::sqrt(v_hat) + epsilon_);

  // 解耦的 weight decay: 仅作用于 weight, 不作用于 bias.
  if (weight_pos != -1 && weight_decay_ != 0.0) {
    adaptive += learning_rate * weight_decay_ * param_value;
  }
  return adaptive;
}

OptimizerType AdamWOptimizer::GetOptimizerType() { return OPTIMIZER_ADAMW; }

} // namespace deeplearning

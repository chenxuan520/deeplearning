#include "adam_optimizer.h"

#include <cmath>

namespace deeplearning {

AdamOptimizer::AdamOptimizer(const std::vector<int> &layer, double beta1,
                             double beta2, double epsilon)
    : OptimizerFunction(layer), beta1_(beta1), beta2_(beta2),
      epsilon_(epsilon) {
  int L = (int)layer.size();
  bias_m_.assign(L, {});
  bias_v_.assign(L, {});
  weight_m_.assign(L, {});
  weight_v_.assign(L, {});

  for (int i = 0; i < L; i++) {
    bias_m_[i].assign(layer[i], 0.0);
    bias_v_[i].assign(layer[i], 0.0);
    if (i != 0) {
      weight_m_[i].assign(layer[i], std::vector<double>(layer[i - 1], 0.0));
      weight_v_[i].assign(layer[i], std::vector<double>(layer[i - 1], 0.0));
    }
  }
}

void AdamOptimizer::BeforeStep() {
  step_ += 1;
  bias_correction1_ = 1.0 - std::pow(beta1_, step_);
  bias_correction2_ = 1.0 - std::pow(beta2_, step_);
}

double AdamOptimizer::CalcChangeValue(double delta, double learning_rate,
                                      const std::pair<int, int> &pos,
                                      int weight_pos, double param_value) {
  auto [x, y] = pos;

  // 经典 Adam: L2 通过 `g += wd * w` 加到梯度里 (只对 weight)
  double g = delta;
  if (weight_pos != -1 && weight_decay_ != 0.0) {
    g += weight_decay_ * param_value;
  }

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
  return learning_rate * m_hat / (std::sqrt(v_hat) + epsilon_);
}

OptimizerType AdamOptimizer::GetOptimizerType() { return OPTIMIZER_ADAM; }

} // namespace deeplearning

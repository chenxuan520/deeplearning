#include "rmsprop_optimizer.h"

#include <cmath>

namespace deeplearning {

RMSPropOptimizer::RMSPropOptimizer(const std::vector<int> &layer, double decay,
                                   double epsilon)
    : OptimizerFunction(layer), decay_(decay), epsilon_(epsilon) {
  ResetState(layer);
}

bool RMSPropOptimizer::ResetState(const std::vector<int> &layer) {
  layer_ = layer;
  int L = (int)layer.size();
  bias_v_.assign(L, {});
  weight_v_.assign(L, {});
  for (int i = 0; i < L; i++) {
    bias_v_[i].assign(layer[i], 0.0);
    if (i != 0) {
      weight_v_[i].assign(layer[i], std::vector<double>(layer[i - 1], 0.0));
    }
  }
  return true;
}

double RMSPropOptimizer::CalcChangeValue(double delta, double learning_rate,
                                         const std::pair<int, int> &pos,
                                         int weight_pos,
                                         double param_value) {
  auto [x, y] = pos;
  // 经典 L2: g <- g + wd * w (只作用于 weight)
  double g = delta;
  if (weight_pos != -1 && weight_decay_ != 0.0) {
    g += weight_decay_ * param_value;
  }

  double v;
  if (weight_pos == -1) {
    bias_v_[x][y] = decay_ * bias_v_[x][y] + (1.0 - decay_) * g * g;
    v = bias_v_[x][y];
  } else {
    auto &vref = weight_v_[x][y][weight_pos];
    vref = decay_ * vref + (1.0 - decay_) * g * g;
    v = vref;
  }
  return learning_rate * g / (std::sqrt(v) + epsilon_);
}

OptimizerType RMSPropOptimizer::GetOptimizerType() { return OPTIMIZER_RMSPROP; }

} // namespace deeplearning

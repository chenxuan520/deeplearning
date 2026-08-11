#include "momentum_optimizer.h"

namespace deeplearning {

MomentumOptimizer::MomentumOptimizer(const std::vector<int> &layer)
    : OptimizerFunction(layer) {
  ResetState(layer);
}

bool MomentumOptimizer::ResetState(const std::vector<int> &layer) {
  layer_ = layer;
  bias_velocity_.assign(layer.size(), {});
  weight_velocity_.assign(layer.size(), {});

  for (int i = 0; i < (int)layer.size(); i++) {
    bias_velocity_[i].assign(layer[i], 0);
    if (i != 0) {
      weight_velocity_[i].assign(layer[i], {});
      for (int j = 0; j < layer[i]; j++) {
        weight_velocity_[i][j].assign(layer[i - 1], 0);
      }
    }
  }
  return true;
}

double MomentumOptimizer::CalcChangeValue(double delta, double learning_rate,
                                          const std::pair<int, int> &pos,
                                          int weight_pos,
                                          double param_value) {
  auto [x, y] = pos;
  // 经典 L2 正则: g <- g + wd * w (只作用于 weight)
  double g = delta;
  if (weight_pos != -1 && weight_decay_ != 0.0) {
    g += weight_decay_ * param_value;
  }
  if (weight_pos == -1) {
    bias_velocity_[x][y] =
        momentum_ * bias_velocity_[x][y] - learning_rate * g;
    return -bias_velocity_[x][y];
  }

  weight_velocity_[x][y][weight_pos] =
      momentum_ * weight_velocity_[x][y][weight_pos] - learning_rate * g;
  return -weight_velocity_[x][y][weight_pos];
}

OptimizerType MomentumOptimizer::GetOptimizerType() {
  return OPTIMIZER_MOMENTUM;
}

} // namespace deeplearning

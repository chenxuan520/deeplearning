#include "momentum_optimizer.h"

namespace deeplearning {

MomentumOptimizer::MomentumOptimizer(const std::vector<int> &layer)
    : OptimizerFunction(layer) {
  bias_velocity_.resize(layer.size());
  weight_velocity_.resize(layer.size());

  for (int i = 0; i < layer.size(); i++) {
    bias_velocity_[i].resize(layer[i], 0);
    if (i != 0) {
      weight_velocity_[i].resize(layer[i]);
      for (int j = 0; j < layer[i]; j++) {
        weight_velocity_[i][j].resize(layer[i - 1], 0);
      }
    }
  }
}

double MomentumOptimizer::CalcChangeValue(double delta, double learning_rate,
                                          const std::pair<int, int> &pos,
                                          int weight_pos) {
  auto [x, y] = pos;
  if (weight_pos == -1) {
    bias_velocity_[x][y] = momentum_ * bias_velocity_[x][y] - learning_rate * delta;
    return -bias_velocity_[x][y];
  }

  weight_velocity_[x][y][weight_pos] =
      momentum_ * weight_velocity_[x][y][weight_pos] - learning_rate * delta;
  return -weight_velocity_[x][y][weight_pos];
}

OptimizerType MomentumOptimizer::GetOptimizerType() {
  return OPTIMIZER_MOMENTUM;
}

} // namespace deeplearning

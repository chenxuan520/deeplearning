#pragma once

#include "optimizer_base.h"
#include <vector>

namespace deeplearning {

class MomentumOptimizer : public OptimizerFunction {
public:
  MomentumOptimizer(const std::vector<int> &layer) : OptimizerFunction(layer) {
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

  double CalcChangeValue(double delta, double learning_rate,
                         const std::pair<int, int> &pos,
                         int weight_pos = -1) override {
    auto [x, y] = pos;
    if (weight_pos == -1) {
      // calc bias
      bias_velocity_[x][y] =
          momentum * bias_velocity_[x][y] - learning_rate * delta;
      return -bias_velocity_[x][y];
    } else {
      // calc weight
      weight_velocity_[x][y][weight_pos] =
          momentum * weight_velocity_[x][y][weight_pos] - learning_rate * delta;
      return -weight_velocity_[x][y][weight_pos];
    }
    return 0;
  }

  OptimizerType GetOptimizerType() override { return OPTIMIZER_MOMENTUM; }

private:
  std::vector<std::vector<std::vector<double>>> weight_velocity_;
  std::vector<std::vector<double>> bias_velocity_;
  double momentum = 0.9;
};

} // namespace deeplearning

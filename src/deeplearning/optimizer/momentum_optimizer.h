#pragma once

#include "optimizer_base.h"
#include <vector>

namespace deeplearning {

class MomentumOptimizer : public OptimizerFunction {
public:
  MomentumOptimizer(const std::vector<int> &layer);

  double CalcChangeValue(double delta, double learning_rate,
                         const std::pair<int, int> &pos,
                         int weight_pos = -1,
                         double param_value = 0.0) override;

  OptimizerType GetOptimizerType() override;

  void set_momentum(double m) { momentum_ = m; }
  double momentum() const { return momentum_; }

private:
  std::vector<std::vector<std::vector<double>>> weight_velocity_;
  std::vector<std::vector<double>> bias_velocity_;
  double momentum_ = 0.9;
};

} // namespace deeplearning

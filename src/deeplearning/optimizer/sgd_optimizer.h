#pragma once

#include "optimizer_base.h"

namespace deeplearning {

class SGDOptimizer : public OptimizerFunction {
public:
  SGDOptimizer(const std::vector<int> &layer);
  double CalcChangeValue(double delta, double learning_rate,
                         const std::pair<int, int> &pos,
                          int weight_pos = -1) override;
  OptimizerType GetOptimizerType() override;
};

} // namespace deeplearning

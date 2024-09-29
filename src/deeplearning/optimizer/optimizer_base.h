#pragma once
#include <utility>
#include <vector>

namespace deeplearning {

enum OptimizerType {
  OPTIMIZER_SGD,
  OPTIMIZER_MOMENTUM,
  OPTIMIZER_ADAM,
};

class OptimizerFunction {
public:
  OptimizerFunction(const std::vector<int> &layer) : layer_(layer) {}
  virtual double CalcChangeValue(double delta, double learning_rate,
                                 const std::pair<int, int> &pos,
                                 int weight_pos = -1) = 0;
  virtual OptimizerType GetOptimizerType() = 0;

protected:
  std::vector<int> layer_;
};

} // namespace deeplearning

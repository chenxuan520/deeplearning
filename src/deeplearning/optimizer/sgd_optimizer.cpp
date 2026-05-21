#include "sgd_optimizer.h"

namespace deeplearning {

SGDOptimizer::SGDOptimizer(const std::vector<int> &layer)
    : OptimizerFunction(layer) {}

double SGDOptimizer::CalcChangeValue(double delta, double learning_rate,
                                     const std::pair<int, int> &, int weight_pos,
                                     double param_value) {
  // 经典 L2 正则: g <- g + wd * w (只作用于 weight, bias 不衰减)
  double g = delta;
  if (weight_pos != -1 && weight_decay_ != 0.0) {
    g += weight_decay_ * param_value;
  }
  return learning_rate * g;
}

OptimizerType SGDOptimizer::GetOptimizerType() { return OPTIMIZER_SGD; }

} // namespace deeplearning

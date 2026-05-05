#include "sgd_optimizer.h"

namespace deeplearning {

SGDOptimizer::SGDOptimizer(const std::vector<int> &layer)
    : OptimizerFunction(layer) {}

double SGDOptimizer::CalcChangeValue(double delta, double learning_rate,
                                     const std::pair<int, int> &, int) {
  return learning_rate * delta;
}

OptimizerType SGDOptimizer::GetOptimizerType() { return OPTIMIZER_SGD; }

} // namespace deeplearning

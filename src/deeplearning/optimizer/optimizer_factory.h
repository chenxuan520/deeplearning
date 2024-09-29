#pragma once

#include "optimizer/momentum_optimizer.h"
#include "optimizer/optimizer_base.h"
#include "optimizer/sgd_optimizer.h"
#include <vector>
namespace deeplearning {

class OptimizerFactory {
public:
  static std::shared_ptr<OptimizerFunction>
  Create(OptimizerType optimizer_type, const std::vector<int> &layer) {
    switch (optimizer_type) {
    case OPTIMIZER_SGD:
      return std::make_shared<SGDOptimizer>(layer);
    case OPTIMIZER_MOMENTUM:
      return std::make_shared<MomentumOptimizer>(layer);
    default:
      return nullptr;
    }
    return nullptr;
  }
};

} // namespace deeplearning

#include "optimizer_factory.h"
#include "momentum_optimizer.h"
#include "sgd_optimizer.h"

namespace deeplearning {

std::shared_ptr<OptimizerFunction>
OptimizerFactory::Create(OptimizerType optimizer_type,
                         const std::vector<int> &layer) {
  switch (optimizer_type) {
  case OPTIMIZER_SGD:
    return std::make_shared<SGDOptimizer>(layer);
  case OPTIMIZER_MOMENTUM:
    return std::make_shared<MomentumOptimizer>(layer);
  default:
    return nullptr;
  }
}

} // namespace deeplearning

#include "optimizer_factory.h"
#include "adam_optimizer.h"
#include "adamw_optimizer.h"
#include "momentum_optimizer.h"
#include "rmsprop_optimizer.h"
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
  case OPTIMIZER_ADAM:
    return std::make_shared<AdamOptimizer>(layer);
  case OPTIMIZER_RMSPROP:
    return std::make_shared<RMSPropOptimizer>(layer);
  case OPTIMIZER_ADAMW:
    return std::make_shared<AdamWOptimizer>(layer);
  default:
    return nullptr;
  }
}

} // namespace deeplearning

#include "loss_factory.h"
#include "cross_entropy_loss.h"
#include "mse_loss.h"

namespace deeplearning {

std::shared_ptr<LossFunction> LossFactory::Create(LossType loss_type) {
  switch (loss_type) {
  case LOSS_MSE:
    return std::make_shared<MSELoss>();
  case LOSS_CROSS_ENTROPY:
    return std::make_shared<CrossEntropyLoss>();
  default:
    return nullptr;
  }
}

} // namespace deeplearning

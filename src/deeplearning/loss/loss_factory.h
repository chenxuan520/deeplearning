#pragma once
#include "cross_entropy_loss.h"
#include "loss_base.h"
#include "mse_loss.h"
#include <memory>
namespace deeplearning {

class LossFactory {
public:
  static std::shared_ptr<LossFunction> Create(LossType loss_type) {
    switch (loss_type) {
    case LOSS_MSE:
      return std::make_shared<MSELoss>();
    case LOSS_CROSS_ENTROPY:
      return std::make_shared<CrossEntropyLoss>();
    default:
      return nullptr;
    }
    return nullptr;
  }
};
} // namespace deeplearning

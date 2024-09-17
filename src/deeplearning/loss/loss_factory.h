#pragma once
#include "loss_base.h"
#include "mse_loss.h"
#include <memory>
#include <vector>
namespace deeplearning {

enum LossType {
  LOSS_MSE,
};
class LossFactory {
public:
  static std::shared_ptr<LossFunction> Create(LossType loss_type) {
    switch (loss_type) {
    case LOSS_MSE:
      return std::make_shared<MSELoss>();
    default:
      return nullptr;
    }
    return nullptr;
  }
};
} // namespace deeplearning

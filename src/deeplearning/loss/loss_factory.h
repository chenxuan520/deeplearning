#pragma once
#include "loss_base.h"
#include <memory>
namespace deeplearning {

class LossFactory {
public:
  static std::shared_ptr<LossFunction> Create(LossType loss_type);
};
} // namespace deeplearning

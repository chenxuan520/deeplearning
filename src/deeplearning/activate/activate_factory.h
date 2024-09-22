#pragma once

#include "activate_base.h"
#include "relu_activate.h"
#include "sigmoid_activate.h"
#include "tanh_activate.h"
#include <memory>

namespace deeplearning {

class ActivateFactory {
public:
  static std::shared_ptr<ActivateFunction> Create(ActivateType activate_type) {
    switch (activate_type) {
    case ACTIVATE_SIGMOID:
      return std::make_shared<SigmoidActivate>();
    case ACTIVATE_RELU:
      return std::make_shared<ReluActivate>();
    case ACTIVATE_TANH:
      return std::make_shared<TanhActivate>();
    default:
      return nullptr;
    }
    return nullptr;
  }
};

} // namespace deeplearning

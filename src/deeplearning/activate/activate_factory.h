#pragma once

#include "activate_base.h"
#include "sigmoid_activate.h"
#include <memory>

namespace deeplearning {

class ActivateFactory {
public:
  static std::shared_ptr<ActivateFunction> Create(ActivateType activate_type) {
    switch (activate_type) {
    case ACTIVATE_SIGMOID:
      return std::make_shared<SigmoidActivate>();
    default:
      return nullptr;
    }
    return nullptr;
  }
};

} // namespace deeplearning

#pragma once

#include "param_init_base.h"
#include "zero_param_init.h"

namespace deeplearning {

class ParamInitFactory {
public:
  static std::shared_ptr<ParamInitFunction>
  Create(ParamInitType param_init_type) {
    switch (param_init_type) {
    case PARAM_INIT_ZERO:
      return std::make_shared<ZeroParamInitFunction>();
    default:
      return nullptr;
    }
    return nullptr;
  }
};

} // namespace deeplearning

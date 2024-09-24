#pragma once

#include "he_param_init.h"
#include "normal_random_param_init.h"
#include "param_init_base.h"
#include "uniform_random_param_init.h"
#include "xavier_param_init.h"
#include "zero_param_init.h"

namespace deeplearning {

class ParamInitFactory {
public:
  static std::shared_ptr<ParamInitFunction>
  Create(ParamInitType param_init_type) {
    switch (param_init_type) {
    case PARAM_INIT_ZERO:
      return std::make_shared<ZeroParamInitFunction>();
    case PARAM_INIT_UNIFORM_RANDOM:
      return std::make_shared<UniformRandomParamInitFunction>();
    case PARAM_INIT_NORMAL_RANDOM:
      return std::make_shared<NormalRandomParamInitFunction>();
    case PARAM_INIT_XAVIER:
      return std::make_shared<XavierParamInitFunction>();
    case PARAM_INIT_HE:
      return std::make_shared<HeParamInitFunction>();
    default:
      return nullptr;
    }
    return nullptr;
  }
};

} // namespace deeplearning

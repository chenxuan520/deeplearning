#pragma once

#include "param_init_base.h"
#include <memory>

namespace deeplearning {

class ParamInitFactory {
public:
  static std::shared_ptr<ParamInitFunction>
  Create(ParamInitType param_init_type);
};

} // namespace deeplearning

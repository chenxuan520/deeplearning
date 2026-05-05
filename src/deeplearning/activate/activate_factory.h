#pragma once

#include "activate_base.h"
#include <memory>

namespace deeplearning {

class ActivateFactory {
public:
  static std::shared_ptr<ActivateFunction> Create(ActivateType activate_type);
};

} // namespace deeplearning

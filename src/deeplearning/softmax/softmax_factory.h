#pragma once

#include "softmax_base.h"
#include <memory>

namespace deeplearning {

class SoftmaxFactory {
public:
  static std::shared_ptr<SoftmaxFunction> Create(SoftmaxType softmax_type);
};
} // namespace deeplearning

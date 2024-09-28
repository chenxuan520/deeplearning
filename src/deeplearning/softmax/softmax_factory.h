#pragma once

#include "softmax/none_softmax.h"
#include "softmax/std_softmax.h"
#include "softmax_base.h"
#include <memory>

namespace deeplearning {

class SoftmaxFactory {
public:
  static std::shared_ptr<SoftmaxFunction> Create(SoftmaxType softmax_type) {
    switch (softmax_type) {
    case SOFTMAX_NONE:
      return std::make_shared<NoneSoftmax>();
    case SOFTMAX_STD:
      return std::make_shared<StdSoftmax>();
    default:
      return nullptr;
    }
    return nullptr;
  }
};
} // namespace deeplearning

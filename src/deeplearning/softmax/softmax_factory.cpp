#include "softmax_factory.h"
#include "none_softmax.h"
#include "std_softmax.h"

namespace deeplearning {

std::shared_ptr<SoftmaxFunction> SoftmaxFactory::Create(SoftmaxType softmax_type) {
  switch (softmax_type) {
  case SOFTMAX_NONE:
    return std::make_shared<NoneSoftmax>();
  case SOFTMAX_STD:
    return std::make_shared<StdSoftmax>();
  default:
    return nullptr;
  }
}

} // namespace deeplearning

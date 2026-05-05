#include "activate_factory.h"
#include "relu_activate.h"
#include "sigmoid_activate.h"
#include "tanh_activate.h"

namespace deeplearning {

std::shared_ptr<ActivateFunction>
ActivateFactory::Create(ActivateType activate_type) {
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
}

} // namespace deeplearning

#include "activate_factory.h"
#include "gelu_activate.h"
#include "leaky_relu_activate.h"
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
  case ACTIVATE_LEAKY_RELU:
    return std::make_shared<LeakyReluActivate>();
  case ACTIVATE_GELU:
    return std::make_shared<GeluActivate>();
  default:
    return nullptr;
  }
}

} // namespace deeplearning

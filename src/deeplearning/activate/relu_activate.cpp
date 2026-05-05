#include "relu_activate.h"

namespace deeplearning {

double ReluActivate::Activate(const double &x) { return x > 0 ? x : 0; }

double ReluActivate::DerivActivate(const double &output) {
  return output > 0 ? 1 : 0;
}

ActivateType ReluActivate::GetActivateType() {
  return ActivateType::ACTIVATE_RELU;
}

} // namespace deeplearning

#include "sigmoid_activate.h"

namespace deeplearning {

double SigmoidActivate::Activate(const double &input) {
  return 1 / (1 + exp(-input));
}

double SigmoidActivate::DerivActivate(const double &output) {
  return output * (1 - output);
}

ActivateType SigmoidActivate::GetActivateType() { return ACTIVATE_SIGMOID; }

} // namespace deeplearning

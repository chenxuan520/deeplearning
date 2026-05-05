#include "tanh_activate.h"

namespace deeplearning {

double TanhActivate::Activate(const double &input) {
  return (1 - exp(-2 * input)) / (1 + exp(-2 * input));
}

double TanhActivate::DerivActivate(const double &output) {
  return 1 - output * output;
}

ActivateType TanhActivate::GetActivateType() { return ACTIVATE_TANH; }

} // namespace deeplearning

#include "none_softmax.h"

namespace deeplearning {

void NoneSoftmax::Normalize(const std::vector<double> &, std::vector<double> &) {
  return;
}

double NoneSoftmax::CalcDelta(double, double, std::shared_ptr<LossFunction>) {
  return 0;
}

SoftmaxType NoneSoftmax::GetSoftmaxType() { return SOFTMAX_NONE; }

} // namespace deeplearning

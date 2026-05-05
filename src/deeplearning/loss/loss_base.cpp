#include "loss_base.h"

namespace deeplearning {

double LossFunction::AverageLoss(const std::vector<double> &target,
                                 const std::vector<double> &output) {
  double result = 0;
  if (target.size() != output.size() || target.size() == 0) {
    return -1;
  }
  for (int i = 0; i < target.size(); i++) {
    result += Loss(target[i], output[i]);
  }
  result /= target.size();
  return result;
}

} // namespace deeplearning

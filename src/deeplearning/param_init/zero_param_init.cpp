#include "zero_param_init.h"
#include <algorithm>

namespace deeplearning {

void ZeroParamInitFunction::InitParam(
    std::vector<std::vector<std::vector<double>>> &weight,
    std::vector<std::vector<double>> &bias) {
  for (auto &w : weight) {
    for (auto &w_ : w) {
      std::fill(w_.begin(), w_.end(), 0.0);
    }
  }
  for (auto &b : bias) {
    std::fill(b.begin(), b.end(), 0.0);
  }
}

ParamInitType ZeroParamInitFunction::GetParamInitType() {
  return ParamInitType::PARAM_INIT_ZERO;
}

} // namespace deeplearning

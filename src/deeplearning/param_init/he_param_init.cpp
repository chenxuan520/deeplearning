#include "he_param_init.h"

namespace deeplearning {

void HeParamInitFunction::InitParam(
    std::vector<std::vector<std::vector<double>>> &weight,
    std::vector<std::vector<double>> &bias) {
  if (weight.size() != bias.size()) {
    return;
  }
  for (int i = 1; i < weight.size(); i++) {
    double limit = std::sqrt(6.0 / weight[i - 1].size());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-limit, limit);

    for (auto &w : weight[i]) {
      for (auto &w_ : w) {
        w_ = dis(gen);
      }
    }
    for (auto &b : bias[i]) {
      b = dis(gen);
    }
  }
}

ParamInitType HeParamInitFunction::GetParamInitType() {
  return ParamInitType::PARAM_INIT_HE;
}

} // namespace deeplearning

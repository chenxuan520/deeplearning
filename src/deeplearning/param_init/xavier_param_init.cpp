#include "xavier_param_init.h"

namespace deeplearning {

void XavierParamInitFunction::InitParam(
    std::vector<std::vector<std::vector<double>>> &weight,
    std::vector<std::vector<double>> &bias) {
  if (weight.size() != bias.size()) {
    return;
  }

  for (int i = 1; i < weight.size(); i++) {
    // Same bug as he_param_init: `weight[i - 1].size()` is 0 for i==1 (the
    // input layer has no weights), so the divisor becomes weight[i].size()
    // alone -- meaning the "fan-in" half of Xavier is missing. Fix by
    // using the actual row length of weight[i][0] (which equals layer[i-1]).
    size_t fan_in = 0;
    size_t fan_out = weight[i].size();
    if (!weight[i].empty()) {
      fan_in = weight[i][0].size();
    } else if (!weight[i - 1].empty()) {
      fan_in = weight[i - 1].size();
    }
    if (fan_in + fan_out == 0) {
      continue;
    }
    double limit = std::sqrt(6.0 / (double)(fan_in + fan_out));
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

ParamInitType XavierParamInitFunction::GetParamInitType() {
  return ParamInitType::PARAM_INIT_XAVIER;
}

} // namespace deeplearning

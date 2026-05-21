#include "he_param_init.h"

namespace deeplearning {

void HeParamInitFunction::InitParam(
    std::vector<std::vector<std::vector<double>>> &weight,
    std::vector<std::vector<double>> &bias) {
  if (weight.size() != bias.size()) {
    return;
  }
  for (int i = 1; i < weight.size(); i++) {
    // BUGFIX (was): `sqrt(6.0 / weight[i - 1].size())` which uses the
    // neuron count of layer i-1. For i==1, weight[0] is empty (the input
    // layer has no weights), so size()==0 and the formula yields sqrt(6/0)
    // = inf, which propagates as NaN through the uniform distribution and
    // poisons every layer-1 weight. The intended fan-in is the input
    // dimension of THIS layer, which is the row length of weight[i][0]
    // (== layer[i-1]). Falling back to weight[i-1].size() preserves
    // backwards-compatible behavior for layers 2+ where the bug was
    // accidentally invisible.
    size_t fan_in = 0;
    if (!weight[i].empty()) {
      fan_in = weight[i][0].size();
    } else if (!weight[i - 1].empty()) {
      fan_in = weight[i - 1].size();
    }
    if (fan_in == 0) {
      continue;
    }
    double limit = std::sqrt(6.0 / (double)fan_in);
    std::mt19937 gen;
    if (seed_ < 0) {
      std::random_device rd;
      gen.seed(rd());
    } else {
      gen.seed(static_cast<uint32_t>(seed_) + static_cast<uint32_t>(i) * 2017u);
    }
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

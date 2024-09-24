#pragma once

#include "param_init_base.h"
#include <random>

namespace deeplearning {

class XavierParamInitFunction : public ParamInitFunction {
public:
  void InitParam(std::vector<std::vector<std::vector<double>>> &weight,
                 std::vector<std::vector<double>> &bias) override {
    if (weight.size() != bias.size()) {
      return;
    }

    for (int i = 1; i < weight.size(); i++) {
      double limit = std::sqrt(6.0 / (weight[i - 1].size() + weight[i].size()));
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

  ParamInitType GetParamInitType() override {
    return ParamInitType::PARAM_INIT_XAVIER;
  }
};

} // namespace deeplearning

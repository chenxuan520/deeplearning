#pragma once

#include "param_init_base.h"
#include <random>

namespace deeplearning {

class UniformRandomParamInitFunction : public ParamInitFunction {
public:
  UniformRandomParamInitFunction() : min_(-0.5), max_(0.5) {};
  UniformRandomParamInitFunction(double min, double max)
      : min_(min), max_(max) {}

  void InitParam(std::vector<std::vector<std::vector<double>>> &weight,
                 std::vector<std::vector<double>> &bias) override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distr(min_, max_);

    for (auto &w : weight) {
      for (auto &w_ : w) {
        for (auto &w__ : w_) {
          w__ = distr(gen);
        }
      }
    }
    for (auto &b : bias) {
      for (auto &b_ : b) { // 这里的b_是bias的元素，即bias中每个元素的元素
        b_ = distr(gen);
      }
    }
  }

  ParamInitType GetParamInitType() override {
    return ParamInitType::PARAM_INIT_UNIFORM_RANDOM;
  }

private:
  double min_;
  double max_;
};
} // namespace deeplearning

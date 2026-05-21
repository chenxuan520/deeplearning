#pragma once

#include <vector>
namespace deeplearning {

enum ParamInitType {
  PARAM_INIT_ZERO,
  PARAM_INIT_UNIFORM_RANDOM,
  PARAM_INIT_NORMAL_RANDOM,
  PARAM_INIT_XAVIER,
  PARAM_INIT_HE,
};

class ParamInitFunction {
public:
  virtual ~ParamInitFunction() = default;
  virtual void InitParam(std::vector<std::vector<std::vector<double>>> &weight,
                         std::vector<std::vector<double>> &bias) = 0;
  virtual ParamInitType GetParamInitType() = 0;

  // 显式给参数初始化器设种子, 让结果可复现. 设为 -1 表示用 random_device
  // (非确定性, 适合 debug/演示, 不适合 CI 跑回归).
  void set_seed(int seed) { seed_ = seed; }
  int seed() const { return seed_; }

protected:
  int seed_ = -1;
};

} // namespace deeplearning

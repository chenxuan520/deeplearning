#pragma once

namespace deeplearning {

enum ActivateType {
  ACTIVATE_SIGMOID,
  ACTIVATE_RELU,
  ACTIVATE_TANH,
  ACTIVATE_LEAKY_RELU,
  ACTIVATE_GELU,
};

class ActivateFunction {
public:
  virtual ~ActivateFunction() = default;

  virtual double Activate(const double &input) = 0;

  // 兼容旧接口: 用 output 计算导数, 适合 sigmoid/tanh/relu.
  virtual double DerivActivate(const double &output) = 0;

  // 新接口: 同时拿到 input (pre-activation) 和 output.
  // 默认回退到旧 1-arg 版本, 老激活无需改动.
  // GELU 等需要 input 的激活在子类里重载此方法.
  virtual double DerivActivate(const double &input, const double &output) {
    (void)input;
    return DerivActivate(output);
  }

  virtual ActivateType GetActivateType() = 0;
};

} // namespace deeplearning

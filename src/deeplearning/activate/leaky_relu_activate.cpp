#include "leaky_relu_activate.h"

namespace deeplearning {

double LeakyReluActivate::Activate(const double &x) {
  return x > 0 ? x : slope_ * x;
}

double LeakyReluActivate::DerivActivate(const double &output) {
  // 这里复用 ReLU 的 "用 output 判断符号" 的约定:
  //   output > 0  -> 输入也 > 0  -> 梯度 = 1
  //   output <= 0 -> 输入 <= 0   -> 梯度 = slope
  // (output == 0 时按 slope 处理, 与上游 ReLU 一致)
  return output > 0 ? 1.0 : slope_;
}

ActivateType LeakyReluActivate::GetActivateType() {
  return ActivateType::ACTIVATE_LEAKY_RELU;
}

} // namespace deeplearning

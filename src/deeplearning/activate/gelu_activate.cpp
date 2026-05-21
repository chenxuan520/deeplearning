#include "gelu_activate.h"

#include <cmath>

namespace deeplearning {

namespace {
constexpr double kInvSqrt2 = 0.7071067811865476;   // 1 / sqrt(2)
constexpr double kInvSqrt2Pi = 0.3989422804014327; // 1 / sqrt(2π)
} // namespace

double GeluActivate::Activate(const double &input) {
  // 精确 GELU = x * 0.5 * (1 + erf(x / sqrt(2)))
  return 0.5 * input * (1.0 + std::erf(input * kInvSqrt2));
}

double GeluActivate::DerivActivate(const double &output) {
  // 仅从 output 反推导数很难 (没有闭式逆), 这里粗略近似:
  // 假设 output ≈ x 时 (x 较大), 导数 ≈ 1; output ≈ 0 时, 导数 ≈ 0.5.
  // 真正训练时建议用 2-arg 版本.
  if (output > 1.0) {
    return 1.0;
  }
  if (output < -0.17) { // GELU 的最小值约 -0.17
    return 0.0;
  }
  return 0.5 * (output + 1.0); // 粗略线性插值
}

double GeluActivate::DerivActivate(const double &input, const double &output) {
  (void)output;
  // GELU'(x) = Φ(x) + x * φ(x)
  // 其中 Φ(x) = 0.5 * (1 + erf(x / sqrt(2)))
  //      φ(x) = exp(-x^2 / 2) / sqrt(2π)
  double cdf = 0.5 * (1.0 + std::erf(input * kInvSqrt2));
  double pdf = std::exp(-0.5 * input * input) * kInvSqrt2Pi;
  return cdf + input * pdf;
}

ActivateType GeluActivate::GetActivateType() { return ACTIVATE_GELU; }

} // namespace deeplearning

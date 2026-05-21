#pragma once

#include "activate_base.h"

namespace deeplearning {

// GELU: Gaussian Error Linear Unit (Hendrycks & Gimpel, 2016)
// 参考: https://arxiv.org/abs/1606.08415
//
//   GELU(x) = x * Φ(x)        (Φ 是标准正态 CDF)
//   GELU'(x) = Φ(x) + x * φ(x)  (φ 是标准正态 PDF)
//
// 实践中常用 tanh 近似:
//   GELU(x) ≈ 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x^3) ))
//
// 这里我们用 erf 的精确版本 (std::erf 是 C++11 标准库的一部分).
// GELU 在 Transformer / BERT / GPT 类模型里非常常见, 通常优于 ReLU.
class GeluActivate : public ActivateFunction {
public:
  double Activate(const double &input) override;
  double DerivActivate(const double &output) override;
  double DerivActivate(const double &input, const double &output) override;
  ActivateType GetActivateType() override;
};

} // namespace deeplearning

#pragma once

#include "activate_base.h"

namespace deeplearning {

// LeakyReLU: f(x) = x if x > 0 else slope * x
// 比 ReLU 改善 "死亡 ReLU" (dead neuron) 问题, 负半轴有非零梯度.
// 默认 slope = 0.01.
class LeakyReluActivate : public ActivateFunction {
public:
  LeakyReluActivate() = default;
  explicit LeakyReluActivate(double slope) : slope_(slope) {}

  double Activate(const double &x) override;
  double DerivActivate(const double &output) override;
  ActivateType GetActivateType() override;

  void set_slope(double s) { slope_ = s; }
  double slope() const { return slope_; }

private:
  double slope_ = 0.01;
};

} // namespace deeplearning

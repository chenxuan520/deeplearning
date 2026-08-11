#pragma once

#include "optimizer_base.h"
#include <vector>

namespace deeplearning {

// Adam: Adaptive Moment Estimation
// 参考: Kingma & Ba 2014, https://arxiv.org/abs/1412.6980
//
//   m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
//   v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
//   m_hat = m_t / (1 - beta1^t)
//   v_hat = v_t / (1 - beta2^t)
//   update = lr * m_hat / (sqrt(v_hat) + eps)
//
// 权重衰减: 经典 Adam 把 L2 直接加到 grad (`g += wd * w`); 这种做法
// 在自适应步长下衰减会被分母削弱, 实践效果偏弱 (AdamW 是修正版本).
class AdamOptimizer : public OptimizerFunction {
public:
  AdamOptimizer(const std::vector<int> &layer, double beta1 = 0.9,
                double beta2 = 0.999, double epsilon = 1e-8);

  void BeforeStep() override;

  double CalcChangeValue(double delta, double learning_rate,
                         const std::pair<int, int> &pos,
                         int weight_pos = -1,
                         double param_value = 0.0) override;

  OptimizerType GetOptimizerType() override;

  bool ResetState(const std::vector<int> &layer) override;

  void set_beta1(double v) { beta1_ = v; }
  void set_beta2(double v) { beta2_ = v; }
  void set_epsilon(double v) { epsilon_ = v; }
  double beta1() const { return beta1_; }
  double beta2() const { return beta2_; }
  double epsilon() const { return epsilon_; }
  int step() const { return step_; }

protected:
  // 一阶矩 (mean): m
  std::vector<std::vector<std::vector<double>>> weight_m_;
  std::vector<std::vector<double>> bias_m_;
  // 二阶矩 (uncentered variance): v
  std::vector<std::vector<std::vector<double>>> weight_v_;
  std::vector<std::vector<double>> bias_v_;

  double beta1_;
  double beta2_;
  double epsilon_;
  // step counter, 每次 BeforeStep 加 1; 用于 bias correction.
  int step_ = 0;
  // 当前 step 的 bias correction 缓存, BeforeStep 计算一次, 复用整步.
  double bias_correction1_ = 1.0;
  double bias_correction2_ = 1.0;
};

} // namespace deeplearning

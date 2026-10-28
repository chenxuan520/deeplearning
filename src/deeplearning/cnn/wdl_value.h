#pragma once

#include <array>
#include <cmath>

namespace deeplearning {

class WdlValue {
public:
  static constexpr int kRawSize = 2;
  static constexpr int kClassNum = 3;
  static constexpr int kWin = 0;
  static constexpr int kDraw = 1;
  static constexpr int kLoss = 2;

  struct Output {
    std::array<float, kClassNum> probabilities_;
    float expected_value_ = 0.0f;
  };

  static Output Forward(float value_logit, float draw_logit) {
    const float value = std::tanh(value_logit);
    const float draw_propensity = Sigmoid(draw_logit);
    const float decisive_mass = 1.0f - std::fabs(value);
    const float draw = draw_propensity * decisive_mass;
    return {{
                0.5f * (1.0f - draw + value),
                draw,
                0.5f * (1.0f - draw - value),
            },
            value};
  }

  static std::array<float, kRawSize> Backward(
      float value_logit, float draw_logit,
      const std::array<float, kClassNum> &grad_probabilities) {
    const float value = std::tanh(value_logit);
    const float draw_propensity = Sigmoid(draw_logit);
    const float grad_value = 1.0f - value * value;
    const float grad_decisive_mass =
        value > 0.0f ? -grad_value : value < 0.0f ? grad_value : 0.0f;
    const float grad_draw_value =
        draw_propensity * grad_decisive_mass;
    const float grad_draw_propensity =
        (1.0f - std::fabs(value)) * draw_propensity *
        (1.0f - draw_propensity);
    const float grad_win_value =
        0.5f * (grad_value - grad_draw_value);
    const float grad_loss_value =
        0.5f * (-grad_value - grad_draw_value);
    const float grad_win_propensity = -0.5f * grad_draw_propensity;
    const float grad_loss_propensity = -0.5f * grad_draw_propensity;
    return {
        grad_probabilities[kWin] * grad_win_value +
            grad_probabilities[kDraw] * grad_draw_value +
            grad_probabilities[kLoss] * grad_loss_value,
        grad_probabilities[kWin] * grad_win_propensity +
            grad_probabilities[kDraw] * grad_draw_propensity +
            grad_probabilities[kLoss] * grad_loss_propensity,
    };
  }

private:
  static float Sigmoid(float value) {
    if (value >= 0.0f) {
      const float exponential = std::exp(-value);
      return 1.0f / (1.0f + exponential);
    }
    const float exponential = std::exp(value);
    return exponential / (1.0f + exponential);
  }
};

} // namespace deeplearning

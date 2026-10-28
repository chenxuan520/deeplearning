#include "loss/policy_value_loss.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace deeplearning {

PolicyValueLoss::RC PolicyValueLoss::Init(const Config &config) {
  if (is_init_) {
    err_msg_ = "[PolicyValueLoss::Init] already initialized";
    return ALREADY_INIT;
  }
  if (config.policy_size_ <= 0 || config.policy_weight_ < 0.0f ||
      config.value_weight_ < 0.0f ||
      (config.policy_weight_ == 0.0f && config.value_weight_ == 0.0f)) {
    err_msg_ = "[PolicyValueLoss::Init] invalid config";
    return INVALID_DATA;
  }
  config_ = config;
  is_init_ = true;
  return SUCCESS;
}

PolicyValueLoss::RC PolicyValueLoss::ForwardBackward(
    const std::vector<float> &policy_logits,
    const std::vector<float> &values,
    const std::vector<std::uint8_t> &legal_mask,
    const std::vector<float> &policy_target,
    const std::vector<float> &value_target, int batch, Result &result) {
  return ForwardBackward(
      policy_logits, values, legal_mask, policy_target, value_target,
      std::vector<std::uint8_t>(std::max(0, batch), 1), batch, result);
}

PolicyValueLoss::RC PolicyValueLoss::ForwardBackward(
    const std::vector<float> &policy_logits,
    const std::vector<float> &values,
    const std::vector<std::uint8_t> &legal_mask,
    const std::vector<float> &policy_target,
    const std::vector<float> &value_target,
    const std::vector<std::uint8_t> &value_mask, int batch, Result &result) {
  if (!is_init_) {
    err_msg_ = "[PolicyValueLoss::ForwardBackward] not initialized";
    return NOT_INIT;
  }
  const std::size_t policy_elements =
      static_cast<std::size_t>(batch) * config_.policy_size_;
  if (batch <= 0 || policy_logits.size() != policy_elements ||
      legal_mask.size() != policy_elements ||
      policy_target.size() != policy_elements ||
      values.size() != static_cast<std::size_t>(batch) ||
      value_target.size() != static_cast<std::size_t>(batch) ||
      value_mask.size() != static_cast<std::size_t>(batch)) {
    err_msg_ = "[PolicyValueLoss::ForwardBackward] invalid shape";
    return INVALID_DATA;
  }
  int value_count = 0;
  for (std::uint8_t mask : value_mask) {
    if (mask > 1) {
      err_msg_ = "[PolicyValueLoss::ForwardBackward] invalid value mask";
      return INVALID_DATA;
    }
    value_count += mask;
  }

  result = Result();
  result.grad_policy_logits_.assign(policy_elements, 0.0f);
  result.grad_values_.assign(batch, 0.0f);
  double policy_loss = 0.0;
  double value_loss = 0.0;
  const float inverse_batch = 1.0f / batch;
  const float inverse_value_count =
      value_count > 0 ? 1.0f / value_count : 0.0f;

  for (int sample = 0; sample < batch; sample++) {
    const std::size_t base =
        static_cast<std::size_t>(sample) * config_.policy_size_;
    float maximum = -std::numeric_limits<float>::infinity();
    int legal_count = 0;
    double target_sum = 0.0;
    for (int action = 0; action < config_.policy_size_; action++) {
      const std::size_t index = base + action;
      const float target = policy_target[index];
      if (!std::isfinite(policy_logits[index]) || !std::isfinite(target) ||
          target < 0.0f || (!legal_mask[index] && target != 0.0f)) {
        err_msg_ =
            "[PolicyValueLoss::ForwardBackward] invalid policy target";
        return INVALID_DATA;
      }
      if (legal_mask[index]) {
        legal_count++;
        maximum = std::max(maximum, policy_logits[index]);
        target_sum += target;
      }
    }
    if (legal_count == 0 || std::fabs(target_sum - 1.0) > 1e-5) {
      err_msg_ =
          "[PolicyValueLoss::ForwardBackward] legal target must sum to one";
      return INVALID_DATA;
    }

    double exponential_sum = 0.0;
    for (int action = 0; action < config_.policy_size_; action++) {
      const std::size_t index = base + action;
      if (legal_mask[index]) {
        exponential_sum +=
            std::exp(static_cast<double>(policy_logits[index] - maximum));
      }
    }
    const double log_normalizer = maximum + std::log(exponential_sum);
    for (int action = 0; action < config_.policy_size_; action++) {
      const std::size_t index = base + action;
      if (!legal_mask[index]) {
        continue;
      }
      const double probability =
          std::exp(static_cast<double>(policy_logits[index] - maximum)) /
          exponential_sum;
      const float target = policy_target[index];
      if (target > 0.0f) {
        policy_loss -=
            target * (static_cast<double>(policy_logits[index]) -
                      log_normalizer);
      }
      result.grad_policy_logits_[index] =
          config_.policy_weight_ * inverse_batch *
          (static_cast<float>(probability) - target);
    }

    if (!std::isfinite(values[sample]) ||
        (value_mask[sample] &&
         (!std::isfinite(value_target[sample]) ||
          value_target[sample] < -1.0f || value_target[sample] > 1.0f))) {
      err_msg_ = "[PolicyValueLoss::ForwardBackward] invalid value target";
      return INVALID_DATA;
    }
    if (value_mask[sample]) {
      const float difference = values[sample] - value_target[sample];
      value_loss += difference * difference;
      result.grad_values_[sample] =
          config_.value_weight_ * inverse_value_count * 2.0f * difference;
    }
  }

  result.policy_loss_ =
      config_.policy_weight_ * static_cast<float>(policy_loss / batch);
  result.value_loss_ =
      config_.value_weight_ * static_cast<float>(
          value_count > 0 ? value_loss / value_count : 0.0);
  result.total_loss_ = result.policy_loss_ + result.value_loss_;
  return SUCCESS;
}

} // namespace deeplearning

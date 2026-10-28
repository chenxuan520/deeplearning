#include "loss/policy_value_wdl_loss.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace deeplearning {

PolicyValueWdlLoss::RC PolicyValueWdlLoss::Init(const Config &config) {
  if (is_init_) {
    err_msg_ = "[PolicyValueWdlLoss::Init] already initialized";
    return ALREADY_INIT;
  }
  if (config.policy_size_ <= 0 || config.policy_weight_ < 0.0f ||
      config.value_weight_ < 0.0f ||
      (config.policy_weight_ == 0.0f && config.value_weight_ == 0.0f)) {
    err_msg_ = "[PolicyValueWdlLoss::Init] invalid config";
    return INVALID_DATA;
  }
  config_ = config;
  is_init_ = true;
  return SUCCESS;
}

PolicyValueWdlLoss::RC PolicyValueWdlLoss::ForwardBackward(
    const std::vector<float> &policy_logits,
    const std::vector<float> &value_probabilities,
    const std::vector<std::uint8_t> &legal_mask,
    const std::vector<float> &policy_target,
    const std::vector<float> &value_target,
    const std::vector<std::uint8_t> &value_mask, int batch, Result &result) {
  if (!is_init_) {
    err_msg_ = "[PolicyValueWdlLoss::ForwardBackward] not initialized";
    return NOT_INIT;
  }
  const std::size_t policy_elements =
      static_cast<std::size_t>(batch) * config_.policy_size_;
  const std::size_t value_elements =
      static_cast<std::size_t>(batch) * kValueClassNum;
  if (batch <= 0 || policy_logits.size() != policy_elements ||
      legal_mask.size() != policy_elements ||
      policy_target.size() != policy_elements ||
      value_probabilities.size() != value_elements ||
      value_target.size() != value_elements ||
      value_mask.size() != static_cast<std::size_t>(batch)) {
    err_msg_ = "[PolicyValueWdlLoss::ForwardBackward] invalid shape";
    return INVALID_DATA;
  }
  int value_count = 0;
  for (std::uint8_t mask : value_mask) {
    if (mask > 1) {
      err_msg_ = "[PolicyValueWdlLoss::ForwardBackward] invalid value mask";
      return INVALID_DATA;
    }
    value_count += mask;
  }

  result = Result();
  result.grad_policy_logits_.assign(policy_elements, 0.0f);
  result.grad_value_probabilities_.assign(value_elements, 0.0f);
  double policy_loss = 0.0;
  double value_loss = 0.0;
  const float inverse_batch = 1.0f / batch;
  const float inverse_value_count =
      value_count > 0 ? 1.0f / value_count : 0.0f;

  for (int sample = 0; sample < batch; sample++) {
    const std::size_t policy_base =
        static_cast<std::size_t>(sample) * config_.policy_size_;
    float policy_maximum = -std::numeric_limits<float>::infinity();
    int legal_count = 0;
    double policy_target_sum = 0.0;
    for (int action = 0; action < config_.policy_size_; action++) {
      const std::size_t index = policy_base + action;
      const float target = policy_target[index];
      if (!std::isfinite(policy_logits[index]) || !std::isfinite(target) ||
          target < 0.0f || (!legal_mask[index] && target != 0.0f)) {
        err_msg_ =
            "[PolicyValueWdlLoss::ForwardBackward] invalid policy target";
        return INVALID_DATA;
      }
      if (legal_mask[index]) {
        legal_count++;
        policy_maximum = std::max(policy_maximum, policy_logits[index]);
        policy_target_sum += target;
      }
    }
    if (legal_count == 0 || std::fabs(policy_target_sum - 1.0) > 1e-5) {
      err_msg_ =
          "[PolicyValueWdlLoss::ForwardBackward] legal target must sum to one";
      return INVALID_DATA;
    }
    double policy_exponential_sum = 0.0;
    for (int action = 0; action < config_.policy_size_; action++) {
      const std::size_t index = policy_base + action;
      if (legal_mask[index]) {
        policy_exponential_sum += std::exp(
            static_cast<double>(policy_logits[index] - policy_maximum));
      }
    }
    const double policy_log_normalizer =
        policy_maximum + std::log(policy_exponential_sum);
    for (int action = 0; action < config_.policy_size_; action++) {
      const std::size_t index = policy_base + action;
      if (!legal_mask[index]) {
        continue;
      }
      const double probability =
          std::exp(static_cast<double>(
              policy_logits[index] - policy_maximum)) /
          policy_exponential_sum;
      const float target = policy_target[index];
      if (target > 0.0f) {
        policy_loss -= target *
            (static_cast<double>(policy_logits[index]) -
             policy_log_normalizer);
      }
      result.grad_policy_logits_[index] =
          config_.policy_weight_ * inverse_batch *
          (static_cast<float>(probability) - target);
    }

    const std::size_t value_base =
        static_cast<std::size_t>(sample) * kValueClassNum;
    double value_target_sum = 0.0;
    double value_probability_sum = 0.0;
    for (int outcome = 0; outcome < kValueClassNum; outcome++) {
      const float probability = value_probabilities[value_base + outcome];
      const float target = value_target[value_base + outcome];
      if (!std::isfinite(probability) || probability < 0.0f ||
          (value_mask[sample] &&
           (!std::isfinite(target) || target < 0.0f))) {
        err_msg_ =
            "[PolicyValueWdlLoss::ForwardBackward] invalid value target";
        return INVALID_DATA;
      }
      value_probability_sum += probability;
      if (value_mask[sample]) {
        value_target_sum += target;
      }
    }
    if (std::fabs(value_probability_sum - 1.0) > 1e-5 ||
        (value_mask[sample] && std::fabs(value_target_sum - 1.0) > 1e-5)) {
      err_msg_ =
          "[PolicyValueWdlLoss::ForwardBackward] value distribution must sum to one";
      return INVALID_DATA;
    }
    if (!value_mask[sample]) {
      continue;
    }
    for (int outcome = 0; outcome < kValueClassNum; outcome++) {
      const std::size_t index = value_base + outcome;
      const float target = value_target[index];
      if (target > 0.0f) {
        const double probability =
            std::max(1e-12, static_cast<double>(value_probabilities[index]));
        value_loss -= target * std::log(probability);
      }
      result.grad_value_probabilities_[index] =
          target > 0.0f
              ? -config_.value_weight_ * inverse_value_count * target /
                    std::max(1e-12f, value_probabilities[index])
              : 0.0f;
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

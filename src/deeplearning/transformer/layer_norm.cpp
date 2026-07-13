#include "layer_norm.h"

#include <cmath>

namespace deeplearning {

LayerNorm::RC LayerNorm::Init(int feature_dim, double epsilon) {
  if (is_init_) {
    err_msg_ = "[LayerNorm::Init] LayerNorm has init";
    return ALREADY_INIT;
  }
  if (feature_dim <= 0 || epsilon <= 0) {
    err_msg_ = "[LayerNorm::Init] Invalid feature dim";
    return INVALID_DATA;
  }

  feature_dim_ = feature_dim;
  epsilon_ = epsilon;
  scale_.assign(feature_dim_, 1.0);
  bias_.assign(feature_dim_, 0.0);
  grad_scale_.assign(feature_dim_, 0.0);
  grad_bias_.assign(feature_dim_, 0.0);
  is_init_ = true;
  return SUCCESS;
}

LayerNorm::RC
LayerNorm::Forward(const std::vector<std::vector<double>> &input,
                   std::vector<std::vector<double>> &output) {
  if (!is_init_) {
    err_msg_ = "[LayerNorm::Forward] LayerNorm not init";
    return NOT_INIT;
  }
  if (input.empty()) {
    err_msg_ = "[LayerNorm::Forward] Invalid data input";
    return INVALID_DATA;
  }

  last_input_ = input;
  output = input;
  for (auto &token : output) {
    if (token.size() != static_cast<size_t>(feature_dim_)) {
      err_msg_ = "[LayerNorm::Forward] Invalid data input";
      return INVALID_DATA;
    }

    double mean = 0;
    for (double value : token) {
      mean += value;
    }
    mean /= feature_dim_;

    double variance = 0;
    for (double value : token) {
      double diff = value - mean;
      variance += diff * diff;
    }
    variance /= feature_dim_;

    double denom = std::sqrt(variance + epsilon_);
    for (int i = 0; i < feature_dim_; i++) {
      token[i] = ((token[i] - mean) / denom) * scale_[i] + bias_[i];
    }
  }
  return SUCCESS;
}

LayerNorm::RC LayerNorm::Backward(
    const std::vector<std::vector<double>> &grad_output,
    std::vector<std::vector<double>> &grad_input, double learning_rate) {
  ClearGradients();
  auto rc = BackwardAccumulate(grad_output, grad_input);
  if (rc != SUCCESS) {
    return rc;
  }
  ApplyGradient(learning_rate);
  return SUCCESS;
}

LayerNorm::RC LayerNorm::BackwardAccumulate(
    const std::vector<std::vector<double>> &grad_output,
    std::vector<std::vector<double>> &grad_input) {
  if (!is_init_) {
    err_msg_ = "[LayerNorm::BackwardAccumulate] LayerNorm not init";
    return NOT_INIT;
  }
  if (grad_output.size() != last_input_.size() || grad_output.empty()) {
    err_msg_ = "[LayerNorm::BackwardAccumulate] Invalid data input";
    return INVALID_DATA;
  }

  grad_input.assign(grad_output.size(), std::vector<double>(feature_dim_, 0));
  for (int token_idx = 0; token_idx < static_cast<int>(grad_output.size());
       token_idx++) {
    if (grad_output[token_idx].size() != static_cast<size_t>(feature_dim_) ||
        last_input_[token_idx].size() != static_cast<size_t>(feature_dim_)) {
      err_msg_ = "[LayerNorm::BackwardAccumulate] Invalid data input";
      return INVALID_DATA;
    }

    double mean = 0;
    for (double value : last_input_[token_idx]) {
      mean += value;
    }
    mean /= feature_dim_;

    double variance = 0;
    std::vector<double> x_hat(feature_dim_, 0);
    for (int i = 0; i < feature_dim_; i++) {
      double diff = last_input_[token_idx][i] - mean;
      variance += diff * diff;
    }
    variance /= feature_dim_;

    double std_inv = 1.0 / std::sqrt(variance + epsilon_);
    double sum_dx_hat = 0;
    double sum_dx_hat_x_hat = 0;
    for (int i = 0; i < feature_dim_; i++) {
      x_hat[i] = (last_input_[token_idx][i] - mean) * std_inv;
      grad_scale_[i] += grad_output[token_idx][i] * x_hat[i];
      grad_bias_[i] += grad_output[token_idx][i];

      double dx_hat = grad_output[token_idx][i] * scale_[i];
      sum_dx_hat += dx_hat;
      sum_dx_hat_x_hat += dx_hat * x_hat[i];
    }

    for (int i = 0; i < feature_dim_; i++) {
      double dx_hat = grad_output[token_idx][i] * scale_[i];
      grad_input[token_idx][i] =
          std_inv * (feature_dim_ * dx_hat - sum_dx_hat -
                     x_hat[i] * sum_dx_hat_x_hat) /
          feature_dim_;
    }
  }

  return SUCCESS;
}

void LayerNorm::ApplyGradient(double learning_rate, double gradient_scale) {
  scale_optimizer_.Apply(scale_, grad_scale_, learning_rate, gradient_scale);
  bias_optimizer_.Apply(bias_, grad_bias_, learning_rate, gradient_scale);
  ClearGradients();
}

void LayerNorm::ClearGradients() {
  grad_scale_.assign(feature_dim_, 0.0);
  grad_bias_.assign(feature_dim_, 0.0);
}

LayerNorm::RC LayerNorm::set_scale(const std::vector<double> &scale) {
  if (!is_init_) {
    err_msg_ = "[LayerNorm::set_scale] LayerNorm not init";
    return NOT_INIT;
  }
  if (scale.size() != static_cast<size_t>(feature_dim_)) {
    err_msg_ = "[LayerNorm::set_scale] Invalid scale size";
    return INVALID_DATA;
  }
  scale_ = scale;
  return SUCCESS;
}

LayerNorm::RC LayerNorm::set_bias(const std::vector<double> &bias) {
  if (!is_init_) {
    err_msg_ = "[LayerNorm::set_bias] LayerNorm not init";
    return NOT_INIT;
  }
  if (bias.size() != static_cast<size_t>(feature_dim_)) {
    err_msg_ = "[LayerNorm::set_bias] Invalid bias size";
    return INVALID_DATA;
  }
  bias_ = bias;
  return SUCCESS;
}

std::string LayerNorm::err_msg() { return err_msg_; }

const std::vector<double> &LayerNorm::scale() const { return scale_; }

const std::vector<double> &LayerNorm::bias() const { return bias_; }

} // namespace deeplearning

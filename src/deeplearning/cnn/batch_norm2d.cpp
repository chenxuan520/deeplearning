#include "cnn/batch_norm2d.h"

#include "util/thread_pool.h"

#include <algorithm>
#include <cmath>

namespace deeplearning {

BatchNorm2D::RC BatchNorm2D::Init(const Config &config) {
  if (is_init_) {
    err_msg_ = "[BatchNorm2D::Init] already initialized";
    return ALREADY_INIT;
  }
  if (config.channels_ <= 0 || config.epsilon_ <= 0.0f ||
      config.momentum_ < 0.0f || config.momentum_ > 1.0f ||
      config.thread_num_ <= 0) {
    err_msg_ = "[BatchNorm2D::Init] invalid config";
    return INVALID_DATA;
  }
  config_ = config;
  scale_.assign(config.channels_, 1.0f);
  bias_.assign(config.channels_, 0.0f);
  running_mean_.assign(config.channels_, 0.0f);
  running_variance_.assign(config.channels_, 1.0f);
  grad_scale_.assign(config.channels_, 0.0f);
  grad_bias_.assign(config.channels_, 0.0f);
  inverse_std_cache_.assign(config.channels_, 0.0f);
  is_init_ = true;
  return SUCCESS;
}

bool BatchNorm2D::ValidateInput(const FloatTensor4D &input,
                                const char *function) {
  if (!is_init_) {
    err_msg_ =
        std::string("[BatchNorm2D::") + function + "] not initialized";
    return false;
  }
  if (input.batch() <= 0 || input.channels() != config_.channels_ ||
      input.height() <= 0 || input.width() <= 0) {
    err_msg_ =
        std::string("[BatchNorm2D::") + function + "] invalid input shape";
    return false;
  }
  return true;
}

BatchNorm2D::RC BatchNorm2D::Forward(const FloatTensor4D &input,
                                     FloatTensor4D &output, bool training,
                                     bool use_running_statistics) {
  if (!ValidateInput(input, "Forward")) {
    return is_init_ ? INVALID_DATA : NOT_INIT;
  }
  const int spatial_count = input.batch() * input.height() * input.width();
  std::vector<float> mean(config_.channels_, 0.0f);
  std::vector<float> variance(config_.channels_, 0.0f);
  if (training && !use_running_statistics) {
    ThreadPool::Global().Run(
        config_.channels_, config_.thread_num_, [&](int begin, int end) {
          for (int channel = begin; channel < end; channel++) {
            double sum = 0.0;
            double squared_sum = 0.0;
            for (int batch = 0; batch < input.batch(); batch++) {
              for (int row = 0; row < input.height(); row++) {
                for (int column = 0; column < input.width(); column++) {
                  const double value = input(batch, channel, row, column);
                  sum += value;
                  squared_sum += value * value;
                }
              }
            }
            const double channel_mean = sum / spatial_count;
            mean[channel] = static_cast<float>(channel_mean);
            variance[channel] = static_cast<float>(
                std::max(0.0, squared_sum / spatial_count -
                                  channel_mean * channel_mean));
          }
        });
    for (int channel = 0; channel < config_.channels_; channel++) {
      running_mean_[channel] =
          (1.0f - config_.momentum_) * running_mean_[channel] +
          config_.momentum_ * mean[channel];
      running_variance_[channel] =
          (1.0f - config_.momentum_) * running_variance_[channel] +
          config_.momentum_ * variance[channel];
    }
  } else {
    mean = running_mean_;
    variance = running_variance_;
  }

  output.Resize(input.batch(), input.channels(), input.height(), input.width());
  if (training) {
    normalized_cache_.Resize(input.batch(), input.channels(), input.height(),
                             input.width());
  }
  ThreadPool::Global().Run(
      config_.channels_, config_.thread_num_, [&](int begin, int end) {
        for (int channel = begin; channel < end; channel++) {
          const float inverse_std =
              1.0f / std::sqrt(variance[channel] + config_.epsilon_);
          if (training) {
            inverse_std_cache_[channel] = inverse_std;
          }
          for (int batch = 0; batch < input.batch(); batch++) {
            for (int row = 0; row < input.height(); row++) {
              for (int column = 0; column < input.width(); column++) {
                const float normalized =
                    (input(batch, channel, row, column) - mean[channel]) *
                    inverse_std;
                if (training) {
                  normalized_cache_(batch, channel, row, column) = normalized;
                }
                output(batch, channel, row, column) =
                    scale_[channel] * normalized + bias_[channel];
              }
            }
          }
        }
      });
  has_cache_ = training;
  fixed_statistics_cache_ = training && use_running_statistics;
  return SUCCESS;
}

BatchNorm2D::RC BatchNorm2D::Backward(
    const FloatTensor4D &grad_output, FloatTensor4D &grad_input) {
  if (!is_init_) {
    err_msg_ = "[BatchNorm2D::Backward] not initialized";
    return NOT_INIT;
  }
  if (!has_cache_) {
    err_msg_ = "[BatchNorm2D::Backward] missing training cache";
    return MISSING_CACHE;
  }
  if (!grad_output.SameShape(normalized_cache_)) {
    err_msg_ = "[BatchNorm2D::Backward] invalid gradient shape";
    return INVALID_DATA;
  }
  const int spatial_count =
      grad_output.batch() * grad_output.height() * grad_output.width();
  grad_input.Resize(grad_output.batch(), grad_output.channels(),
                    grad_output.height(), grad_output.width());
  if (fixed_statistics_cache_) {
    ThreadPool::Global().Run(
        config_.channels_, config_.thread_num_, [&](int begin, int end) {
          for (int channel = begin; channel < end; channel++) {
            double sum_gradient = 0.0;
            double sum_gradient_normalized = 0.0;
            const float coefficient =
                scale_[channel] * inverse_std_cache_[channel];
            for (int batch = 0; batch < grad_output.batch(); batch++) {
              for (int row = 0; row < grad_output.height(); row++) {
                for (int column = 0; column < grad_output.width(); column++) {
                  const float gradient =
                      grad_output(batch, channel, row, column);
                  sum_gradient += gradient;
                  sum_gradient_normalized +=
                      gradient *
                      normalized_cache_(batch, channel, row, column);
                  grad_input(batch, channel, row, column) =
                      coefficient * gradient;
                }
              }
            }
            grad_bias_[channel] += static_cast<float>(sum_gradient);
            grad_scale_[channel] +=
                static_cast<float>(sum_gradient_normalized);
          }
        });
    return SUCCESS;
  }
  ThreadPool::Global().Run(
      config_.channels_, config_.thread_num_, [&](int begin, int end) {
        for (int channel = begin; channel < end; channel++) {
          double sum_gradient = 0.0;
          double sum_gradient_normalized = 0.0;
          for (int batch = 0; batch < grad_output.batch(); batch++) {
            for (int row = 0; row < grad_output.height(); row++) {
              for (int column = 0; column < grad_output.width(); column++) {
                const float gradient =
                    grad_output(batch, channel, row, column);
                sum_gradient += gradient;
                sum_gradient_normalized +=
                    gradient *
                    normalized_cache_(batch, channel, row, column);
              }
            }
          }
          grad_bias_[channel] += static_cast<float>(sum_gradient);
          grad_scale_[channel] +=
              static_cast<float>(sum_gradient_normalized);
          const float coefficient =
              scale_[channel] * inverse_std_cache_[channel] / spatial_count;
          for (int batch = 0; batch < grad_output.batch(); batch++) {
            for (int row = 0; row < grad_output.height(); row++) {
              for (int column = 0; column < grad_output.width(); column++) {
                const float gradient =
                    grad_output(batch, channel, row, column);
                const float normalized =
                    normalized_cache_(batch, channel, row, column);
                grad_input(batch, channel, row, column) =
                    coefficient *
                    (spatial_count * gradient -
                     static_cast<float>(sum_gradient) -
                     normalized *
                         static_cast<float>(sum_gradient_normalized));
              }
            }
          }
        }
      });
  return SUCCESS;
}

void BatchNorm2D::ZeroGrad() {
  std::fill(grad_scale_.begin(), grad_scale_.end(), 0.0f);
  std::fill(grad_bias_.begin(), grad_bias_.end(), 0.0f);
}

void BatchNorm2D::InferenceAffine(std::vector<float> &scale,
                                  std::vector<float> &bias) const {
  scale.resize(config_.channels_);
  bias.resize(config_.channels_);
  for (int channel = 0; channel < config_.channels_; channel++) {
    scale[channel] =
        scale_[channel] /
        std::sqrt(running_variance_[channel] + config_.epsilon_);
    bias[channel] =
        bias_[channel] - running_mean_[channel] * scale[channel];
  }
}

BatchNorm2D::RC BatchNorm2D::set_scale(
    const std::vector<float> &scale) {
  if (!is_init_) {
    return NOT_INIT;
  }
  if (scale.size() != scale_.size()) {
    return INVALID_DATA;
  }
  scale_ = scale;
  return SUCCESS;
}

BatchNorm2D::RC BatchNorm2D::set_bias(const std::vector<float> &bias) {
  if (!is_init_) {
    return NOT_INIT;
  }
  if (bias.size() != bias_.size()) {
    return INVALID_DATA;
  }
  bias_ = bias;
  return SUCCESS;
}

BatchNorm2D::RC BatchNorm2D::set_running_mean(
    const std::vector<float> &mean) {
  if (!is_init_) {
    return NOT_INIT;
  }
  if (mean.size() != running_mean_.size()) {
    return INVALID_DATA;
  }
  running_mean_ = mean;
  return SUCCESS;
}

BatchNorm2D::RC BatchNorm2D::set_running_variance(
    const std::vector<float> &variance) {
  if (!is_init_) {
    return NOT_INIT;
  }
  if (variance.size() != running_variance_.size() ||
      std::any_of(variance.begin(), variance.end(),
                  [](float value) { return value < 0.0f; })) {
    return INVALID_DATA;
  }
  running_variance_ = variance;
  return SUCCESS;
}

} // namespace deeplearning

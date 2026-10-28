#include "cnn/float_linear.h"

#include "util/thread_pool.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace deeplearning {

FloatLinear::RC FloatLinear::Init(const Config &config) {
  if (is_init_) {
    err_msg_ = "[FloatLinear::Init] already initialized";
    return ALREADY_INIT;
  }
  if (config.input_dim_ <= 0 || config.output_dim_ <= 0 ||
      config.thread_num_ <= 0) {
    err_msg_ = "[FloatLinear::Init] invalid config";
    return INVALID_DATA;
  }
  config_ = config;
  weight_.resize(static_cast<std::size_t>(config.output_dim_) *
                 config.input_dim_);
  bias_.assign(config.output_dim_, 0.0f);
  grad_weight_.assign(weight_.size(), 0.0f);
  grad_bias_.assign(bias_.size(), 0.0f);
  const float limit =
      std::sqrt(6.0f / static_cast<float>(config.input_dim_ +
                                         config.output_dim_));
  std::mt19937 random(static_cast<std::mt19937::result_type>(config.rand_seed_));
  std::uniform_real_distribution<float> distribution(-limit, limit);
  for (float &value : weight_) {
    value = distribution(random);
  }
  is_init_ = true;
  return SUCCESS;
}

FloatLinear::RC FloatLinear::Forward(const std::vector<float> &input,
                                     int batch,
                                     std::vector<float> &output,
                                     bool cache_for_backward) {
  if (!is_init_) {
    err_msg_ = "[FloatLinear::Forward] not initialized";
    return NOT_INIT;
  }
  if (batch <= 0 ||
      input.size() !=
          static_cast<std::size_t>(batch) * config_.input_dim_) {
    err_msg_ = "[FloatLinear::Forward] invalid input shape";
    return INVALID_DATA;
  }
  output.assign(static_cast<std::size_t>(batch) * config_.output_dim_, 0.0f);
  ThreadPool::Global().Run(
      batch, config_.thread_num_, [&](int begin, int end) {
        for (int sample = begin; sample < end; sample++) {
          const float *source =
              input.data() + static_cast<std::size_t>(sample) *
                                 config_.input_dim_;
          float *destination =
              output.data() + static_cast<std::size_t>(sample) *
                                  config_.output_dim_;
          for (int output_index = 0; output_index < config_.output_dim_;
               output_index++) {
            const float *weight =
                weight_.data() +
                static_cast<std::size_t>(output_index) * config_.input_dim_;
            float sum = config_.use_bias_ ? bias_[output_index] : 0.0f;
            for (int input_index = 0; input_index < config_.input_dim_;
                 input_index++) {
              sum += source[input_index] * weight[input_index];
            }
            destination[output_index] = sum;
          }
        }
      });
  if (cache_for_backward) {
    input_cache_ = input;
    cached_batch_ = batch;
    has_cache_ = true;
  } else {
    input_cache_.clear();
    cached_batch_ = 0;
    has_cache_ = false;
  }
  return SUCCESS;
}

FloatLinear::RC FloatLinear::Backward(
    const std::vector<float> &grad_output,
    std::vector<float> &grad_input) {
  if (!is_init_) {
    err_msg_ = "[FloatLinear::Backward] not initialized";
    return NOT_INIT;
  }
  if (!has_cache_) {
    err_msg_ = "[FloatLinear::Backward] missing forward cache";
    return MISSING_CACHE;
  }
  if (grad_output.size() !=
      static_cast<std::size_t>(cached_batch_) * config_.output_dim_) {
    err_msg_ = "[FloatLinear::Backward] invalid gradient shape";
    return INVALID_DATA;
  }
  grad_input.assign(
      static_cast<std::size_t>(cached_batch_) * config_.input_dim_, 0.0f);

  // Each output unit owns one disjoint parameter row.
  ThreadPool::Global().Run(
      config_.output_dim_, config_.thread_num_, [&](int begin, int end) {
        for (int output_index = begin; output_index < end; output_index++) {
          float bias_gradient = 0.0f;
          float *weight_gradient =
              grad_weight_.data() +
              static_cast<std::size_t>(output_index) * config_.input_dim_;
          for (int sample = 0; sample < cached_batch_; sample++) {
            const float gradient =
                grad_output[static_cast<std::size_t>(sample) *
                                config_.output_dim_ +
                            output_index];
            bias_gradient += gradient;
            const float *source =
                input_cache_.data() +
                static_cast<std::size_t>(sample) * config_.input_dim_;
            for (int input_index = 0; input_index < config_.input_dim_;
                 input_index++) {
              weight_gradient[input_index] += gradient * source[input_index];
            }
          }
          if (config_.use_bias_) {
            grad_bias_[output_index] += bias_gradient;
          }
        }
      });

  ThreadPool::Global().Run(
      cached_batch_, config_.thread_num_, [&](int begin, int end) {
        for (int sample = begin; sample < end; sample++) {
          float *destination =
              grad_input.data() +
              static_cast<std::size_t>(sample) * config_.input_dim_;
          const float *output_gradient =
              grad_output.data() +
              static_cast<std::size_t>(sample) * config_.output_dim_;
          for (int output_index = 0; output_index < config_.output_dim_;
               output_index++) {
            const float gradient = output_gradient[output_index];
            const float *weight =
                weight_.data() +
                static_cast<std::size_t>(output_index) * config_.input_dim_;
            for (int input_index = 0; input_index < config_.input_dim_;
                 input_index++) {
              destination[input_index] += gradient * weight[input_index];
            }
          }
        }
      });
  return SUCCESS;
}

void FloatLinear::ZeroGrad() {
  std::fill(grad_weight_.begin(), grad_weight_.end(), 0.0f);
  std::fill(grad_bias_.begin(), grad_bias_.end(), 0.0f);
}

FloatLinear::RC FloatLinear::set_weight(
    const std::vector<float> &weight) {
  if (!is_init_) {
    return NOT_INIT;
  }
  if (weight.size() != weight_.size()) {
    return INVALID_DATA;
  }
  weight_ = weight;
  return SUCCESS;
}

FloatLinear::RC FloatLinear::set_bias(const std::vector<float> &bias) {
  if (!is_init_) {
    return NOT_INIT;
  }
  if (bias.size() != bias_.size()) {
    return INVALID_DATA;
  }
  bias_ = bias;
  return SUCCESS;
}

} // namespace deeplearning

#include "cnn/single_residual_adapter2d.h"

#include "util/thread_pool.h"

#include <algorithm>

#ifdef DEEPLEARNING_EMBED_SINGLE_RESIDUAL_ADAPTER2D

namespace deeplearning {

SingleResidualAdapter2D::RC SingleResidualAdapter2D::Init(
    const Config &config) {
  if (is_init_) {
    err_msg_ = "[SingleResidualAdapter2D::Init] already initialized";
    return ALREADY_INIT;
  }
  if (config.channels_ <= 0 || config.bottleneck_channels_ <= 0 ||
      config.height_ <= 0 || config.width_ <= 0 ||
      config.thread_num_ <= 0 || config.rand_seed_ <= 0) {
    err_msg_ = "[SingleResidualAdapter2D::Init] invalid config";
    return INVALID_DATA;
  }
  config_ = config;
  BatchedConv2D::Config down;
  down.input_channels_ = config.channels_;
  down.output_channels_ = config.bottleneck_channels_;
  down.kernel_height_ = 1;
  down.kernel_width_ = 1;
  down.padding_ = 0;
  down.use_bias_ = false;
  down.thread_num_ = config.thread_num_;
  down.rand_seed_ = config.rand_seed_;
  BatchedConv2D::Config up = down;
  up.input_channels_ = config.bottleneck_channels_;
  up.output_channels_ = config.channels_;
  up.rand_seed_ = config.rand_seed_ + 1;
  if (down_.Init(down) != BatchedConv2D::SUCCESS ||
      up_.Init(up) != BatchedConv2D::SUCCESS) {
    err_msg_ = "[SingleResidualAdapter2D::Init] layer initialization failed";
    return INVALID_DATA;
  }
  std::vector<float> zero(up_.weight().size(), 0.0f);
  if (up_.set_weight(zero) != BatchedConv2D::SUCCESS) {
    err_msg_ = "[SingleResidualAdapter2D::Init] zero output failed";
    return INVALID_DATA;
  }
  is_init_ = true;
  return SUCCESS;
}

bool SingleResidualAdapter2D::ValidateInput(
    const FloatTensor4D &input, const char *function) {
  if (!is_init_) {
    err_msg_ = std::string("[SingleResidualAdapter2D::") + function +
               "] not initialized";
    return false;
  }
  if (input.batch() <= 0 || input.channels() != config_.channels_ ||
      input.height() != config_.height_ ||
      input.width() != config_.width_) {
    err_msg_ = std::string("[SingleResidualAdapter2D::") + function +
               "] invalid input shape";
    return false;
  }
  return true;
}

SingleResidualAdapter2D::RC SingleResidualAdapter2D::Forward(
    const FloatTensor4D &input, FloatTensor4D &output, bool training) {
  if (!ValidateInput(input, "Forward")) {
    return is_init_ ? INVALID_DATA : NOT_INIT;
  }
  if (!training) return ForwardInference(input, output);
  FloatTensor4D hidden;
  if (down_.Forward(input, hidden, true) != BatchedConv2D::SUCCESS) {
    err_msg_ = down_.err_msg();
    return INVALID_DATA;
  }
  for (float &value : hidden.values()) value = std::max(0.0f, value);
  FloatTensor4D residual;
  if (up_.Forward(hidden, residual, true) != BatchedConv2D::SUCCESS) {
    err_msg_ = up_.err_msg();
    return INVALID_DATA;
  }
  output = input;
  for (std::size_t index = 0; index < output.size(); index++) {
    output.values()[index] += residual.values()[index];
  }
  relu_cache_ = std::move(hidden);
  has_cache_ = true;
  return SUCCESS;
}

SingleResidualAdapter2D::RC SingleResidualAdapter2D::ForwardInference(
    const FloatTensor4D &input, FloatTensor4D &output) {
  const int batch = input.batch();
  const int channels = config_.channels_;
  const int bottleneck = config_.bottleneck_channels_;
  const int spatial = config_.height_ * config_.width_;
  const int positions = batch * spatial;
  const auto &down_weight = down_.weight();
  const auto &up_weight = up_.weight();
  output = input;
  ThreadPool::Global().Run(
      positions, config_.thread_num_, [&](int begin, int end) {
        std::vector<float> hidden(bottleneck);
        for (int position = begin; position < end; position++) {
          const int sample = position / spatial;
          const int cell = position % spatial;
          for (int unit = 0; unit < bottleneck; unit++) {
            float sum = 0.0f;
            const float *row =
                down_weight.data() +
                static_cast<std::size_t>(unit) * channels;
            for (int channel = 0; channel < channels; channel++) {
              const float value = input.values()[
                  (static_cast<std::size_t>(sample) * channels + channel) *
                      spatial +
                  cell];
              sum += row[channel] * value;
            }
            hidden[unit] = std::max(0.0f, sum);
          }
          for (int channel = 0; channel < channels; channel++) {
            float sum = 0.0f;
            const float *row =
                up_weight.data() +
                static_cast<std::size_t>(channel) * bottleneck;
            for (int unit = 0; unit < bottleneck; unit++) {
              sum += row[unit] * hidden[unit];
            }
            const std::size_t offset =
                (static_cast<std::size_t>(sample) * channels + channel) *
                    spatial +
                cell;
            output.values()[offset] += sum;
          }
        }
      });
  has_cache_ = false;
  return SUCCESS;
}

SingleResidualAdapter2D::RC SingleResidualAdapter2D::Backward(
    const FloatTensor4D &grad_output, FloatTensor4D &grad_input) {
  if (!is_init_) {
    err_msg_ = "[SingleResidualAdapter2D::Backward] not initialized";
    return NOT_INIT;
  }
  if (!has_cache_) {
    err_msg_ = "[SingleResidualAdapter2D::Backward] missing training cache";
    return MISSING_CACHE;
  }
  if (grad_output.batch() <= 0 ||
      grad_output.channels() != config_.channels_ ||
      grad_output.height() != config_.height_ ||
      grad_output.width() != config_.width_) {
    err_msg_ = "[SingleResidualAdapter2D::Backward] invalid gradient shape";
    return INVALID_DATA;
  }
  FloatTensor4D grad_hidden;
  if (up_.Backward(grad_output, grad_hidden) != BatchedConv2D::SUCCESS) {
    err_msg_ = up_.err_msg();
    return INVALID_DATA;
  }
  for (std::size_t index = 0; index < grad_hidden.size(); index++) {
    if (relu_cache_.values()[index] <= 0.0f) {
      grad_hidden.values()[index] = 0.0f;
    }
  }
  FloatTensor4D grad_residual_input;
  if (down_.Backward(grad_hidden, grad_residual_input) !=
      BatchedConv2D::SUCCESS) {
    err_msg_ = down_.err_msg();
    return INVALID_DATA;
  }
  grad_input = grad_output;
  for (std::size_t index = 0; index < grad_input.size(); index++) {
    grad_input.values()[index] += grad_residual_input.values()[index];
  }
  return SUCCESS;
}

void SingleResidualAdapter2D::ZeroGrad() {
  down_.ZeroGrad();
  up_.ZeroGrad();
}

} // namespace deeplearning

#endif

#include "cnn/residual_block2d.h"

#include "util/thread_pool.h"

#include <algorithm>

namespace deeplearning {

ResidualBlock2D::RC ResidualBlock2D::Init(const Config &config) {
  if (is_init_) {
    err_msg_ = "[ResidualBlock2D::Init] already initialized";
    return ALREADY_INIT;
  }
  if (config.channels_ <= 0 || config.height_ <= 0 || config.width_ <= 0 ||
      config.thread_num_ <= 0 || config.batch_norm_epsilon_ <= 0.0f ||
      config.batch_norm_momentum_ < 0.0f ||
      config.batch_norm_momentum_ > 1.0f) {
    err_msg_ = "[ResidualBlock2D::Init] invalid config";
    return INVALID_DATA;
  }
  config_ = config;
  BatchedConv2D::Config convolution;
  convolution.input_channels_ = config.channels_;
  convolution.output_channels_ = config.channels_;
  convolution.kernel_height_ = 3;
  convolution.kernel_width_ = 3;
  convolution.padding_ = 1;
  convolution.use_bias_ = false;
  convolution.thread_num_ = config.thread_num_;
  convolution.rand_seed_ = config.rand_seed_;
  if (conv1_.Init(convolution) != BatchedConv2D::SUCCESS) {
    err_msg_ = conv1_.err_msg();
    return INVALID_DATA;
  }
  convolution.rand_seed_ = config.rand_seed_ + 1;
  if (conv2_.Init(convolution) != BatchedConv2D::SUCCESS) {
    err_msg_ = conv2_.err_msg();
    return INVALID_DATA;
  }
  BatchNorm2D::Config normalization;
  normalization.channels_ = config.channels_;
  normalization.epsilon_ = config.batch_norm_epsilon_;
  normalization.momentum_ = config.batch_norm_momentum_;
  normalization.thread_num_ = config.thread_num_;
  if (norm1_.Init(normalization) != BatchNorm2D::SUCCESS ||
      norm2_.Init(normalization) != BatchNorm2D::SUCCESS) {
    err_msg_ = "[ResidualBlock2D::Init] normalization initialization failed";
    return INVALID_DATA;
  }
  is_init_ = true;
  return SUCCESS;
}

bool ResidualBlock2D::ValidateInput(const FloatTensor4D &input,
                                    const char *function) {
  if (!is_init_) {
    err_msg_ = std::string("[ResidualBlock2D::") + function +
               "] not initialized";
    return false;
  }
  if (input.batch() <= 0 || input.channels() != config_.channels_ ||
      input.height() != config_.height_ || input.width() != config_.width_) {
    err_msg_ = std::string("[ResidualBlock2D::") + function +
               "] invalid input shape";
    return false;
  }
  return true;
}

ResidualBlock2D::RC ResidualBlock2D::Forward(
    const FloatTensor4D &input, FloatTensor4D &output, bool training,
    bool use_running_statistics) {
  if (!ValidateInput(input, "Forward")) {
    return is_init_ ? INVALID_DATA : NOT_INIT;
  }
  if (!training) {
    std::vector<float> scale1, bias1, scale2, bias2;
    norm1_.InferenceAffine(scale1, bias1);
    norm2_.InferenceAffine(scale2, bias2);
    FloatTensor4D first;
    if (conv1_.ForwardAffine(input, first, scale1, bias1) !=
        BatchedConv2D::SUCCESS) {
      err_msg_ = conv1_.err_msg();
      return INVALID_DATA;
    }
    for (float &value : first.values()) {
      value = std::max(0.0f, value);
    }
    FloatTensor4D second;
    if (conv2_.ForwardAffine(first, second, scale2, bias2) !=
        BatchedConv2D::SUCCESS) {
      err_msg_ = conv2_.err_msg();
      return INVALID_DATA;
    }
    output.Resize(input.batch(), input.channels(), input.height(),
                  input.width());
    for (std::size_t index = 0; index < output.size(); index++) {
      output.values()[index] =
          std::max(0.0f, input.values()[index] + second.values()[index]);
    }
    has_cache_ = false;
    return SUCCESS;
  }
  FloatTensor4D conv1_output;
  if (conv1_.Forward(input, conv1_output, training) !=
      BatchedConv2D::SUCCESS) {
    err_msg_ = conv1_.err_msg();
    return INVALID_DATA;
  }
  FloatTensor4D norm1_output;
  if (norm1_.Forward(conv1_output, norm1_output, training,
                     use_running_statistics) !=
      BatchNorm2D::SUCCESS) {
    err_msg_ = norm1_.err_msg();
    return INVALID_DATA;
  }
  FloatTensor4D relu1 = norm1_output;
  for (float &value : relu1.values()) {
    value = std::max(0.0f, value);
  }
  FloatTensor4D conv2_output;
  if (conv2_.Forward(relu1, conv2_output, training) !=
      BatchedConv2D::SUCCESS) {
    err_msg_ = conv2_.err_msg();
    return INVALID_DATA;
  }
  FloatTensor4D norm2_output;
  if (norm2_.Forward(conv2_output, norm2_output, training,
                     use_running_statistics) !=
      BatchNorm2D::SUCCESS) {
    err_msg_ = norm2_.err_msg();
    return INVALID_DATA;
  }
  output.Resize(input.batch(), input.channels(), input.height(), input.width());
  for (std::size_t index = 0; index < output.size(); index++) {
    output.values()[index] =
        std::max(0.0f, input.values()[index] + norm2_output.values()[index]);
  }
  if (training) {
    relu1_cache_ = std::move(relu1);
    pre_relu2_cache_.Resize(input.batch(), input.channels(), input.height(),
                           input.width());
    for (std::size_t index = 0; index < input.size(); index++) {
      pre_relu2_cache_.values()[index] =
          input.values()[index] + norm2_output.values()[index];
    }
    has_cache_ = true;
  } else {
    has_cache_ = false;
  }
  return SUCCESS;
}

ResidualBlock2D::RC ResidualBlock2D::Backward(
    const FloatTensor4D &grad_output, FloatTensor4D &grad_input) {
  if (!is_init_) {
    err_msg_ = "[ResidualBlock2D::Backward] not initialized";
    return NOT_INIT;
  }
  if (!has_cache_) {
    err_msg_ = "[ResidualBlock2D::Backward] missing training cache";
    return MISSING_CACHE;
  }
  if (!grad_output.SameShape(pre_relu2_cache_)) {
    err_msg_ = "[ResidualBlock2D::Backward] invalid gradient shape";
    return INVALID_DATA;
  }
  FloatTensor4D grad_pre_relu2 = grad_output;
  for (std::size_t index = 0; index < grad_pre_relu2.size(); index++) {
    if (pre_relu2_cache_.values()[index] <= 0.0f) {
      grad_pre_relu2.values()[index] = 0.0f;
    }
  }
  FloatTensor4D grad_conv2;
  if (norm2_.Backward(grad_pre_relu2, grad_conv2) != BatchNorm2D::SUCCESS) {
    err_msg_ = norm2_.err_msg();
    return INVALID_DATA;
  }
  FloatTensor4D grad_relu1;
  if (conv2_.Backward(grad_conv2, grad_relu1) != BatchedConv2D::SUCCESS) {
    err_msg_ = conv2_.err_msg();
    return INVALID_DATA;
  }
  for (std::size_t index = 0; index < grad_relu1.size(); index++) {
    if (relu1_cache_.values()[index] <= 0.0f) {
      grad_relu1.values()[index] = 0.0f;
    }
  }
  FloatTensor4D grad_conv1;
  if (norm1_.Backward(grad_relu1, grad_conv1) != BatchNorm2D::SUCCESS) {
    err_msg_ = norm1_.err_msg();
    return INVALID_DATA;
  }
  FloatTensor4D grad_residual;
  if (conv1_.Backward(grad_conv1, grad_residual) !=
      BatchedConv2D::SUCCESS) {
    err_msg_ = conv1_.err_msg();
    return INVALID_DATA;
  }
  grad_input = grad_pre_relu2;
  for (std::size_t index = 0; index < grad_input.size(); index++) {
    grad_input.values()[index] += grad_residual.values()[index];
  }
  return SUCCESS;
}

void ResidualBlock2D::ZeroGrad() {
  conv1_.ZeroGrad();
  conv2_.ZeroGrad();
  norm1_.ZeroGrad();
  norm2_.ZeroGrad();
}

} // namespace deeplearning

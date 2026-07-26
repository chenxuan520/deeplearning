#include "cnn/conv2d.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace deeplearning {
namespace {

using Matrix = Conv2D::Matrix;
using Tensor3D = Conv2D::Tensor3D;
using Tensor4D = Conv2D::Tensor4D;

bool Tensor3DHasShape(const Tensor3D &tensor, int channel_num, int height,
                      int width) {
  if (static_cast<int>(tensor.size()) != channel_num) {
    return false;
  }
  for (const auto &channel : tensor) {
    if (static_cast<int>(channel.size()) != height) {
      return false;
    }
    for (const auto &row : channel) {
      if (static_cast<int>(row.size()) != width) {
        return false;
      }
    }
  }
  return true;
}

Tensor3D MakeTensor3D(int channel_num, int height, int width, double value = 0) {
  return Tensor3D(channel_num, Matrix(height, std::vector<double>(width, value)));
}

} // namespace

Conv2D::RC Conv2D::Init(int input_channels, int output_channels,
                        int kernel_height, int kernel_width, int stride,
                        int padding) {
  if (is_init_) {
    err_msg_ = "[Conv2D::Init] Conv2D has init";
    return ALREADY_INIT;
  }
  if (input_channels <= 0 || output_channels <= 0 || kernel_height <= 0 ||
      kernel_width <= 0 || stride <= 0 || padding < 0) {
    err_msg_ = "[Conv2D::Init] Invalid config";
    return INVALID_DATA;
  }

  input_channels_ = input_channels;
  output_channels_ = output_channels;
  kernel_height_ = kernel_height;
  kernel_width_ = kernel_width;
  stride_ = stride;
  padding_ = padding;

  weight_.assign(output_channels_,
                 Tensor3D(input_channels_,
                          Matrix(kernel_height_,
                                 std::vector<double>(kernel_width_, 0.0))));
  bias_.assign(output_channels_, 0.0);

  const double fan_in = static_cast<double>(input_channels_ * kernel_height_ *
                                            kernel_width_);
  const double fan_out = static_cast<double>(output_channels_ * kernel_height_ *
                                             kernel_width_);
  const double limit = std::sqrt(6.0 / (fan_in + fan_out));
  std::mt19937 gen(static_cast<std::mt19937::result_type>(rand_seed_));
  std::uniform_real_distribution<double> dist(-limit, limit);
  for (auto &out_channel : weight_) {
    for (auto &in_channel : out_channel) {
      for (auto &row : in_channel) {
        for (double &value : row) {
          value = dist(gen);
        }
      }
    }
  }

  is_init_ = true;
  return SUCCESS;
}

Conv2D::RC Conv2D::Forward(const Tensor3D &input, Tensor3D &output) {
  if (!is_init_) {
    err_msg_ = "[Conv2D::Forward] Conv2D not init";
    return NOT_INIT;
  }
  auto rc = ValidateTensor3D(input, input_channels_, "Forward");
  if (rc != SUCCESS) {
    return rc;
  }

  last_input_height_ = static_cast<int>(input[0].size());
  last_input_width_ = static_cast<int>(input[0][0].size());
  const int padded_height = last_input_height_ + 2 * padding_;
  const int padded_width = last_input_width_ + 2 * padding_;
  const int out_height =
      (padded_height - kernel_height_) / stride_ + 1;
  const int out_width = (padded_width - kernel_width_) / stride_ + 1;
  if (out_height <= 0 || out_width <= 0) {
    err_msg_ = "[Conv2D::Forward] Invalid spatial size";
    return INVALID_DATA;
  }

  last_input_padded_ = MakeTensor3D(input_channels_, padded_height, padded_width);
  for (int ic = 0; ic < input_channels_; ic++) {
    for (int row = 0; row < last_input_height_; row++) {
      for (int col = 0; col < last_input_width_; col++) {
        last_input_padded_[ic][row + padding_][col + padding_] =
            input[ic][row][col];
      }
    }
  }

  output = MakeTensor3D(output_channels_, out_height, out_width);
  for (int oc = 0; oc < output_channels_; oc++) {
    for (int out_row = 0; out_row < out_height; out_row++) {
      for (int out_col = 0; out_col < out_width; out_col++) {
        double sum = bias_[oc];
        const int base_row = out_row * stride_;
        const int base_col = out_col * stride_;
        for (int ic = 0; ic < input_channels_; ic++) {
          for (int kh = 0; kh < kernel_height_; kh++) {
            for (int kw = 0; kw < kernel_width_; kw++) {
              sum += weight_[oc][ic][kh][kw] *
                     last_input_padded_[ic][base_row + kh][base_col + kw];
            }
          }
        }
        output[oc][out_row][out_col] = sum;
      }
    }
  }

  has_forward_cache_ = true;
  return SUCCESS;
}

Conv2D::RC Conv2D::Backward(const Tensor3D &grad_output, Tensor3D &grad_input,
                            double learning_rate) {
  if (!is_init_) {
    err_msg_ = "[Conv2D::Backward] Conv2D not init";
    return NOT_INIT;
  }
  if (!has_forward_cache_) {
    err_msg_ = "[Conv2D::Backward] Missing forward cache";
    return INVALID_DATA;
  }
  if (learning_rate < 0) {
    err_msg_ = "[Conv2D::Backward] Invalid learning rate";
    return INVALID_DATA;
  }
  const int padded_height = static_cast<int>(last_input_padded_[0].size());
  const int padded_width = static_cast<int>(last_input_padded_[0][0].size());
  const int out_height = (padded_height - kernel_height_) / stride_ + 1;
  const int out_width = (padded_width - kernel_width_) / stride_ + 1;
  if (!Tensor3DHasShape(grad_output, output_channels_, out_height, out_width)) {
    err_msg_ = "[Conv2D::Backward] Invalid grad_output shape";
    return INVALID_DATA;
  }
  Tensor4D grad_weight;
  std::vector<double> grad_bias;
  auto rc = BackwardGradient(grad_output, grad_input, grad_weight, grad_bias);
  if (rc != SUCCESS) {
    return rc;
  }
  if (learning_rate > 0) {
    return ApplyGradient(grad_weight, grad_bias, learning_rate);
  }
  return SUCCESS;
}

Conv2D::RC Conv2D::BackwardGradient(const Tensor3D &grad_output,
                                    Tensor3D &grad_input,
                                    Tensor4D &grad_weight,
                                    std::vector<double> &grad_bias) {
  if (!is_init_) {
    err_msg_ = "[Conv2D::BackwardGradient] Conv2D not init";
    return NOT_INIT;
  }
  if (!has_forward_cache_) {
    err_msg_ = "[Conv2D::BackwardGradient] Missing forward cache";
    return INVALID_DATA;
  }

  const int padded_height = static_cast<int>(last_input_padded_[0].size());
  const int padded_width = static_cast<int>(last_input_padded_[0][0].size());
  const int out_height =
      (padded_height - kernel_height_) / stride_ + 1;
  const int out_width = (padded_width - kernel_width_) / stride_ + 1;
  if (!Tensor3DHasShape(grad_output, output_channels_, out_height, out_width)) {
    err_msg_ = "[Conv2D::BackwardGradient] Invalid grad_output shape";
    return INVALID_DATA;
  }

  grad_weight.assign(output_channels_,
                     Tensor3D(input_channels_,
                              Matrix(kernel_height_,
                                     std::vector<double>(kernel_width_, 0.0))));
  grad_bias.assign(output_channels_, 0.0);
  Tensor3D grad_input_padded =
      MakeTensor3D(input_channels_, padded_height, padded_width);

  for (int oc = 0; oc < output_channels_; oc++) {
    for (int out_row = 0; out_row < out_height; out_row++) {
      for (int out_col = 0; out_col < out_width; out_col++) {
        const double grad = grad_output[oc][out_row][out_col];
        grad_bias[oc] += grad;
        const int base_row = out_row * stride_;
        const int base_col = out_col * stride_;
        for (int ic = 0; ic < input_channels_; ic++) {
          for (int kh = 0; kh < kernel_height_; kh++) {
            for (int kw = 0; kw < kernel_width_; kw++) {
              grad_weight[oc][ic][kh][kw] +=
                  grad * last_input_padded_[ic][base_row + kh][base_col + kw];
              grad_input_padded[ic][base_row + kh][base_col + kw] +=
                  grad * weight_[oc][ic][kh][kw];
            }
          }
        }
      }
    }
  }

  grad_input = MakeTensor3D(input_channels_, last_input_height_, last_input_width_);
  for (int ic = 0; ic < input_channels_; ic++) {
    for (int row = 0; row < last_input_height_; row++) {
      for (int col = 0; col < last_input_width_; col++) {
        grad_input[ic][row][col] =
            grad_input_padded[ic][row + padding_][col + padding_];
      }
    }
  }

  return SUCCESS;
}

Conv2D::RC Conv2D::ApplyGradient(const Tensor4D &grad_weight,
                                 const std::vector<double> &grad_bias,
                                 double learning_rate,
                                 double gradient_scale) {
  if (!is_init_) {
    err_msg_ = "[Conv2D::ApplyGradient] Conv2D not init";
    return NOT_INIT;
  }
  if (learning_rate < 0) {
    err_msg_ = "[Conv2D::ApplyGradient] Invalid learning rate";
    return INVALID_DATA;
  }
  if (static_cast<int>(grad_bias.size()) != output_channels_ ||
      static_cast<int>(grad_weight.size()) != output_channels_) {
    err_msg_ = "[Conv2D::ApplyGradient] Invalid gradient shape";
    return INVALID_DATA;
  }
  for (const auto &out_channel : grad_weight) {
    if (static_cast<int>(out_channel.size()) != input_channels_) {
      err_msg_ = "[Conv2D::ApplyGradient] Invalid gradient shape";
      return INVALID_DATA;
    }
    for (const auto &in_channel : out_channel) {
      if (static_cast<int>(in_channel.size()) != kernel_height_) {
        err_msg_ = "[Conv2D::ApplyGradient] Invalid gradient shape";
        return INVALID_DATA;
      }
      for (const auto &row : in_channel) {
        if (static_cast<int>(row.size()) != kernel_width_) {
          err_msg_ = "[Conv2D::ApplyGradient] Invalid gradient shape";
          return INVALID_DATA;
        }
      }
    }
  }

  if (learning_rate == 0.0 || gradient_scale == 0.0) {
    return SUCCESS;
  }
  for (int oc = 0; oc < output_channels_; oc++) {
    bias_[oc] -= learning_rate * gradient_scale * grad_bias[oc];
    for (int ic = 0; ic < input_channels_; ic++) {
      for (int kh = 0; kh < kernel_height_; kh++) {
        for (int kw = 0; kw < kernel_width_; kw++) {
          weight_[oc][ic][kh][kw] -=
              learning_rate * gradient_scale * grad_weight[oc][ic][kh][kw];
        }
      }
    }
  }
  return SUCCESS;
}

void Conv2D::set_random_seed(int seed) { rand_seed_ = seed; }

Conv2D::RC Conv2D::set_weight(const Tensor4D &weight) {
  if (!is_init_) {
    err_msg_ = "[Conv2D::set_weight] Conv2D not init";
    return NOT_INIT;
  }
  if (static_cast<int>(weight.size()) != output_channels_) {
    err_msg_ = "[Conv2D::set_weight] Invalid weight shape";
    return INVALID_DATA;
  }
  for (const auto &out_channel : weight) {
    if (static_cast<int>(out_channel.size()) != input_channels_) {
      err_msg_ = "[Conv2D::set_weight] Invalid weight shape";
      return INVALID_DATA;
    }
    for (const auto &in_channel : out_channel) {
      if (static_cast<int>(in_channel.size()) != kernel_height_) {
        err_msg_ = "[Conv2D::set_weight] Invalid weight shape";
        return INVALID_DATA;
      }
      for (const auto &row : in_channel) {
        if (static_cast<int>(row.size()) != kernel_width_) {
          err_msg_ = "[Conv2D::set_weight] Invalid weight shape";
          return INVALID_DATA;
        }
      }
    }
  }
  weight_ = weight;
  return SUCCESS;
}

Conv2D::RC Conv2D::set_bias(const std::vector<double> &bias) {
  if (!is_init_) {
    err_msg_ = "[Conv2D::set_bias] Conv2D not init";
    return NOT_INIT;
  }
  if (static_cast<int>(bias.size()) != output_channels_) {
    err_msg_ = "[Conv2D::set_bias] Invalid bias size";
    return INVALID_DATA;
  }
  bias_ = bias;
  return SUCCESS;
}

std::string Conv2D::err_msg() { return err_msg_; }

int Conv2D::input_channels() const { return input_channels_; }

int Conv2D::output_channels() const { return output_channels_; }

int Conv2D::kernel_height() const { return kernel_height_; }

int Conv2D::kernel_width() const { return kernel_width_; }

int Conv2D::stride() const { return stride_; }

int Conv2D::padding() const { return padding_; }

const Conv2D::Tensor4D &Conv2D::weight() const { return weight_; }

const std::vector<double> &Conv2D::bias() const { return bias_; }

Conv2D::RC Conv2D::ValidateTensor3D(const Conv2D::Tensor3D &tensor,
                                    int channel_num, const char *func_name) {
  if (tensor.empty() || channel_num <= 0 ||
      static_cast<int>(tensor.size()) != channel_num) {
    err_msg_ = std::string("[Conv2D::") + func_name + "] Invalid tensor";
    return INVALID_DATA;
  }
  const int height = static_cast<int>(tensor[0].size());
  if (height <= 0) {
    err_msg_ = std::string("[Conv2D::") + func_name + "] Invalid tensor";
    return INVALID_DATA;
  }
  const int width = static_cast<int>(tensor[0][0].size());
  if (width <= 0) {
    err_msg_ = std::string("[Conv2D::") + func_name + "] Invalid tensor";
    return INVALID_DATA;
  }
  for (const auto &channel : tensor) {
    if (static_cast<int>(channel.size()) != height) {
      err_msg_ = std::string("[Conv2D::") + func_name + "] Invalid tensor";
      return INVALID_DATA;
    }
    for (const auto &row : channel) {
      if (static_cast<int>(row.size()) != width) {
        err_msg_ = std::string("[Conv2D::") + func_name + "] Invalid tensor";
        return INVALID_DATA;
      }
    }
  }
  return SUCCESS;
}

} // namespace deeplearning

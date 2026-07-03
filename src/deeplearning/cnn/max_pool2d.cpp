#include "cnn/max_pool2d.h"

#include <limits>

namespace deeplearning {
namespace {

using Matrix = MaxPool2D::Matrix;
using Tensor3D = MaxPool2D::Tensor3D;

bool ValidateTensor3D(const Tensor3D &tensor) {
  if (tensor.empty() || tensor[0].empty() || tensor[0][0].empty()) {
    return false;
  }
  const int height = static_cast<int>(tensor[0].size());
  const int width = static_cast<int>(tensor[0][0].size());
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

MaxPool2D::RC MaxPool2D::Init(int kernel_height, int kernel_width, int stride) {
  if (is_init_) {
    err_msg_ = "[MaxPool2D::Init] MaxPool2D has init";
    return ALREADY_INIT;
  }
  if (kernel_height <= 0 || kernel_width <= 0 || stride <= 0) {
    err_msg_ = "[MaxPool2D::Init] Invalid config";
    return INVALID_DATA;
  }
  kernel_height_ = kernel_height;
  kernel_width_ = kernel_width;
  stride_ = stride;
  is_init_ = true;
  return SUCCESS;
}

MaxPool2D::RC MaxPool2D::Forward(const Tensor3D &input, Tensor3D &output) {
  if (!is_init_) {
    err_msg_ = "[MaxPool2D::Forward] MaxPool2D not init";
    return NOT_INIT;
  }
  if (!ValidateTensor3D(input)) {
    err_msg_ = "[MaxPool2D::Forward] Invalid input tensor";
    return INVALID_DATA;
  }

  const int channel_num = static_cast<int>(input.size());
  last_input_height_ = static_cast<int>(input[0].size());
  last_input_width_ = static_cast<int>(input[0][0].size());
  const int out_height = (last_input_height_ - kernel_height_) / stride_ + 1;
  const int out_width = (last_input_width_ - kernel_width_) / stride_ + 1;
  if (out_height <= 0 || out_width <= 0) {
    err_msg_ = "[MaxPool2D::Forward] Invalid spatial size";
    return INVALID_DATA;
  }

  output = MakeTensor3D(channel_num, out_height, out_width);
  last_max_row_.assign(channel_num,
                       std::vector<std::vector<int>>(out_height,
                                                     std::vector<int>(out_width, 0)));
  last_max_col_.assign(channel_num,
                       std::vector<std::vector<int>>(out_height,
                                                     std::vector<int>(out_width, 0)));

  for (int channel = 0; channel < channel_num; channel++) {
    for (int out_row = 0; out_row < out_height; out_row++) {
      for (int out_col = 0; out_col < out_width; out_col++) {
        const int base_row = out_row * stride_;
        const int base_col = out_col * stride_;
        double best_value = -std::numeric_limits<double>::infinity();
        int best_row = base_row;
        int best_col = base_col;
        for (int kh = 0; kh < kernel_height_; kh++) {
          for (int kw = 0; kw < kernel_width_; kw++) {
            const int row = base_row + kh;
            const int col = base_col + kw;
            const double value = input[channel][row][col];
            if (value > best_value) {
              best_value = value;
              best_row = row;
              best_col = col;
            }
          }
        }
        output[channel][out_row][out_col] = best_value;
        last_max_row_[channel][out_row][out_col] = best_row;
        last_max_col_[channel][out_row][out_col] = best_col;
      }
    }
  }

  has_forward_cache_ = true;
  return SUCCESS;
}

MaxPool2D::RC MaxPool2D::Backward(const Tensor3D &grad_output,
                                  Tensor3D &grad_input) {
  if (!is_init_) {
    err_msg_ = "[MaxPool2D::Backward] MaxPool2D not init";
    return NOT_INIT;
  }
  if (!has_forward_cache_) {
    err_msg_ = "[MaxPool2D::Backward] Missing forward cache";
    return INVALID_DATA;
  }
  if (!ValidateTensor3D(grad_output)) {
    err_msg_ = "[MaxPool2D::Backward] Invalid grad_output tensor";
    return INVALID_DATA;
  }

  const int channel_num = static_cast<int>(last_max_row_.size());
  const int out_height = static_cast<int>(last_max_row_[0].size());
  const int out_width = static_cast<int>(last_max_row_[0][0].size());
  if (static_cast<int>(grad_output.size()) != channel_num ||
      static_cast<int>(grad_output[0].size()) != out_height ||
      static_cast<int>(grad_output[0][0].size()) != out_width) {
    err_msg_ = "[MaxPool2D::Backward] Invalid grad_output shape";
    return INVALID_DATA;
  }

  grad_input = MakeTensor3D(channel_num, last_input_height_, last_input_width_);
  for (int channel = 0; channel < channel_num; channel++) {
    for (int out_row = 0; out_row < out_height; out_row++) {
      for (int out_col = 0; out_col < out_width; out_col++) {
        const int row = last_max_row_[channel][out_row][out_col];
        const int col = last_max_col_[channel][out_row][out_col];
        grad_input[channel][row][col] += grad_output[channel][out_row][out_col];
      }
    }
  }
  return SUCCESS;
}

std::string MaxPool2D::err_msg() { return err_msg_; }

int MaxPool2D::kernel_height() const { return kernel_height_; }

int MaxPool2D::kernel_width() const { return kernel_width_; }

int MaxPool2D::stride() const { return stride_; }

} // namespace deeplearning

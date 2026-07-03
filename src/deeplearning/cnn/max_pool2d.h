#pragma once

#include <string>
#include <vector>

namespace deeplearning {

class MaxPool2D {
public:
  using Matrix = std::vector<std::vector<double>>;
  using Tensor3D = std::vector<Matrix>;

  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
  };

public:
  RC Init(int kernel_height, int kernel_width, int stride);
  RC Forward(const Tensor3D &input, Tensor3D &output);
  RC Backward(const Tensor3D &grad_output, Tensor3D &grad_input);

  std::string err_msg();
  int kernel_height() const;
  int kernel_width() const;
  int stride() const;

private:
  int kernel_height_ = 0;
  int kernel_width_ = 0;
  int stride_ = 0;
  int last_input_height_ = 0;
  int last_input_width_ = 0;
  std::vector<std::vector<std::vector<int>>> last_max_row_;
  std::vector<std::vector<std::vector<int>>> last_max_col_;
  std::string err_msg_;
  bool is_init_ = false;
  bool has_forward_cache_ = false;
};

} // namespace deeplearning

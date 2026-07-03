#pragma once

#include <string>
#include <vector>

namespace deeplearning {

class Conv2D {
public:
  using Matrix = std::vector<std::vector<double>>;
  using Tensor3D = std::vector<Matrix>;
  using Tensor4D = std::vector<Tensor3D>;

  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
  };

public:
  RC Init(int input_channels, int output_channels, int kernel_height,
          int kernel_width, int stride = 1, int padding = 0);
  RC Forward(const Tensor3D &input, Tensor3D &output);
  RC Backward(const Tensor3D &grad_output, Tensor3D &grad_input,
              double learning_rate);

  void set_random_seed(int seed);
  RC set_weight(const Tensor4D &weight);
  RC set_bias(const std::vector<double> &bias);

  std::string err_msg();
  int input_channels() const;
  int output_channels() const;
  int kernel_height() const;
  int kernel_width() const;
  int stride() const;
  int padding() const;
  const Tensor4D &weight() const;
  const std::vector<double> &bias() const;

private:
  RC ValidateTensor3D(const Tensor3D &tensor, int channel_num,
                      const char *func_name);

private:
  int input_channels_ = 0;
  int output_channels_ = 0;
  int kernel_height_ = 0;
  int kernel_width_ = 0;
  int stride_ = 1;
  int padding_ = 0;
  int rand_seed_ = 0;
  int last_input_height_ = 0;
  int last_input_width_ = 0;
  Tensor4D weight_;
  std::vector<double> bias_;
  Tensor3D last_input_padded_;
  std::string err_msg_;
  bool is_init_ = false;
  bool has_forward_cache_ = false;
};

} // namespace deeplearning

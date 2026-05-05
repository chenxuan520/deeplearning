#pragma once

#include <string>
#include <vector>

namespace deeplearning {

class LayerNorm {
public:
  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
  };

public:
  RC Init(int feature_dim, double epsilon = 1e-6);
  RC Forward(const std::vector<std::vector<double>> &input,
             std::vector<std::vector<double>> &output);
  RC Backward(const std::vector<std::vector<double>> &grad_output,
              std::vector<std::vector<double>> &grad_input,
              double learning_rate);

  RC set_scale(const std::vector<double> &scale);
  RC set_bias(const std::vector<double> &bias);

  std::string err_msg();
  const std::vector<double> &scale() const;
  const std::vector<double> &bias() const;

private:
  int feature_dim_ = 0;
  double epsilon_ = 1e-6;
  std::vector<double> scale_;
  std::vector<double> bias_;
  std::vector<std::vector<double>> last_input_;
  std::string err_msg_;
  bool is_init_ = false;
};

} // namespace deeplearning

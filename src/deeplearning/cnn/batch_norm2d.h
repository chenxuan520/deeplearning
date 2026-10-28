#pragma once

#include "cnn/float_tensor.h"

#include <string>
#include <vector>

namespace deeplearning {

class BatchNorm2D {
public:
  struct Config {
    int channels_ = 0;
    float epsilon_ = 1e-5f;
    float momentum_ = 0.1f;
    int thread_num_ = 1;
  };

  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
    MISSING_CACHE,
  };

  RC Init(const Config &config);
  RC Forward(const FloatTensor4D &input, FloatTensor4D &output,
             bool training, bool use_running_statistics = false);
  RC Backward(const FloatTensor4D &grad_output, FloatTensor4D &grad_input);
  void ZeroGrad();

  RC set_scale(const std::vector<float> &scale);
  RC set_bias(const std::vector<float> &bias);
  RC set_running_mean(const std::vector<float> &mean);
  RC set_running_variance(const std::vector<float> &variance);
  void InferenceAffine(std::vector<float> &scale,
                       std::vector<float> &bias) const;

  const Config &config() const { return config_; }
  const std::vector<float> &scale() const { return scale_; }
  const std::vector<float> &bias() const { return bias_; }
  const std::vector<float> &running_mean() const { return running_mean_; }
  const std::vector<float> &running_variance() const {
    return running_variance_;
  }
  const std::vector<float> &grad_scale() const { return grad_scale_; }
  const std::vector<float> &grad_bias() const { return grad_bias_; }
  std::vector<float> &mutable_scale() { return scale_; }
  std::vector<float> &mutable_bias() { return bias_; }
  std::string err_msg() const { return err_msg_; }

private:
  bool ValidateInput(const FloatTensor4D &input, const char *function);

  Config config_;
  std::vector<float> scale_;
  std::vector<float> bias_;
  std::vector<float> running_mean_;
  std::vector<float> running_variance_;
  std::vector<float> grad_scale_;
  std::vector<float> grad_bias_;

  FloatTensor4D normalized_cache_;
  std::vector<float> inverse_std_cache_;
  bool is_init_ = false;
  bool has_cache_ = false;
  bool fixed_statistics_cache_ = false;
  std::string err_msg_;
};

} // namespace deeplearning

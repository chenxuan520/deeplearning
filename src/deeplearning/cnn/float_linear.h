#pragma once

#include <string>
#include <vector>

namespace deeplearning {

class FloatLinear {
public:
  struct Config {
    int input_dim_ = 0;
    int output_dim_ = 0;
    bool use_bias_ = true;
    int thread_num_ = 1;
    int rand_seed_ = 0;
  };

  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
    MISSING_CACHE,
  };

  RC Init(const Config &config);
  RC Forward(const std::vector<float> &input, int batch,
             std::vector<float> &output, bool cache_for_backward = false);
  RC Backward(const std::vector<float> &grad_output,
              std::vector<float> &grad_input);
  void ZeroGrad();

  RC set_weight(const std::vector<float> &weight);
  RC set_bias(const std::vector<float> &bias);

  const Config &config() const { return config_; }
  const std::vector<float> &weight() const { return weight_; }
  const std::vector<float> &bias() const { return bias_; }
  std::vector<float> &mutable_weight() { return weight_; }
  std::vector<float> &mutable_bias() { return bias_; }
  const std::vector<float> &grad_weight() const { return grad_weight_; }
  const std::vector<float> &grad_bias() const { return grad_bias_; }
  std::string err_msg() const { return err_msg_; }

private:
  Config config_;
  std::vector<float> weight_;
  std::vector<float> bias_;
  std::vector<float> grad_weight_;
  std::vector<float> grad_bias_;
  std::vector<float> input_cache_;
  int cached_batch_ = 0;
  bool is_init_ = false;
  bool has_cache_ = false;
  std::string err_msg_;
};

} // namespace deeplearning

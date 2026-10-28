#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace deeplearning {

// AdamW optimizer for a fixed ordered list of contiguous float parameter
// tensors. The caller supplies parameter/gradient references in the same order
// on every Step. One global step is shared by all tensors.
class FloatAdamW {
public:
  struct Config {
    float beta1_ = 0.9f;
    float beta2_ = 0.999f;
    float epsilon_ = 1e-8f;
    float weight_decay_ = 1e-4f;
  };

  struct Parameter {
    std::vector<float> *value_ = nullptr;
    const std::vector<float> *gradient_ = nullptr;
    bool apply_weight_decay_ = true;
  };

  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
  };

  RC Init(const std::vector<std::size_t> &parameter_sizes);
  RC Init(const std::vector<std::size_t> &parameter_sizes,
          const Config &config);
  RC Step(const std::vector<Parameter> &parameters, float learning_rate,
          float gradient_scale = 1.0f);
  RC Save(const std::string &file) const;
  RC Load(const std::string &file);

  int step() const { return step_; }
  const Config &config() const { return config_; }
  const std::vector<std::vector<float>> &first_moment() const {
    return first_moment_;
  }
  const std::vector<std::vector<float>> &second_moment() const {
    return second_moment_;
  }
  std::string err_msg() const { return err_msg_; }

private:
  Config config_;
  std::vector<std::size_t> parameter_sizes_;
  std::vector<std::vector<float>> first_moment_;
  std::vector<std::vector<float>> second_moment_;
  int step_ = 0;
  bool is_init_ = false;
  std::string err_msg_;
};

} // namespace deeplearning

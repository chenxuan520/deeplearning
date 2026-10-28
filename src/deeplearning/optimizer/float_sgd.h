#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace deeplearning {

class FloatSGD {
public:
  struct Config {
    float weight_decay_ = 0.0f;
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
  static float GradientScaleForMaxUpdateNorm(
      double gradient_norm, float learning_rate, float max_update_norm);

  int step() const { return step_; }
  const Config &config() const { return config_; }
  const std::vector<std::size_t> &parameter_sizes() const {
    return parameter_sizes_;
  }
  std::string err_msg() const { return err_msg_; }

private:
  Config config_;
  std::vector<std::size_t> parameter_sizes_;
  int step_ = 0;
  bool is_init_ = false;
  std::string err_msg_;
};

} // namespace deeplearning

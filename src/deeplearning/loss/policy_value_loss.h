#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace deeplearning {

class PolicyValueLoss {
public:
  struct Config {
    int policy_size_ = 0;
    float policy_weight_ = 1.0f;
    float value_weight_ = 1.0f;
  };

  struct Result {
    float total_loss_ = 0.0f;
    float policy_loss_ = 0.0f;
    float value_loss_ = 0.0f;
    std::vector<float> grad_policy_logits_;
    std::vector<float> grad_values_;
  };

  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
  };

  RC Init(const Config &config);

  // legal_mask and policy_target use [batch, policy_size] row-major layout.
  // The target distribution must be nonnegative, zero on illegal actions, and
  // sum to one over legal actions for every sample.
  RC ForwardBackward(const std::vector<float> &policy_logits,
                     const std::vector<float> &values,
                     const std::vector<std::uint8_t> &legal_mask,
                     const std::vector<float> &policy_target,
                     const std::vector<float> &value_target, int batch,
                     Result &result);
  // value_mask is one for samples with a known terminal target and zero for
  // truncated/unknown outcomes. Value loss is normalized over known targets.
  RC ForwardBackward(const std::vector<float> &policy_logits,
                     const std::vector<float> &values,
                     const std::vector<std::uint8_t> &legal_mask,
                     const std::vector<float> &policy_target,
                     const std::vector<float> &value_target,
                     const std::vector<std::uint8_t> &value_mask, int batch,
                     Result &result);

  const Config &config() const { return config_; }
  std::string err_msg() const { return err_msg_; }

private:
  Config config_;
  bool is_init_ = false;
  std::string err_msg_;
};

} // namespace deeplearning

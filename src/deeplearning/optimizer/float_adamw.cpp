#include "optimizer/float_adamw.h"

#include "util/thread_pool.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <fstream>

namespace deeplearning {
namespace {

template <typename Value>
bool WriteValue(std::ofstream &output, const Value &value) {
  output.write(reinterpret_cast<const char *>(&value), sizeof(Value));
  return output.good();
}

template <typename Value>
bool ReadValue(std::ifstream &input, Value &value) {
  input.read(reinterpret_cast<char *>(&value), sizeof(Value));
  return input.good();
}

bool WriteVector(std::ofstream &output, const std::vector<float> &values) {
  const std::uint64_t size = values.size();
  if (!WriteValue(output, size)) {
    return false;
  }
  output.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(size * sizeof(float)));
  return output.good();
}

bool ReadVector(std::ifstream &input, std::size_t expected_size,
                std::vector<float> &values) {
  std::uint64_t size = 0;
  if (!ReadValue(input, size) || size != expected_size) {
    return false;
  }
  values.resize(static_cast<std::size_t>(size));
  input.read(reinterpret_cast<char *>(values.data()),
             static_cast<std::streamsize>(size * sizeof(float)));
  return input.good();
}

} // namespace

FloatAdamW::RC FloatAdamW::Init(
    const std::vector<std::size_t> &parameter_sizes) {
  return Init(parameter_sizes, Config());
}

FloatAdamW::RC FloatAdamW::Init(
    const std::vector<std::size_t> &parameter_sizes, const Config &config) {
  if (is_init_) {
    err_msg_ = "[FloatAdamW::Init] already initialized";
    return ALREADY_INIT;
  }
  if (parameter_sizes.empty() ||
      std::any_of(parameter_sizes.begin(), parameter_sizes.end(),
                  [](std::size_t size) { return size == 0; }) ||
      config.beta1_ < 0.0f || config.beta1_ >= 1.0f ||
      config.beta2_ < 0.0f || config.beta2_ >= 1.0f ||
      config.epsilon_ <= 0.0f || config.weight_decay_ < 0.0f) {
    err_msg_ = "[FloatAdamW::Init] invalid config";
    return INVALID_DATA;
  }
  config_ = config;
  parameter_sizes_ = parameter_sizes;
  first_moment_.resize(parameter_sizes.size());
  second_moment_.resize(parameter_sizes.size());
  for (std::size_t tensor = 0; tensor < parameter_sizes.size(); tensor++) {
    first_moment_[tensor].assign(parameter_sizes[tensor], 0.0f);
    second_moment_[tensor].assign(parameter_sizes[tensor], 0.0f);
  }
  is_init_ = true;
  return SUCCESS;
}

FloatAdamW::RC FloatAdamW::Step(
    const std::vector<Parameter> &parameters, float learning_rate,
    float gradient_scale) {
  if (!is_init_) {
    err_msg_ = "[FloatAdamW::Step] not initialized";
    return NOT_INIT;
  }
  if (parameters.size() != parameter_sizes_.size() ||
      learning_rate <= 0.0f || !std::isfinite(gradient_scale)) {
    err_msg_ = "[FloatAdamW::Step] invalid arguments";
    return INVALID_DATA;
  }
  for (std::size_t tensor = 0; tensor < parameters.size(); tensor++) {
    const Parameter &parameter = parameters[tensor];
    if (parameter.value_ == nullptr || parameter.gradient_ == nullptr ||
        parameter.value_->size() != parameter_sizes_[tensor] ||
        parameter.gradient_->size() != parameter_sizes_[tensor]) {
      err_msg_ = "[FloatAdamW::Step] parameter shape mismatch";
      return INVALID_DATA;
    }
  }

  step_++;
  const float correction1 = 1.0f - std::pow(config_.beta1_, step_);
  const float correction2 = 1.0f - std::pow(config_.beta2_, step_);
  ThreadPool::Global().Run(
      static_cast<int>(parameters.size()),
      static_cast<int>(parameters.size()),
      [&](int begin, int end) {
        for (int tensor = begin; tensor < end; tensor++) {
          const Parameter &parameter = parameters[tensor];
          std::vector<float> &value = *parameter.value_;
          const std::vector<float> &gradient = *parameter.gradient_;
          std::vector<float> &first = first_moment_[tensor];
          std::vector<float> &second = second_moment_[tensor];
          for (std::size_t index = 0; index < value.size(); index++) {
            const float grad = gradient[index] * gradient_scale;
            first[index] =
                config_.beta1_ * first[index] + (1.0f - config_.beta1_) * grad;
            second[index] = config_.beta2_ * second[index] +
                            (1.0f - config_.beta2_) * grad * grad;
            const float adaptive =
                (first[index] / correction1) /
                (std::sqrt(second[index] / correction2) + config_.epsilon_);
            const float decay =
                parameter.apply_weight_decay_ ? config_.weight_decay_ * value[index]
                                              : 0.0f;
            value[index] -= learning_rate * (adaptive + decay);
          }
        }
      });
  return SUCCESS;
}

FloatAdamW::RC FloatAdamW::Save(const std::string &file) const {
  if (!is_init_) {
    return NOT_INIT;
  }
  std::ofstream output(file, std::ios::binary | std::ios::trunc);
  const char magic[8] = {'F', 'A', 'D', 'A', 'M', 'W', '0', '1'};
  output.write(magic, sizeof(magic));
  const std::uint32_t version = 1;
  const std::uint64_t tensor_count = parameter_sizes_.size();
  if (!WriteValue(output, version) || !WriteValue(output, config_.beta1_) ||
      !WriteValue(output, config_.beta2_) ||
      !WriteValue(output, config_.epsilon_) ||
      !WriteValue(output, config_.weight_decay_) ||
      !WriteValue(output, step_) || !WriteValue(output, tensor_count)) {
    return INVALID_DATA;
  }
  for (std::size_t tensor = 0; tensor < parameter_sizes_.size(); tensor++) {
    const std::uint64_t size = parameter_sizes_[tensor];
    if (!WriteValue(output, size) ||
        !WriteVector(output, first_moment_[tensor]) ||
        !WriteVector(output, second_moment_[tensor])) {
      return INVALID_DATA;
    }
  }
  output.flush();
  return output.good() ? SUCCESS : INVALID_DATA;
}

FloatAdamW::RC FloatAdamW::Load(const std::string &file) {
  std::ifstream input(file, std::ios::binary);
  char magic[8] = {};
  input.read(magic, sizeof(magic));
  const char expected_magic[8] = {'F', 'A', 'D', 'A', 'M', 'W', '0', '1'};
  std::uint32_t version = 0;
  Config loaded_config;
  int loaded_step = 0;
  std::uint64_t tensor_count = 0;
  if (!input.good() ||
      !std::equal(std::begin(magic), std::end(magic),
                  std::begin(expected_magic)) ||
      !ReadValue(input, version) || version != 1 ||
      !ReadValue(input, loaded_config.beta1_) ||
      !ReadValue(input, loaded_config.beta2_) ||
      !ReadValue(input, loaded_config.epsilon_) ||
      !ReadValue(input, loaded_config.weight_decay_) ||
      !ReadValue(input, loaded_step) ||
      !ReadValue(input, tensor_count) || loaded_step < 0 ||
      tensor_count == 0) {
    err_msg_ = "[FloatAdamW::Load] invalid header";
    return INVALID_DATA;
  }
  std::vector<std::size_t> sizes(static_cast<std::size_t>(tensor_count));
  std::vector<std::vector<float>> first(sizes.size());
  std::vector<std::vector<float>> second(sizes.size());
  for (std::size_t tensor = 0; tensor < sizes.size(); tensor++) {
    std::uint64_t size = 0;
    if (!ReadValue(input, size) || size == 0) {
      err_msg_ = "[FloatAdamW::Load] invalid tensor size";
      return INVALID_DATA;
    }
    sizes[tensor] = static_cast<std::size_t>(size);
    if (!ReadVector(input, sizes[tensor], first[tensor]) ||
        !ReadVector(input, sizes[tensor], second[tensor])) {
      err_msg_ = "[FloatAdamW::Load] truncated moment tensor";
      return INVALID_DATA;
    }
  }
  char trailing = 0;
  if (input.read(&trailing, 1) || !input.eof()) {
    err_msg_ = "[FloatAdamW::Load] trailing data";
    return INVALID_DATA;
  }
  if (!is_init_) {
    if (Init(sizes, loaded_config) != SUCCESS) {
      return INVALID_DATA;
    }
  } else if (sizes != parameter_sizes_) {
    err_msg_ = "[FloatAdamW::Load] parameter structure mismatch";
    return INVALID_DATA;
  }
  config_ = loaded_config;
  step_ = loaded_step;
  first_moment_ = std::move(first);
  second_moment_ = std::move(second);
  return SUCCESS;
}

} // namespace deeplearning

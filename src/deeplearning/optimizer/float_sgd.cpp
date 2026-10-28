#include "optimizer/float_sgd.h"

#include "util/thread_pool.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>

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

} // namespace

FloatSGD::RC FloatSGD::Init(
    const std::vector<std::size_t> &parameter_sizes) {
  return Init(parameter_sizes, Config());
}

FloatSGD::RC FloatSGD::Init(
    const std::vector<std::size_t> &parameter_sizes, const Config &config) {
  if (is_init_) {
    err_msg_ = "[FloatSGD::Init] already initialized";
    return ALREADY_INIT;
  }
  if (parameter_sizes.empty() ||
      std::any_of(parameter_sizes.begin(), parameter_sizes.end(),
                  [](std::size_t size) { return size == 0; }) ||
      !std::isfinite(config.weight_decay_) || config.weight_decay_ < 0.0f) {
    err_msg_ = "[FloatSGD::Init] invalid config";
    return INVALID_DATA;
  }
  config_ = config;
  parameter_sizes_ = parameter_sizes;
  is_init_ = true;
  return SUCCESS;
}

FloatSGD::RC FloatSGD::Step(
    const std::vector<Parameter> &parameters, float learning_rate,
    float gradient_scale) {
  if (!is_init_) {
    err_msg_ = "[FloatSGD::Step] not initialized";
    return NOT_INIT;
  }
  if (parameters.size() != parameter_sizes_.size() ||
      !std::isfinite(learning_rate) || learning_rate <= 0.0f ||
      !std::isfinite(gradient_scale)) {
    err_msg_ = "[FloatSGD::Step] invalid arguments";
    return INVALID_DATA;
  }
  for (std::size_t tensor = 0; tensor < parameters.size(); tensor++) {
    const Parameter &parameter = parameters[tensor];
    if (parameter.value_ == nullptr || parameter.gradient_ == nullptr ||
        parameter.value_->size() != parameter_sizes_[tensor] ||
        parameter.gradient_->size() != parameter_sizes_[tensor]) {
      err_msg_ = "[FloatSGD::Step] parameter shape mismatch";
      return INVALID_DATA;
    }
  }

  step_++;
  ThreadPool::Global().Run(
      static_cast<int>(parameters.size()),
      static_cast<int>(parameters.size()),
      [&](int begin, int end) {
        for (int tensor = begin; tensor < end; tensor++) {
          const Parameter &parameter = parameters[tensor];
          std::vector<float> &value = *parameter.value_;
          const std::vector<float> &gradient = *parameter.gradient_;
          for (std::size_t index = 0; index < value.size(); index++) {
            const float decay =
                parameter.apply_weight_decay_
                    ? config_.weight_decay_ * value[index]
                    : 0.0f;
            value[index] -= learning_rate *
                            (gradient[index] * gradient_scale + decay);
          }
        }
      });
  return SUCCESS;
}

float FloatSGD::GradientScaleForMaxUpdateNorm(
    double gradient_norm, float learning_rate, float max_update_norm) {
  if (!std::isfinite(gradient_norm) || gradient_norm < 0.0 ||
      !std::isfinite(learning_rate) || learning_rate <= 0.0f ||
      !std::isfinite(max_update_norm) || max_update_norm < 0.0f) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (max_update_norm == 0.0f || gradient_norm == 0.0) {
    return 1.0f;
  }
  const double raw_update_norm = learning_rate * gradient_norm;
  return raw_update_norm > max_update_norm
             ? static_cast<float>(max_update_norm / raw_update_norm)
             : 1.0f;
}

FloatSGD::RC FloatSGD::Save(const std::string &file) const {
  if (!is_init_) return NOT_INIT;
  std::ofstream output(file, std::ios::binary | std::ios::trunc);
  const char magic[8] = {'F', 'S', 'G', 'D', '0', '0', '0', '1'};
  output.write(magic, sizeof(magic));
  const std::uint32_t version = 1;
  const std::uint64_t tensor_count = parameter_sizes_.size();
  if (!WriteValue(output, version) ||
      !WriteValue(output, config_.weight_decay_) ||
      !WriteValue(output, step_) ||
      !WriteValue(output, tensor_count)) {
    return INVALID_DATA;
  }
  for (std::size_t size : parameter_sizes_) {
    const std::uint64_t serialized_size = size;
    if (!WriteValue(output, serialized_size)) return INVALID_DATA;
  }
  output.flush();
  return output.good() ? SUCCESS : INVALID_DATA;
}

FloatSGD::RC FloatSGD::Load(const std::string &file) {
  std::ifstream input(file, std::ios::binary);
  char magic[8] = {};
  input.read(magic, sizeof(magic));
  const char expected_magic[8] =
      {'F', 'S', 'G', 'D', '0', '0', '0', '1'};
  std::uint32_t version = 0;
  Config loaded_config;
  int loaded_step = 0;
  std::uint64_t tensor_count = 0;
  if (!input.good() ||
      !std::equal(std::begin(magic), std::end(magic),
                  std::begin(expected_magic)) ||
      !ReadValue(input, version) || version != 1 ||
      !ReadValue(input, loaded_config.weight_decay_) ||
      !ReadValue(input, loaded_step) ||
      !ReadValue(input, tensor_count) || loaded_step < 0 ||
      tensor_count == 0) {
    err_msg_ = "[FloatSGD::Load] invalid header";
    return INVALID_DATA;
  }
  std::vector<std::size_t> sizes(static_cast<std::size_t>(tensor_count));
  for (std::size_t tensor = 0; tensor < sizes.size(); tensor++) {
    std::uint64_t size = 0;
    if (!ReadValue(input, size) || size == 0) {
      err_msg_ = "[FloatSGD::Load] invalid tensor size";
      return INVALID_DATA;
    }
    sizes[tensor] = static_cast<std::size_t>(size);
  }
  char trailing = 0;
  if (input.read(&trailing, 1) || !input.eof()) {
    err_msg_ = "[FloatSGD::Load] trailing data";
    return INVALID_DATA;
  }
  if (!is_init_) {
    if (Init(sizes, loaded_config) != SUCCESS) return INVALID_DATA;
  } else if (sizes != parameter_sizes_) {
    err_msg_ = "[FloatSGD::Load] parameter structure mismatch";
    return INVALID_DATA;
  }
  config_ = loaded_config;
  step_ = loaded_step;
  return SUCCESS;
}

} // namespace deeplearning

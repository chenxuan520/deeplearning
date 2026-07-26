#include "cnn/mini_cnn_classifier.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <system_error>
#include <thread>

namespace deeplearning {
namespace {

using Matrix = MiniCNNClassifier::Matrix;
using Tensor3D = MiniCNNClassifier::Tensor3D;

std::vector<double> Softmax(const std::vector<double> &logits) {
  std::vector<double> probs(logits.size(), 0.0);
  if (logits.empty()) {
    return probs;
  }
  double max_logit = logits[0];
  for (double value : logits) {
    if (value > max_logit) {
      max_logit = value;
    }
  }
  double sum = 0.0;
  for (int i = 0; i < static_cast<int>(logits.size()); i++) {
    probs[i] = std::exp(logits[i] - max_logit);
    sum += probs[i];
  }
  if (sum == 0.0) {
    return probs;
  }
  for (double &value : probs) {
    value /= sum;
  }
  return probs;
}

Tensor3D MakeTensor3D(int channel_num, int height, int width, double value = 0) {
  return Tensor3D(channel_num, Matrix(height, std::vector<double>(width, value)));
}

void ApplyRelu(Tensor3D &tensor) {
  for (auto &channel : tensor) {
    for (auto &row : channel) {
      for (double &value : row) {
        if (value < 0.0) {
          value = 0.0;
        }
      }
    }
  }
}

std::vector<double> FlattenTensor(const Tensor3D &tensor) {
  std::vector<double> flattened;
  for (const auto &channel : tensor) {
    for (const auto &row : channel) {
      for (double value : row) {
        flattened.push_back(value);
      }
    }
  }
  return flattened;
}

Tensor3D UnflattenTensor(const std::vector<double> &values, int channel_num,
                         int height, int width) {
  Tensor3D tensor = MakeTensor3D(channel_num, height, width);
  int pos = 0;
  for (int channel = 0; channel < channel_num; channel++) {
    for (int row = 0; row < height; row++) {
      for (int col = 0; col < width; col++) {
        tensor[channel][row][col] = values[pos++];
      }
    }
  }
  return tensor;
}

} // namespace

MiniCNNClassifier::RC MiniCNNClassifier::Init(const Config &config) {
  if (is_init_) {
    err_msg_ = "[MiniCNNClassifier::Init] MiniCNNClassifier has init";
    return ALREADY_INIT;
  }
  if (config.input_channels_ <= 0 || config.input_height_ <= 0 ||
      config.input_width_ <= 0 || config.conv_channels_ <= 0 ||
      config.kernel_height_ <= 0 || config.kernel_width_ <= 0 ||
      config.conv_stride_ <= 0 || config.conv_padding_ < 0 ||
      config.pool_height_ <= 0 || config.pool_width_ <= 0 ||
      config.pool_stride_ <= 0 || config.class_num_ <= 0) {
    err_msg_ = "[MiniCNNClassifier::Init] Invalid config";
    return INVALID_DATA;
  }

  config_ = config;
  conv_.set_random_seed(config.rand_seed_);
  auto conv_rc = conv_.Init(config.input_channels_, config.conv_channels_,
                            config.kernel_height_, config.kernel_width_,
                            config.conv_stride_, config.conv_padding_);
  if (conv_rc != Conv2D::SUCCESS) {
    err_msg_ = conv_.err_msg();
    return INVALID_DATA;
  }
  auto pool_rc =
      pool_.Init(config.pool_height_, config.pool_width_, config.pool_stride_);
  if (pool_rc != MaxPool2D::SUCCESS) {
    err_msg_ = pool_.err_msg();
    return INVALID_DATA;
  }

  const int conv_height =
      (config.input_height_ + 2 * config.conv_padding_ - config.kernel_height_) /
          config.conv_stride_ +
      1;
  const int conv_width =
      (config.input_width_ + 2 * config.conv_padding_ - config.kernel_width_) /
          config.conv_stride_ +
      1;
  const int pooled_height =
      (conv_height - config.pool_height_) / config.pool_stride_ + 1;
  const int pooled_width =
      (conv_width - config.pool_width_) / config.pool_stride_ + 1;
  if (conv_height <= 0 || conv_width <= 0 || pooled_height <= 0 ||
      pooled_width <= 0) {
    err_msg_ = "[MiniCNNClassifier::Init] Invalid spatial size";
    return INVALID_DATA;
  }

  flattened_dim_ = config.conv_channels_ * pooled_height * pooled_width;
  fc_weight_.assign(config.class_num_, std::vector<double>(flattened_dim_, 0.0));
  fc_bias_.assign(config.class_num_, 0.0);

  const double limit =
      std::sqrt(6.0 / (static_cast<double>(flattened_dim_ + config.class_num_)));
  std::mt19937 gen(static_cast<std::mt19937::result_type>(config.rand_seed_ + 17));
  std::uniform_real_distribution<double> dist(-limit, limit);
  for (auto &row : fc_weight_) {
    for (double &value : row) {
      value = dist(gen);
    }
  }

  is_init_ = true;
  return SUCCESS;
}

MiniCNNClassifier::RC MiniCNNClassifier::Forward(const Tensor3D &input,
                                                 std::vector<double> &logits) {
  if (!is_init_) {
    err_msg_ = "[MiniCNNClassifier::Forward] MiniCNNClassifier not init";
    return NOT_INIT;
  }
  Tensor3D conv_output, relu_output, pooled_output;
  std::vector<double> flattened;
  auto rc = ForwardFeature(input, conv_output, relu_output, pooled_output,
                           flattened);
  if (rc != SUCCESS) {
    return rc;
  }

  logits.assign(config_.class_num_, 0.0);
  for (int cls = 0; cls < config_.class_num_; cls++) {
    logits[cls] = fc_bias_[cls];
    for (int dim = 0; dim < flattened_dim_; dim++) {
      logits[cls] += fc_weight_[cls][dim] * flattened[dim];
    }
  }
  return SUCCESS;
}

MiniCNNClassifier::RC
MiniCNNClassifier::PredictProbs(const Tensor3D &input, std::vector<double> &probs) {
  std::vector<double> logits;
  auto rc = Forward(input, logits);
  if (rc != SUCCESS) {
    return rc;
  }
  probs = Softmax(logits);
  return SUCCESS;
}

MiniCNNClassifier::RC MiniCNNClassifier::Predict(const Tensor3D &input,
                                                 int &label) {
  std::vector<double> probs;
  auto rc = PredictProbs(input, probs);
  if (rc != SUCCESS) {
    return rc;
  }
  label = 0;
  for (int i = 1; i < static_cast<int>(probs.size()); i++) {
    if (probs[i] > probs[label]) {
      label = i;
    }
  }
  return SUCCESS;
}

MiniCNNClassifier::RC MiniCNNClassifier::Train(
    const std::vector<Tensor3D> &images, const std::vector<int> &labels,
    std::function<void(int epoch_num, double average_loss, bool &early_stop)>
        each_epoch_call,
    int epoch_num, double learning_rate) {
  if (!is_init_) {
    err_msg_ = "[MiniCNNClassifier::Train] MiniCNNClassifier not init";
    return NOT_INIT;
  }
  if (images.empty() || images.size() != labels.size() || epoch_num <= 0 ||
      learning_rate <= 0.0) {
    err_msg_ = "[MiniCNNClassifier::Train] Invalid training input";
    return INVALID_DATA;
  }

  std::vector<int> order(images.size(), 0);
  for (int i = 0; i < static_cast<int>(images.size()); i++) {
    order[i] = i;
  }
  std::mt19937 gen(static_cast<std::mt19937::result_type>(config_.rand_seed_));

  for (int epoch = 0; epoch < epoch_num; epoch++) {
    std::shuffle(order.begin(), order.end(), gen);
    double loss_sum = 0.0;
    for (int pos = 0; pos < static_cast<int>(order.size()); pos++) {
      const int sample_idx = order[pos];
      const int label = labels[sample_idx];
      if (label < 0 || label >= config_.class_num_) {
        err_msg_ = "[MiniCNNClassifier::Train] Invalid label";
        return INVALID_DATA;
      }

      Tensor3D conv_output, relu_output, pooled_output;
      std::vector<double> flattened;
      auto feature_rc = ForwardFeature(images[sample_idx], conv_output,
                                       relu_output, pooled_output, flattened);
      if (feature_rc != SUCCESS) {
        return feature_rc;
      }

      std::vector<double> logits(config_.class_num_, 0.0);
      for (int cls = 0; cls < config_.class_num_; cls++) {
        logits[cls] = fc_bias_[cls];
        for (int dim = 0; dim < flattened_dim_; dim++) {
          logits[cls] += fc_weight_[cls][dim] * flattened[dim];
        }
      }
      auto probs = Softmax(logits);
      loss_sum += -std::log(std::max(probs[label], 1e-12));

      std::vector<double> grad_logits = probs;
      grad_logits[label] -= 1.0;

      Matrix grad_fc_weight(config_.class_num_,
                            std::vector<double>(flattened_dim_, 0.0));
      std::vector<double> grad_fc_bias(config_.class_num_, 0.0);
      std::vector<double> grad_flat(flattened_dim_, 0.0);
      for (int cls = 0; cls < config_.class_num_; cls++) {
        const double grad = grad_logits[cls];
        grad_fc_bias[cls] = grad;
        for (int dim = 0; dim < flattened_dim_; dim++) {
          grad_fc_weight[cls][dim] = grad * flattened[dim];
          grad_flat[dim] += fc_weight_[cls][dim] * grad;
        }
      }

      const int pooled_channels = static_cast<int>(pooled_output.size());
      const int pooled_height = static_cast<int>(pooled_output[0].size());
      const int pooled_width = static_cast<int>(pooled_output[0][0].size());
      Tensor3D grad_pooled =
          UnflattenTensor(grad_flat, pooled_channels, pooled_height, pooled_width);
      Tensor3D grad_relu;
      auto pool_backward_rc = pool_.Backward(grad_pooled, grad_relu);
      if (pool_backward_rc != MaxPool2D::SUCCESS) {
        err_msg_ = pool_.err_msg();
        return INVALID_DATA;
      }

      Tensor3D grad_conv = grad_relu;
      for (int channel = 0; channel < static_cast<int>(grad_conv.size()); channel++) {
        for (int row = 0; row < static_cast<int>(grad_conv[channel].size()); row++) {
          for (int col = 0; col < static_cast<int>(grad_conv[channel][row].size());
               col++) {
            if (conv_output[channel][row][col] <= 0.0) {
              grad_conv[channel][row][col] = 0.0;
            }
          }
        }
      }

      Tensor3D grad_input;
      auto conv_backward_rc = conv_.Backward(grad_conv, grad_input, learning_rate);
      if (conv_backward_rc != Conv2D::SUCCESS) {
        err_msg_ = conv_.err_msg();
        return INVALID_DATA;
      }

      for (int cls = 0; cls < config_.class_num_; cls++) {
        fc_bias_[cls] -= learning_rate * grad_fc_bias[cls];
        for (int dim = 0; dim < flattened_dim_; dim++) {
          fc_weight_[cls][dim] -= learning_rate * grad_fc_weight[cls][dim];
        }
      }
    }

    bool early_stop = false;
    if (each_epoch_call != nullptr) {
      each_epoch_call(epoch, loss_sum / images.size(), early_stop);
    }
    if (early_stop) {
      break;
    }
  }
  return SUCCESS;
}

MiniCNNClassifier::RC MiniCNNClassifier::TrainBatch(
    const std::vector<Tensor3D> &images, const std::vector<int> &labels,
    int batch_size,
    std::function<void(int epoch_num, double average_loss, bool &early_stop)>
        each_epoch_call,
    int epoch_num, double learning_rate) {
  if (!is_init_) {
    err_msg_ = "[MiniCNNClassifier::TrainBatch] MiniCNNClassifier not init";
    return NOT_INIT;
  }
  if (images.empty() || images.size() != labels.size() || epoch_num <= 0 ||
      learning_rate <= 0.0 || batch_size <= 0) {
    err_msg_ = "[MiniCNNClassifier::TrainBatch] Invalid training input";
    return INVALID_DATA;
  }
  if (batch_size == 1) {
    return Train(images, labels, each_epoch_call, epoch_num, learning_rate);
  }

  std::vector<int> order(images.size(), 0);
  for (int i = 0; i < static_cast<int>(images.size()); i++) {
    order[i] = i;
  }
  std::mt19937 gen(static_cast<std::mt19937::result_type>(config_.rand_seed_));

  for (int epoch = 0; epoch < epoch_num; epoch++) {
    std::shuffle(order.begin(), order.end(), gen);
    double loss_sum = 0.0;
    for (int batch_begin = 0; batch_begin < static_cast<int>(order.size());
         batch_begin += batch_size) {
      const int batch_end =
          std::min(batch_begin + batch_size, static_cast<int>(order.size()));
      const int batch_sample_num = batch_end - batch_begin;
      GradientBuffer gradient;
      InitGradientBuffer(gradient);
      double batch_loss_sum = 0.0;

      RC rc = SUCCESS;
      if (train_thread_num_ > 1 && batch_sample_num > 1) {
        rc = AccumulateGradientsBatchParallel(images, labels, order, batch_begin,
                                              batch_end, gradient,
                                              batch_loss_sum);
      } else {
        std::string err_msg;
        rc = AccumulateGradientsRange(images, labels, order, batch_begin,
                                      batch_end, gradient, batch_loss_sum,
                                      err_msg);
        if (rc != SUCCESS) {
          err_msg_ = err_msg;
        }
      }
      if (rc != SUCCESS) {
        return rc;
      }

      rc = ApplyGradientBuffer(gradient, learning_rate,
                               1.0 / static_cast<double>(batch_sample_num));
      if (rc != SUCCESS) {
        return rc;
      }
      loss_sum += batch_loss_sum;
    }

    bool early_stop = false;
    if (each_epoch_call != nullptr) {
      each_epoch_call(epoch, loss_sum / images.size(), early_stop);
    }
    if (early_stop) {
      break;
    }
  }
  return SUCCESS;
}

void MiniCNNClassifier::set_random_seed(int seed) { config_.rand_seed_ = seed; }

void MiniCNNClassifier::set_train_thread_num(int thread_num) {
  train_thread_num_ = std::max(1, thread_num);
}

MiniCNNClassifier::RC MiniCNNClassifier::set_fc_weight(const Matrix &weight) {
  if (!is_init_) {
    err_msg_ = "[MiniCNNClassifier::set_fc_weight] MiniCNNClassifier not init";
    return NOT_INIT;
  }
  if (static_cast<int>(weight.size()) != config_.class_num_) {
    err_msg_ = "[MiniCNNClassifier::set_fc_weight] Invalid weight size";
    return INVALID_DATA;
  }
  for (const auto &row : weight) {
    if (static_cast<int>(row.size()) != flattened_dim_) {
      err_msg_ = "[MiniCNNClassifier::set_fc_weight] Invalid weight size";
      return INVALID_DATA;
    }
  }
  fc_weight_ = weight;
  return SUCCESS;
}

MiniCNNClassifier::RC
MiniCNNClassifier::set_fc_bias(const std::vector<double> &bias) {
  if (!is_init_) {
    err_msg_ = "[MiniCNNClassifier::set_fc_bias] MiniCNNClassifier not init";
    return NOT_INIT;
  }
  if (static_cast<int>(bias.size()) != config_.class_num_) {
    err_msg_ = "[MiniCNNClassifier::set_fc_bias] Invalid bias size";
    return INVALID_DATA;
  }
  fc_bias_ = bias;
  return SUCCESS;
}

std::string MiniCNNClassifier::err_msg() { return err_msg_; }

MiniCNNClassifier::Config MiniCNNClassifier::config() const { return config_; }

int MiniCNNClassifier::flattened_dim() const { return flattened_dim_; }

Conv2D &MiniCNNClassifier::conv() { return conv_; }

MaxPool2D &MiniCNNClassifier::pool() { return pool_; }

const MiniCNNClassifier::Matrix &MiniCNNClassifier::fc_weight() const {
  return fc_weight_;
}

const std::vector<double> &MiniCNNClassifier::fc_bias() const { return fc_bias_; }

MiniCNNClassifier::RC MiniCNNClassifier::ValidateInput(
    const MiniCNNClassifier::Tensor3D &input, const char *func_name) {
  if (!InputHasValidShape(input)) {
    err_msg_ = std::string("[MiniCNNClassifier::") + func_name +
               "] Invalid input";
    return INVALID_DATA;
  }
  return SUCCESS;
}

bool MiniCNNClassifier::InputHasValidShape(
    const MiniCNNClassifier::Tensor3D &input) const {
  if (static_cast<int>(input.size()) != config_.input_channels_ || input.empty() ||
      input[0].empty() || input[0][0].empty()) {
    return false;
  }
  if (static_cast<int>(input[0].size()) != config_.input_height_ ||
      static_cast<int>(input[0][0].size()) != config_.input_width_) {
    return false;
  }
  for (const auto &channel : input) {
    if (static_cast<int>(channel.size()) != config_.input_height_) {
      return false;
    }
    for (const auto &row : channel) {
      if (static_cast<int>(row.size()) != config_.input_width_) {
        return false;
      }
    }
  }
  return true;
}

MiniCNNClassifier::RC MiniCNNClassifier::ForwardFeature(
    const Tensor3D &input, Tensor3D &conv_output, Tensor3D &relu_output,
    Tensor3D &pooled_output, std::vector<double> &flattened) {
  auto validate_rc = ValidateInput(input, "ForwardFeature");
  if (validate_rc != SUCCESS) {
    return validate_rc;
  }
  auto conv_rc = conv_.Forward(input, conv_output);
  if (conv_rc != Conv2D::SUCCESS) {
    err_msg_ = conv_.err_msg();
    return INVALID_DATA;
  }
  relu_output = conv_output;
  ApplyRelu(relu_output);
  auto pool_rc = pool_.Forward(relu_output, pooled_output);
  if (pool_rc != MaxPool2D::SUCCESS) {
    err_msg_ = pool_.err_msg();
    return INVALID_DATA;
  }
  flattened = FlattenTensor(pooled_output);
  if (static_cast<int>(flattened.size()) != flattened_dim_) {
    err_msg_ = "[MiniCNNClassifier::ForwardFeature] Invalid flatten result";
    return INVALID_DATA;
  }
  return SUCCESS;
}

MiniCNNClassifier::RC MiniCNNClassifier::ForwardFeatureWithLayers(
    const Tensor3D &input, Conv2D &conv, MaxPool2D &pool,
    Tensor3D &conv_output, Tensor3D &relu_output, Tensor3D &pooled_output,
    std::vector<double> &flattened, std::string &err_msg) const {
  if (!InputHasValidShape(input)) {
    err_msg = "[MiniCNNClassifier::ForwardFeatureWithLayers] Invalid input";
    return INVALID_DATA;
  }
  auto conv_rc = conv.Forward(input, conv_output);
  if (conv_rc != Conv2D::SUCCESS) {
    err_msg = conv.err_msg();
    return INVALID_DATA;
  }
  relu_output = conv_output;
  ApplyRelu(relu_output);
  auto pool_rc = pool.Forward(relu_output, pooled_output);
  if (pool_rc != MaxPool2D::SUCCESS) {
    err_msg = pool.err_msg();
    return INVALID_DATA;
  }
  flattened = FlattenTensor(pooled_output);
  if (static_cast<int>(flattened.size()) != flattened_dim_) {
    err_msg = "[MiniCNNClassifier::ForwardFeatureWithLayers] Invalid flatten result";
    return INVALID_DATA;
  }
  return SUCCESS;
}

void MiniCNNClassifier::InitGradientBuffer(GradientBuffer &gradient) const {
  gradient.fc_weight.assign(config_.class_num_,
                            std::vector<double>(flattened_dim_, 0.0));
  gradient.fc_bias.assign(config_.class_num_, 0.0);
  gradient.conv_weight.assign(
      config_.conv_channels_,
      Tensor3D(config_.input_channels_,
               Matrix(config_.kernel_height_,
                      std::vector<double>(config_.kernel_width_, 0.0))));
  gradient.conv_bias.assign(config_.conv_channels_, 0.0);
}

void MiniCNNClassifier::AddGradientBuffer(GradientBuffer &dst,
                                          const GradientBuffer &src) const {
  for (int cls = 0; cls < config_.class_num_; cls++) {
    dst.fc_bias[cls] += src.fc_bias[cls];
    for (int dim = 0; dim < flattened_dim_; dim++) {
      dst.fc_weight[cls][dim] += src.fc_weight[cls][dim];
    }
  }
  for (int oc = 0; oc < config_.conv_channels_; oc++) {
    dst.conv_bias[oc] += src.conv_bias[oc];
    for (int ic = 0; ic < config_.input_channels_; ic++) {
      for (int kh = 0; kh < config_.kernel_height_; kh++) {
        for (int kw = 0; kw < config_.kernel_width_; kw++) {
          dst.conv_weight[oc][ic][kh][kw] += src.conv_weight[oc][ic][kh][kw];
        }
      }
    }
  }
}

MiniCNNClassifier::RC MiniCNNClassifier::ApplyGradientBuffer(
    const GradientBuffer &gradient, double learning_rate, double gradient_scale) {
  auto conv_rc = conv_.ApplyGradient(gradient.conv_weight, gradient.conv_bias,
                                     learning_rate, gradient_scale);
  if (conv_rc != Conv2D::SUCCESS) {
    err_msg_ = conv_.err_msg();
    return INVALID_DATA;
  }
  for (int cls = 0; cls < config_.class_num_; cls++) {
    fc_bias_[cls] -= learning_rate * gradient_scale * gradient.fc_bias[cls];
    for (int dim = 0; dim < flattened_dim_; dim++) {
      fc_weight_[cls][dim] -=
          learning_rate * gradient_scale * gradient.fc_weight[cls][dim];
    }
  }
  return SUCCESS;
}

MiniCNNClassifier::RC MiniCNNClassifier::AccumulateGradientsRange(
    const std::vector<Tensor3D> &images, const std::vector<int> &labels,
    const std::vector<int> &order, int begin, int end,
    GradientBuffer &gradient, double &loss_sum, std::string &err_msg) const {
  Conv2D worker_conv = conv_;
  MaxPool2D worker_pool = pool_;
  loss_sum = 0.0;

  for (int pos = begin; pos < end; pos++) {
    const int sample_idx = order[pos];
    const int label = labels[sample_idx];
    if (label < 0 || label >= config_.class_num_) {
      err_msg = "[MiniCNNClassifier::AccumulateGradientsRange] Invalid label";
      return INVALID_DATA;
    }

    Tensor3D conv_output, relu_output, pooled_output;
    std::vector<double> flattened;
    auto feature_rc = ForwardFeatureWithLayers(
        images[sample_idx], worker_conv, worker_pool, conv_output, relu_output,
        pooled_output, flattened, err_msg);
    if (feature_rc != SUCCESS) {
      return feature_rc;
    }

    std::vector<double> logits(config_.class_num_, 0.0);
    for (int cls = 0; cls < config_.class_num_; cls++) {
      logits[cls] = fc_bias_[cls];
      for (int dim = 0; dim < flattened_dim_; dim++) {
        logits[cls] += fc_weight_[cls][dim] * flattened[dim];
      }
    }
    auto probs = Softmax(logits);
    loss_sum += -std::log(std::max(probs[label], 1e-12));

    std::vector<double> grad_logits = probs;
    grad_logits[label] -= 1.0;

    std::vector<double> grad_flat(flattened_dim_, 0.0);
    for (int cls = 0; cls < config_.class_num_; cls++) {
      const double grad = grad_logits[cls];
      gradient.fc_bias[cls] += grad;
      for (int dim = 0; dim < flattened_dim_; dim++) {
        gradient.fc_weight[cls][dim] += grad * flattened[dim];
        grad_flat[dim] += fc_weight_[cls][dim] * grad;
      }
    }

    const int pooled_channels = static_cast<int>(pooled_output.size());
    const int pooled_height = static_cast<int>(pooled_output[0].size());
    const int pooled_width = static_cast<int>(pooled_output[0][0].size());
    Tensor3D grad_pooled =
        UnflattenTensor(grad_flat, pooled_channels, pooled_height, pooled_width);
    Tensor3D grad_relu;
    auto pool_backward_rc = worker_pool.Backward(grad_pooled, grad_relu);
    if (pool_backward_rc != MaxPool2D::SUCCESS) {
      err_msg = worker_pool.err_msg();
      return INVALID_DATA;
    }

    Tensor3D grad_conv = grad_relu;
    for (int channel = 0; channel < static_cast<int>(grad_conv.size());
         channel++) {
      for (int row = 0; row < static_cast<int>(grad_conv[channel].size());
           row++) {
        for (int col = 0; col < static_cast<int>(grad_conv[channel][row].size());
             col++) {
          if (conv_output[channel][row][col] <= 0.0) {
            grad_conv[channel][row][col] = 0.0;
          }
        }
      }
    }

    Tensor3D grad_input;
    Conv2D::Tensor4D grad_conv_weight;
    std::vector<double> grad_conv_bias;
    auto conv_backward_rc = worker_conv.BackwardGradient(
        grad_conv, grad_input, grad_conv_weight, grad_conv_bias);
    if (conv_backward_rc != Conv2D::SUCCESS) {
      err_msg = worker_conv.err_msg();
      return INVALID_DATA;
    }
    for (int oc = 0; oc < config_.conv_channels_; oc++) {
      gradient.conv_bias[oc] += grad_conv_bias[oc];
      for (int ic = 0; ic < config_.input_channels_; ic++) {
        for (int kh = 0; kh < config_.kernel_height_; kh++) {
          for (int kw = 0; kw < config_.kernel_width_; kw++) {
            gradient.conv_weight[oc][ic][kh][kw] +=
                grad_conv_weight[oc][ic][kh][kw];
          }
        }
      }
    }
  }
  return SUCCESS;
}

MiniCNNClassifier::RC MiniCNNClassifier::AccumulateGradientsBatchParallel(
    const std::vector<Tensor3D> &images, const std::vector<int> &labels,
    const std::vector<int> &order, int begin, int end,
    GradientBuffer &gradient, double &loss_sum) {
  const int batch_sample_num = end - begin;
  int worker_num = std::min(train_thread_num_, batch_sample_num);
  int hardware_thread_num =
      static_cast<int>(std::thread::hardware_concurrency());
  if (hardware_thread_num > 0) {
    worker_num = std::min(worker_num, hardware_thread_num);
  }
  if (worker_num <= 1) {
    std::string err_msg;
    auto rc = AccumulateGradientsRange(images, labels, order, begin, end,
                                       gradient, loss_sum, err_msg);
    if (rc != SUCCESS) {
      err_msg_ = err_msg;
    }
    return rc;
  }

  std::vector<GradientBuffer> worker_grad(worker_num);
  for (auto &grad : worker_grad) {
    InitGradientBuffer(grad);
  }
  std::vector<double> worker_loss_sum(worker_num, 0.0);
  std::vector<RC> worker_rc(worker_num, SUCCESS);
  std::vector<std::string> worker_err(worker_num);
  std::vector<std::thread> threads;
  threads.reserve(worker_num);

  try {
    for (int worker_idx = 0; worker_idx < worker_num; worker_idx++) {
      int worker_begin = begin + batch_sample_num * worker_idx / worker_num;
      int worker_end = begin + batch_sample_num * (worker_idx + 1) / worker_num;
      threads.emplace_back([&, worker_idx, worker_begin, worker_end]() {
        worker_rc[worker_idx] = AccumulateGradientsRange(
            images, labels, order, worker_begin, worker_end,
            worker_grad[worker_idx], worker_loss_sum[worker_idx],
            worker_err[worker_idx]);
      });
    }
  } catch (const std::system_error &e) {
    for (auto &thread : threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    err_msg_ =
        std::string("[MiniCNNClassifier::AccumulateGradientsBatchParallel] ") +
        "failed to create worker thread: " + e.what();
    return INVALID_DATA;
  }

  for (auto &thread : threads) {
    thread.join();
  }

  loss_sum = 0.0;
  for (int worker_idx = 0; worker_idx < worker_num; worker_idx++) {
    if (worker_rc[worker_idx] != SUCCESS) {
      err_msg_ = worker_err[worker_idx];
      return worker_rc[worker_idx];
    }
    loss_sum += worker_loss_sum[worker_idx];
    AddGradientBuffer(gradient, worker_grad[worker_idx]);
  }
  return SUCCESS;
}

} // namespace deeplearning

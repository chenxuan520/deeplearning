#include "cnn/mini_cnn_classifier.h"

#include <algorithm>
#include <cmath>
#include <random>
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
  if (config.conv2_channels_ > 0) {
    conv2_.set_random_seed(config.rand_seed_ + 101);
    auto conv2_rc = conv2_.Init(config.conv_channels_, config.conv2_channels_,
                                config.kernel_height_, config.kernel_width_,
                                config.conv_stride_, config.conv_padding_);
    if (conv2_rc != Conv2D::SUCCESS) {
      err_msg_ = conv2_.err_msg();
      return INVALID_DATA;
    }
  }

  const int conv_height =
      (config.input_height_ + 2 * config.conv_padding_ - config.kernel_height_) /
          config.conv_stride_ +
      1;
  const int conv_width =
      (config.input_width_ + 2 * config.conv_padding_ - config.kernel_width_) /
          config.conv_stride_ +
      1;
  const int conv2_height =
      config.conv2_channels_ > 0
          ? (conv_height + 2 * config.conv_padding_ - config.kernel_height_) /
                    config.conv_stride_ +
                1
          : conv_height;
  const int conv2_width =
      config.conv2_channels_ > 0
          ? (conv_width + 2 * config.conv_padding_ - config.kernel_width_) /
                    config.conv_stride_ +
                1
          : conv_width;
  const int pooled_height =
      (conv2_height - config.pool_height_) / config.pool_stride_ + 1;
  const int pooled_width =
      (conv2_width - config.pool_width_) / config.pool_stride_ + 1;
  if (conv_height <= 0 || conv_width <= 0 || pooled_height <= 0 ||
      pooled_width <= 0) {
    err_msg_ = "[MiniCNNClassifier::Init] Invalid spatial size";
    return INVALID_DATA;
  }

  flattened_dim_ = (config.conv2_channels_ > 0 ? config.conv2_channels_
                                               : config.conv_channels_) *
                   pooled_height * pooled_width;
  int fc_input_dim =
      config.hidden_dim_ > 0 ? config.hidden_dim_ : flattened_dim_;
  hidden_weight_.clear();
  hidden_bias_.clear();
  if (config.hidden_dim_ > 0) {
    hidden_weight_.assign(config.hidden_dim_,
                          std::vector<double>(flattened_dim_, 0.0));
    hidden_bias_.assign(config.hidden_dim_, 0.0);
  }
  fc_weight_.assign(config.class_num_, std::vector<double>(fc_input_dim, 0.0));
  fc_bias_.assign(config.class_num_, 0.0);

  const double limit =
      std::sqrt(6.0 / (static_cast<double>(fc_input_dim + config.class_num_)));
  std::mt19937 gen(static_cast<std::mt19937::result_type>(config.rand_seed_ + 17));
  std::uniform_real_distribution<double> dist(-limit, limit);
  for (auto &row : hidden_weight_) {
    for (double &value : row) value = dist(gen);
  }
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

  std::vector<double> head_input = flattened;
  if (config_.hidden_dim_ > 0) {
    head_input.assign(config_.hidden_dim_, 0.0);
    for (int h = 0; h < config_.hidden_dim_; ++h) {
      double v = hidden_bias_[h];
      for (int dim = 0; dim < flattened_dim_; ++dim) {
        v += hidden_weight_[h][dim] * flattened[dim];
      }
      head_input[h] = v > 0.0 ? v : 0.0;
    }
  }

  logits.assign(config_.class_num_, 0.0);
  for (int cls = 0; cls < config_.class_num_; cls++) {
    logits[cls] = fc_bias_[cls];
    for (int dim = 0; dim < static_cast<int>(head_input.size()); dim++) {
      logits[cls] += fc_weight_[cls][dim] * head_input[dim];
    }
  }
  return SUCCESS;
}

MiniCNNClassifier::RC MiniCNNClassifier::ForwardScore(const Tensor3D &input,
                                                      int class_idx,
                                                      double &score) {
  std::vector<double> logits;
  auto rc = Forward(input, logits);
  if (rc != SUCCESS) return rc;
  if (class_idx < 0 || class_idx >= static_cast<int>(logits.size())) {
    err_msg_ = "[MiniCNNClassifier::ForwardScore] Invalid class_idx";
    return INVALID_DATA;
  }
  score = logits[class_idx];
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

      Tensor3D conv_output, relu_output, conv2_output, relu2_output;
      Tensor3D pooled_output;
      std::vector<double> flattened;
      auto validate_rc = ValidateInput(images[sample_idx], "Train");
      if (validate_rc != SUCCESS) {
        return validate_rc;
      }
      auto conv_rc = conv_.Forward(images[sample_idx], conv_output);
      if (conv_rc != Conv2D::SUCCESS) {
        err_msg_ = conv_.err_msg();
        return INVALID_DATA;
      }
      relu_output = conv_output;
      ApplyRelu(relu_output);
      Tensor3D pool_input = relu_output;
      if (config_.conv2_channels_ > 0) {
        auto conv2_rc = conv2_.Forward(relu_output, conv2_output);
        if (conv2_rc != Conv2D::SUCCESS) {
          err_msg_ = conv2_.err_msg();
          return INVALID_DATA;
        }
        relu2_output = conv2_output;
        ApplyRelu(relu2_output);
        pool_input = relu2_output;
      }
      auto pool_rc = pool_.Forward(pool_input, pooled_output);
      if (pool_rc != MaxPool2D::SUCCESS) {
        err_msg_ = pool_.err_msg();
        return INVALID_DATA;
      }
      flattened = FlattenTensor(pooled_output);
      if (static_cast<int>(flattened.size()) != flattened_dim_) {
        err_msg_ = "[MiniCNNClassifier::Train] Invalid flatten result";
        return INVALID_DATA;
      }

      std::vector<double> head_input = flattened;
      std::vector<double> hidden_preact;
      if (config_.hidden_dim_ > 0) {
        head_input.assign(config_.hidden_dim_, 0.0);
        hidden_preact.assign(config_.hidden_dim_, 0.0);
        for (int h = 0; h < config_.hidden_dim_; ++h) {
          double v = hidden_bias_[h];
          for (int dim = 0; dim < flattened_dim_; ++dim) {
            v += hidden_weight_[h][dim] * flattened[dim];
          }
          hidden_preact[h] = v;
          head_input[h] = v > 0.0 ? v : 0.0;
        }
      }

      std::vector<double> logits(config_.class_num_, 0.0);
      for (int cls = 0; cls < config_.class_num_; cls++) {
        logits[cls] = fc_bias_[cls];
        for (int dim = 0; dim < static_cast<int>(head_input.size()); dim++) {
          logits[cls] += fc_weight_[cls][dim] * head_input[dim];
        }
      }
      auto probs = Softmax(logits);
      loss_sum += -std::log(std::max(probs[label], 1e-12));

      std::vector<double> grad_logits = probs;
      grad_logits[label] -= 1.0;

      Matrix grad_fc_weight(config_.class_num_,
                            std::vector<double>(head_input.size(), 0.0));
      std::vector<double> grad_fc_bias(config_.class_num_, 0.0);
      std::vector<double> grad_head(head_input.size(), 0.0);
      for (int cls = 0; cls < config_.class_num_; cls++) {
        const double grad = grad_logits[cls];
        grad_fc_bias[cls] = grad;
        for (int dim = 0; dim < static_cast<int>(head_input.size()); dim++) {
          grad_fc_weight[cls][dim] = grad * head_input[dim];
          grad_head[dim] += fc_weight_[cls][dim] * grad;
        }
      }
      std::vector<double> grad_flat(flattened_dim_, 0.0);
      Matrix grad_hidden_weight;
      std::vector<double> grad_hidden_bias;
      if (config_.hidden_dim_ > 0) {
        grad_hidden_weight.assign(config_.hidden_dim_,
                                  std::vector<double>(flattened_dim_, 0.0));
        grad_hidden_bias.assign(config_.hidden_dim_, 0.0);
        for (int h = 0; h < config_.hidden_dim_; ++h) {
          double grad = hidden_preact[h] > 0.0 ? grad_head[h] : 0.0;
          grad_hidden_bias[h] = grad;
          for (int dim = 0; dim < flattened_dim_; ++dim) {
            grad_hidden_weight[h][dim] = grad * flattened[dim];
            grad_flat[dim] += hidden_weight_[h][dim] * grad;
          }
        }
      } else {
        grad_flat = std::move(grad_head);
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

      Tensor3D grad_to_conv1;
      if (config_.conv2_channels_ > 0) {
        Tensor3D grad_conv2 = grad_relu;
        for (int channel = 0; channel < static_cast<int>(grad_conv2.size()); channel++) {
          for (int row = 0; row < static_cast<int>(grad_conv2[channel].size()); row++) {
            for (int col = 0;
                 col < static_cast<int>(grad_conv2[channel][row].size()); col++) {
              if (conv2_output[channel][row][col] <= 0.0) {
                grad_conv2[channel][row][col] = 0.0;
              }
            }
          }
        }
        auto conv2_backward_rc =
            conv2_.Backward(grad_conv2, grad_to_conv1, learning_rate);
        if (conv2_backward_rc != Conv2D::SUCCESS) {
          err_msg_ = conv2_.err_msg();
          return INVALID_DATA;
        }
      } else {
        grad_to_conv1 = grad_relu;
      }
      Tensor3D grad_conv = grad_to_conv1;
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
        for (int dim = 0; dim < static_cast<int>(head_input.size()); dim++) {
          fc_weight_[cls][dim] -= learning_rate * grad_fc_weight[cls][dim];
        }
      }
      if (config_.hidden_dim_ > 0) {
        for (int h = 0; h < config_.hidden_dim_; ++h) {
          hidden_bias_[h] -= learning_rate * grad_hidden_bias[h];
          for (int dim = 0; dim < flattened_dim_; ++dim) {
            hidden_weight_[h][dim] -= learning_rate * grad_hidden_weight[h][dim];
          }
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
  if (batch_size <= 0) {
    err_msg_ = "[MiniCNNClassifier::TrainBatch] Invalid training input";
    return INVALID_DATA;
  }
  return Train(images, labels, each_epoch_call, epoch_num, learning_rate);
}

MiniCNNClassifier::RC MiniCNNClassifier::TrainParallel(
    const std::vector<Tensor3D> &images, const std::vector<int> &labels,
    std::function<void(int epoch_num, double average_loss, bool &early_stop)>
        each_epoch_call,
    int epoch_num, double learning_rate, int batch_size, int thread_num) {
  if (!is_init_) {
    err_msg_ = "[MiniCNNClassifier::TrainParallel] MiniCNNClassifier not init";
    return NOT_INIT;
  }
  if (images.empty() || images.size() != labels.size() || epoch_num <= 0 ||
      learning_rate <= 0.0 || batch_size <= 0 || thread_num <= 0) {
    err_msg_ = "[MiniCNNClassifier::TrainParallel] Invalid training input";
    return INVALID_DATA;
  }
  if (thread_num <= 1 || batch_size <= 1) {
    return Train(images, labels, each_epoch_call, epoch_num, learning_rate);
  }

  std::vector<int> order(images.size(), 0);
  for (int i = 0; i < static_cast<int>(images.size()); i++) order[i] = i;
  std::mt19937 gen(static_cast<std::mt19937::result_type>(config_.rand_seed_));

  auto clone_from_self = [&]() {
    MiniCNNClassifier copy;
    copy.Init(config_);
    copy.conv().set_weight(conv_.weight());
    copy.conv().set_bias(conv_.bias());
    if (config_.conv2_channels_ > 0) {
      copy.conv2().set_weight(conv2_.weight());
      copy.conv2().set_bias(conv2_.bias());
    }
    copy.set_hidden_weight(hidden_weight_);
    copy.set_hidden_bias(hidden_bias_);
    copy.set_fc_weight(fc_weight_);
    copy.set_fc_bias(fc_bias_);
    return copy;
  };

  for (int epoch = 0; epoch < epoch_num; ++epoch) {
    std::shuffle(order.begin(), order.end(), gen);
    double loss_sum = 0.0;
    long loss_count = 0;
    for (int begin = 0; begin < static_cast<int>(order.size());
         begin += batch_size) {
      int end = std::min(begin + batch_size, static_cast<int>(order.size()));
      int B = end - begin;
      int workers = std::min(thread_num, B);
      std::vector<MiniCNNClassifier> models;
      models.reserve(workers);
      for (int w = 0; w < workers; ++w) models.push_back(clone_from_self());
      std::vector<double> worker_loss(workers, 0.0);
      std::vector<int> worker_count(workers, 0);
      std::vector<std::thread> threads;
      threads.reserve(workers);
      for (int w = 0; w < workers; ++w) {
        int wb = begin + B * w / workers;
        int we = begin + B * (w + 1) / workers;
        threads.emplace_back([&, w, wb, we]() {
          for (int pos = wb; pos < we; ++pos) {
            int idx = order[pos];
            std::vector<Tensor3D> one_img{images[idx]};
            std::vector<int> one_lbl{labels[idx]};
            models[w].Train(one_img, one_lbl, nullptr, 1, learning_rate);
            std::vector<double> probs;
            if (models[w].PredictProbs(images[idx], probs) == SUCCESS &&
                labels[idx] >= 0 && labels[idx] < static_cast<int>(probs.size())) {
              worker_loss[w] += -std::log(std::max(probs[labels[idx]], 1e-12));
              worker_count[w]++;
            }
          }
        });
      }
      for (auto &t : threads) t.join();

      // Average worker parameters. This is local-SGD style parallel training:
      // each worker takes a few SGD samples from the same starting point, then
      // the main model becomes the arithmetic mean of workers.
      auto avg_conv_w = conv_.weight();
      auto avg_conv_b = conv_.bias();
      auto avg_conv2_w = config_.conv2_channels_ > 0 ? conv2_.weight()
                                                     : Conv2D::Tensor4D();
      auto avg_conv2_b = config_.conv2_channels_ > 0 ? conv2_.bias()
                                                     : std::vector<double>();
      auto avg_hidden_w = hidden_weight_;
      auto avg_hidden_b = hidden_bias_;
      auto avg_fc_w = fc_weight_;
      auto avg_fc_b = fc_bias_;
      for (auto &oc : avg_conv_w)
        for (auto &ic : oc)
          for (auto &row : ic)
            for (double &v : row) v = 0.0;
      std::fill(avg_conv_b.begin(), avg_conv_b.end(), 0.0);
      for (auto &oc : avg_conv2_w)
        for (auto &ic : oc)
          for (auto &row : ic)
            for (double &v : row) v = 0.0;
      std::fill(avg_conv2_b.begin(), avg_conv2_b.end(), 0.0);
      for (auto &row : avg_hidden_w)
        std::fill(row.begin(), row.end(), 0.0);
      std::fill(avg_hidden_b.begin(), avg_hidden_b.end(), 0.0);
      for (auto &row : avg_fc_w)
        std::fill(row.begin(), row.end(), 0.0);
      std::fill(avg_fc_b.begin(), avg_fc_b.end(), 0.0);

      for (int w = 0; w < workers; ++w) {
        const auto &cw = models[w].conv().weight();
        const auto &cb = models[w].conv().bias();
        const auto &c2w = models[w].conv2().weight();
        const auto &c2b = models[w].conv2().bias();
        const auto &hw = models[w].hidden_weight();
        const auto &hb = models[w].hidden_bias();
        const auto &fw = models[w].fc_weight();
        const auto &fb = models[w].fc_bias();
        for (int oc = 0; oc < static_cast<int>(avg_conv_w.size()); ++oc)
          for (int ic = 0; ic < static_cast<int>(avg_conv_w[oc].size()); ++ic)
            for (int kh = 0; kh < static_cast<int>(avg_conv_w[oc][ic].size());
                 ++kh)
              for (int kw = 0;
                   kw < static_cast<int>(avg_conv_w[oc][ic][kh].size()); ++kw)
                avg_conv_w[oc][ic][kh][kw] += cw[oc][ic][kh][kw] / workers;
        for (int i = 0; i < static_cast<int>(avg_conv_b.size()); ++i)
          avg_conv_b[i] += cb[i] / workers;
        for (int oc = 0; oc < static_cast<int>(avg_conv2_w.size()); ++oc)
          for (int ic = 0; ic < static_cast<int>(avg_conv2_w[oc].size()); ++ic)
            for (int kh = 0; kh < static_cast<int>(avg_conv2_w[oc][ic].size());
                 ++kh)
              for (int kw = 0;
                   kw < static_cast<int>(avg_conv2_w[oc][ic][kh].size()); ++kw)
                avg_conv2_w[oc][ic][kh][kw] += c2w[oc][ic][kh][kw] / workers;
        for (int i = 0; i < static_cast<int>(avg_conv2_b.size()); ++i)
          avg_conv2_b[i] += c2b[i] / workers;
        for (int r = 0; r < static_cast<int>(avg_hidden_w.size()); ++r)
          for (int c = 0; c < static_cast<int>(avg_hidden_w[r].size()); ++c)
            avg_hidden_w[r][c] += hw[r][c] / workers;
        for (int i = 0; i < static_cast<int>(avg_hidden_b.size()); ++i)
          avg_hidden_b[i] += hb[i] / workers;
        for (int r = 0; r < static_cast<int>(avg_fc_w.size()); ++r)
          for (int c = 0; c < static_cast<int>(avg_fc_w[r].size()); ++c)
            avg_fc_w[r][c] += fw[r][c] / workers;
        for (int i = 0; i < static_cast<int>(avg_fc_b.size()); ++i)
          avg_fc_b[i] += fb[i] / workers;
        loss_sum += worker_loss[w];
        loss_count += worker_count[w];
      }
      conv_.set_weight(avg_conv_w);
      conv_.set_bias(avg_conv_b);
      if (config_.conv2_channels_ > 0) {
        conv2_.set_weight(avg_conv2_w);
        conv2_.set_bias(avg_conv2_b);
      }
      set_hidden_weight(avg_hidden_w);
      set_hidden_bias(avg_hidden_b);
      set_fc_weight(avg_fc_w);
      set_fc_bias(avg_fc_b);
    }
    bool early_stop = false;
    if (each_epoch_call != nullptr) {
      each_epoch_call(epoch, loss_count ? loss_sum / loss_count : 0.0,
                      early_stop);
    }
    if (early_stop) break;
  }
  return SUCCESS;
}

MiniCNNClassifier::RC MiniCNNClassifier::TrainPairwise(
    const std::vector<Tensor3D> &positive_images,
    const std::vector<Tensor3D> &negative_images,
    std::function<void(int epoch_num, double average_loss, bool &early_stop)>
        each_epoch_call,
    int epoch_num, double learning_rate) {
  if (!is_init_) {
    err_msg_ = "[MiniCNNClassifier::TrainPairwise] MiniCNNClassifier not init";
    return NOT_INIT;
  }
  if (positive_images.empty() || positive_images.size() != negative_images.size() ||
      epoch_num <= 0 || learning_rate <= 0.0 || config_.class_num_ < 2) {
    err_msg_ = "[MiniCNNClassifier::TrainPairwise] Invalid training input";
    return INVALID_DATA;
  }
  std::vector<int> order(positive_images.size(), 0);
  for (int i = 0; i < static_cast<int>(order.size()); ++i) order[i] = i;
  std::mt19937 gen(static_cast<std::mt19937::result_type>(config_.rand_seed_ + 31));
  for (int epoch = 0; epoch < epoch_num; ++epoch) {
    std::shuffle(order.begin(), order.end(), gen);
    double loss_sum = 0.0;
    for (int idx : order) {
      double loss = 0.0;
      auto rc =
          ApplyPairwiseGradient(positive_images[idx], negative_images[idx],
                                learning_rate, &loss);
      if (rc != SUCCESS) return rc;
      loss_sum += loss;
    }
    bool early_stop = false;
    if (each_epoch_call != nullptr) {
      each_epoch_call(epoch, loss_sum / positive_images.size(), early_stop);
    }
    if (early_stop) break;
  }
  return SUCCESS;
}

MiniCNNClassifier::RC MiniCNNClassifier::TrainPairwiseParallel(
    const std::vector<Tensor3D> &positive_images,
    const std::vector<Tensor3D> &negative_images,
    std::function<void(int epoch_num, double average_loss, bool &early_stop)>
        each_epoch_call,
    int epoch_num, double learning_rate, int batch_size, int thread_num) {
  if (!is_init_) {
    err_msg_ =
        "[MiniCNNClassifier::TrainPairwiseParallel] MiniCNNClassifier not init";
    return NOT_INIT;
  }
  if (positive_images.empty() ||
      positive_images.size() != negative_images.size() || epoch_num <= 0 ||
      learning_rate <= 0.0 || batch_size <= 0 || thread_num <= 0 ||
      config_.class_num_ < 2) {
    err_msg_ = "[MiniCNNClassifier::TrainPairwiseParallel] Invalid training input";
    return INVALID_DATA;
  }
  if (thread_num <= 1 || batch_size <= 1) {
    return TrainPairwise(positive_images, negative_images, each_epoch_call,
                         epoch_num, learning_rate);
  }

  std::vector<int> order(positive_images.size(), 0);
  for (int i = 0; i < static_cast<int>(order.size()); ++i) order[i] = i;
  std::mt19937 gen(static_cast<std::mt19937::result_type>(config_.rand_seed_ + 31));

  auto clone_from_self = [&]() {
    MiniCNNClassifier copy;
    copy.Init(config_);
    copy.conv().set_weight(conv_.weight());
    copy.conv().set_bias(conv_.bias());
    if (config_.conv2_channels_ > 0) {
      copy.conv2().set_weight(conv2_.weight());
      copy.conv2().set_bias(conv2_.bias());
    }
    copy.set_hidden_weight(hidden_weight_);
    copy.set_hidden_bias(hidden_bias_);
    copy.set_fc_weight(fc_weight_);
    copy.set_fc_bias(fc_bias_);
    return copy;
  };

  for (int epoch = 0; epoch < epoch_num; ++epoch) {
    std::shuffle(order.begin(), order.end(), gen);
    double loss_sum = 0.0;
    long loss_count = 0;
    for (int begin = 0; begin < static_cast<int>(order.size());
         begin += batch_size) {
      int end = std::min(begin + batch_size, static_cast<int>(order.size()));
      int B = end - begin;
      int workers = std::min(thread_num, B);
      std::vector<MiniCNNClassifier> models;
      models.reserve(workers);
      for (int w = 0; w < workers; ++w) models.push_back(clone_from_self());
      std::vector<double> worker_loss(workers, 0.0);
      std::vector<int> worker_count(workers, 0);
      std::vector<std::thread> threads;
      threads.reserve(workers);
      for (int w = 0; w < workers; ++w) {
        int wb = begin + B * w / workers;
        int we = begin + B * (w + 1) / workers;
        threads.emplace_back([&, w, wb, we]() {
          for (int pos = wb; pos < we; ++pos) {
            int idx = order[pos];
            double loss = 0.0;
            auto rc = models[w].ApplyPairwiseGradient(
                positive_images[idx], negative_images[idx], learning_rate, &loss);
            if (rc == SUCCESS) {
              worker_loss[w] += loss;
              worker_count[w]++;
            }
          }
        });
      }
      for (auto &t : threads) t.join();

      auto avg_conv_w = conv_.weight();
      auto avg_conv_b = conv_.bias();
      auto avg_conv2_w = config_.conv2_channels_ > 0 ? conv2_.weight()
                                                     : Conv2D::Tensor4D();
      auto avg_conv2_b = config_.conv2_channels_ > 0 ? conv2_.bias()
                                                     : std::vector<double>();
      auto avg_hidden_w = hidden_weight_;
      auto avg_hidden_b = hidden_bias_;
      auto avg_fc_w = fc_weight_;
      auto avg_fc_b = fc_bias_;
      for (auto &oc : avg_conv_w)
        for (auto &ic : oc)
          for (auto &row : ic)
            for (double &v : row) v = 0.0;
      std::fill(avg_conv_b.begin(), avg_conv_b.end(), 0.0);
      for (auto &oc : avg_conv2_w)
        for (auto &ic : oc)
          for (auto &row : ic)
            for (double &v : row) v = 0.0;
      std::fill(avg_conv2_b.begin(), avg_conv2_b.end(), 0.0);
      for (auto &row : avg_hidden_w)
        std::fill(row.begin(), row.end(), 0.0);
      std::fill(avg_hidden_b.begin(), avg_hidden_b.end(), 0.0);
      for (auto &row : avg_fc_w)
        std::fill(row.begin(), row.end(), 0.0);
      std::fill(avg_fc_b.begin(), avg_fc_b.end(), 0.0);

      for (int w = 0; w < workers; ++w) {
        const auto &cw = models[w].conv().weight();
        const auto &cb = models[w].conv().bias();
        const auto &c2w = models[w].conv2().weight();
        const auto &c2b = models[w].conv2().bias();
        const auto &hw = models[w].hidden_weight();
        const auto &hb = models[w].hidden_bias();
        const auto &fw = models[w].fc_weight();
        const auto &fb = models[w].fc_bias();
        for (int oc = 0; oc < static_cast<int>(avg_conv_w.size()); ++oc)
          for (int ic = 0; ic < static_cast<int>(avg_conv_w[oc].size()); ++ic)
            for (int kh = 0; kh < static_cast<int>(avg_conv_w[oc][ic].size());
                 ++kh)
              for (int kw = 0;
                   kw < static_cast<int>(avg_conv_w[oc][ic][kh].size()); ++kw)
                avg_conv_w[oc][ic][kh][kw] += cw[oc][ic][kh][kw] / workers;
        for (int i = 0; i < static_cast<int>(avg_conv_b.size()); ++i)
          avg_conv_b[i] += cb[i] / workers;
        for (int oc = 0; oc < static_cast<int>(avg_conv2_w.size()); ++oc)
          for (int ic = 0; ic < static_cast<int>(avg_conv2_w[oc].size()); ++ic)
            for (int kh = 0; kh < static_cast<int>(avg_conv2_w[oc][ic].size());
                 ++kh)
              for (int kw = 0;
                   kw < static_cast<int>(avg_conv2_w[oc][ic][kh].size()); ++kw)
                avg_conv2_w[oc][ic][kh][kw] += c2w[oc][ic][kh][kw] / workers;
        for (int i = 0; i < static_cast<int>(avg_conv2_b.size()); ++i)
          avg_conv2_b[i] += c2b[i] / workers;
        for (int r = 0; r < static_cast<int>(avg_hidden_w.size()); ++r)
          for (int c = 0; c < static_cast<int>(avg_hidden_w[r].size()); ++c)
            avg_hidden_w[r][c] += hw[r][c] / workers;
        for (int i = 0; i < static_cast<int>(avg_hidden_b.size()); ++i)
          avg_hidden_b[i] += hb[i] / workers;
        for (int r = 0; r < static_cast<int>(avg_fc_w.size()); ++r)
          for (int c = 0; c < static_cast<int>(avg_fc_w[r].size()); ++c)
            avg_fc_w[r][c] += fw[r][c] / workers;
        for (int i = 0; i < static_cast<int>(avg_fc_b.size()); ++i)
          avg_fc_b[i] += fb[i] / workers;
        loss_sum += worker_loss[w];
        loss_count += worker_count[w];
      }
      conv_.set_weight(avg_conv_w);
      conv_.set_bias(avg_conv_b);
      if (config_.conv2_channels_ > 0) {
        conv2_.set_weight(avg_conv2_w);
        conv2_.set_bias(avg_conv2_b);
      }
      set_hidden_weight(avg_hidden_w);
      set_hidden_bias(avg_hidden_b);
      set_fc_weight(avg_fc_w);
      set_fc_bias(avg_fc_b);
    }
    bool early_stop = false;
    if (each_epoch_call != nullptr) {
      each_epoch_call(epoch, loss_count ? loss_sum / loss_count : 0.0,
                      early_stop);
    }
    if (early_stop) break;
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
  int expected_dim = config_.hidden_dim_ > 0 ? config_.hidden_dim_ : flattened_dim_;
  for (const auto &row : weight) {
    if (static_cast<int>(row.size()) != expected_dim) {
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

MiniCNNClassifier::RC MiniCNNClassifier::set_hidden_weight(const Matrix &weight) {
  if (!is_init_) {
    err_msg_ = "[MiniCNNClassifier::set_hidden_weight] MiniCNNClassifier not init";
    return NOT_INIT;
  }
  if (config_.hidden_dim_ <= 0) {
    return weight.empty() ? SUCCESS : INVALID_DATA;
  }
  if (static_cast<int>(weight.size()) != config_.hidden_dim_) {
    err_msg_ = "[MiniCNNClassifier::set_hidden_weight] Invalid weight size";
    return INVALID_DATA;
  }
  for (const auto &row : weight) {
    if (static_cast<int>(row.size()) != flattened_dim_) {
      err_msg_ = "[MiniCNNClassifier::set_hidden_weight] Invalid weight size";
      return INVALID_DATA;
    }
  }
  hidden_weight_ = weight;
  return SUCCESS;
}

MiniCNNClassifier::RC
MiniCNNClassifier::set_hidden_bias(const std::vector<double> &bias) {
  if (!is_init_) {
    err_msg_ = "[MiniCNNClassifier::set_hidden_bias] MiniCNNClassifier not init";
    return NOT_INIT;
  }
  if (config_.hidden_dim_ <= 0) {
    return bias.empty() ? SUCCESS : INVALID_DATA;
  }
  if (static_cast<int>(bias.size()) != config_.hidden_dim_) {
    err_msg_ = "[MiniCNNClassifier::set_hidden_bias] Invalid bias size";
    return INVALID_DATA;
  }
  hidden_bias_ = bias;
  return SUCCESS;
}

std::string MiniCNNClassifier::err_msg() { return err_msg_; }

MiniCNNClassifier::Config MiniCNNClassifier::config() const { return config_; }

int MiniCNNClassifier::flattened_dim() const { return flattened_dim_; }

Conv2D &MiniCNNClassifier::conv() { return conv_; }

const Conv2D &MiniCNNClassifier::conv() const { return conv_; }

Conv2D &MiniCNNClassifier::conv2() { return conv2_; }

const Conv2D &MiniCNNClassifier::conv2() const { return conv2_; }

MaxPool2D &MiniCNNClassifier::pool() { return pool_; }

const MaxPool2D &MiniCNNClassifier::pool() const { return pool_; }

const MiniCNNClassifier::Matrix &MiniCNNClassifier::fc_weight() const {
  return fc_weight_;
}

const std::vector<double> &MiniCNNClassifier::fc_bias() const { return fc_bias_; }

const MiniCNNClassifier::Matrix &MiniCNNClassifier::hidden_weight() const {
  return hidden_weight_;
}

const std::vector<double> &MiniCNNClassifier::hidden_bias() const {
  return hidden_bias_;
}

MiniCNNClassifier::RC MiniCNNClassifier::ApplyLogitGradient(
    const Tensor3D &input, int class_idx, double grad_logit,
    double learning_rate) {
  if (!is_init_) {
    err_msg_ =
        "[MiniCNNClassifier::ApplyLogitGradient] MiniCNNClassifier not init";
    return NOT_INIT;
  }
  if (class_idx < 0 || class_idx >= config_.class_num_ ||
      learning_rate <= 0.0) {
    err_msg_ = "[MiniCNNClassifier::ApplyLogitGradient] Invalid input";
    return INVALID_DATA;
  }

  Tensor3D conv_output, relu_output, conv2_output, relu2_output;
  Tensor3D pooled_output;
  auto validate_rc = ValidateInput(input, "ApplyLogitGradient");
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
  Tensor3D pool_input = relu_output;
  if (config_.conv2_channels_ > 0) {
    auto conv2_rc = conv2_.Forward(relu_output, conv2_output);
    if (conv2_rc != Conv2D::SUCCESS) {
      err_msg_ = conv2_.err_msg();
      return INVALID_DATA;
    }
    relu2_output = conv2_output;
    ApplyRelu(relu2_output);
    pool_input = relu2_output;
  }
  auto pool_rc = pool_.Forward(pool_input, pooled_output);
  if (pool_rc != MaxPool2D::SUCCESS) {
    err_msg_ = pool_.err_msg();
    return INVALID_DATA;
  }
  std::vector<double> flattened = FlattenTensor(pooled_output);
  if (static_cast<int>(flattened.size()) != flattened_dim_) {
    err_msg_ = "[MiniCNNClassifier::ApplyLogitGradient] Invalid flatten result";
    return INVALID_DATA;
  }

  std::vector<double> head_input = flattened;
  std::vector<double> hidden_preact;
  if (config_.hidden_dim_ > 0) {
    head_input.assign(config_.hidden_dim_, 0.0);
    hidden_preact.assign(config_.hidden_dim_, 0.0);
    for (int h = 0; h < config_.hidden_dim_; ++h) {
      double v = hidden_bias_[h];
      for (int dim = 0; dim < flattened_dim_; ++dim) {
        v += hidden_weight_[h][dim] * flattened[dim];
      }
      hidden_preact[h] = v;
      head_input[h] = v > 0.0 ? v : 0.0;
    }
  }

  Matrix grad_fc_weight(config_.class_num_,
                        std::vector<double>(head_input.size(), 0.0));
  std::vector<double> grad_fc_bias(config_.class_num_, 0.0);
  std::vector<double> grad_head(head_input.size(), 0.0);
  grad_fc_bias[class_idx] = grad_logit;
  for (int dim = 0; dim < static_cast<int>(head_input.size()); dim++) {
    grad_fc_weight[class_idx][dim] = grad_logit * head_input[dim];
    grad_head[dim] += fc_weight_[class_idx][dim] * grad_logit;
  }

  std::vector<double> grad_flat(flattened_dim_, 0.0);
  Matrix grad_hidden_weight;
  std::vector<double> grad_hidden_bias;
  if (config_.hidden_dim_ > 0) {
    grad_hidden_weight.assign(config_.hidden_dim_,
                              std::vector<double>(flattened_dim_, 0.0));
    grad_hidden_bias.assign(config_.hidden_dim_, 0.0);
    for (int h = 0; h < config_.hidden_dim_; ++h) {
      double grad = hidden_preact[h] > 0.0 ? grad_head[h] : 0.0;
      grad_hidden_bias[h] = grad;
      for (int dim = 0; dim < flattened_dim_; ++dim) {
        grad_hidden_weight[h][dim] = grad * flattened[dim];
        grad_flat[dim] += hidden_weight_[h][dim] * grad;
      }
    }
  } else {
    grad_flat = std::move(grad_head);
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

  Tensor3D grad_to_conv1;
  if (config_.conv2_channels_ > 0) {
    Tensor3D grad_conv2 = grad_relu;
    for (int channel = 0; channel < static_cast<int>(grad_conv2.size());
         channel++) {
      for (int row = 0; row < static_cast<int>(grad_conv2[channel].size());
           row++) {
        for (int col = 0;
             col < static_cast<int>(grad_conv2[channel][row].size()); col++) {
          if (conv2_output[channel][row][col] <= 0.0) {
            grad_conv2[channel][row][col] = 0.0;
          }
        }
      }
    }
    auto conv2_backward_rc =
        conv2_.Backward(grad_conv2, grad_to_conv1, learning_rate);
    if (conv2_backward_rc != Conv2D::SUCCESS) {
      err_msg_ = conv2_.err_msg();
      return INVALID_DATA;
    }
  } else {
    grad_to_conv1 = grad_relu;
  }

  Tensor3D grad_conv = grad_to_conv1;
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
  auto conv_backward_rc = conv_.Backward(grad_conv, grad_input, learning_rate);
  if (conv_backward_rc != Conv2D::SUCCESS) {
    err_msg_ = conv_.err_msg();
    return INVALID_DATA;
  }

  fc_bias_[class_idx] -= learning_rate * grad_fc_bias[class_idx];
  for (int dim = 0; dim < static_cast<int>(head_input.size()); dim++) {
    fc_weight_[class_idx][dim] -= learning_rate * grad_fc_weight[class_idx][dim];
  }
  if (config_.hidden_dim_ > 0) {
    for (int h = 0; h < config_.hidden_dim_; ++h) {
      hidden_bias_[h] -= learning_rate * grad_hidden_bias[h];
      for (int dim = 0; dim < flattened_dim_; ++dim) {
        hidden_weight_[h][dim] -= learning_rate * grad_hidden_weight[h][dim];
      }
    }
  }
  return SUCCESS;
}

MiniCNNClassifier::RC MiniCNNClassifier::ApplyPairwiseGradient(
    const Tensor3D &positive_image, const Tensor3D &negative_image,
    double learning_rate, double *loss) {
  double sp = 0.0;
  double sn = 0.0;
  auto rcp = ForwardScore(positive_image, 1, sp);
  auto rcn = ForwardScore(negative_image, 1, sn);
  if (rcp != SUCCESS) return rcp;
  if (rcn != SUCCESS) return rcn;
  const double margin = sp - sn;
  if (loss != nullptr) {
    if (margin > 0.0) {
      *loss = std::log1p(std::exp(-margin));
    } else {
      *loss = -margin + std::log1p(std::exp(margin));
    }
  }
  const double grad_margin =
      margin >= 0.0 ? -std::exp(-margin) / (1.0 + std::exp(-margin))
                    : -1.0 / (1.0 + std::exp(margin));
  auto rc =
      ApplyLogitGradient(positive_image, 1, grad_margin, learning_rate);
  if (rc != SUCCESS) return rc;
  rc = ApplyLogitGradient(negative_image, 1, -grad_margin, learning_rate);
  if (rc != SUCCESS) return rc;
  return SUCCESS;
}

MiniCNNClassifier::RC MiniCNNClassifier::ValidateInput(
    const MiniCNNClassifier::Tensor3D &input, const char *func_name) {
  if (static_cast<int>(input.size()) != config_.input_channels_ || input.empty() ||
      input[0].empty() || input[0][0].empty()) {
    err_msg_ = std::string("[MiniCNNClassifier::") + func_name +
               "] Invalid input";
    return INVALID_DATA;
  }
  if (static_cast<int>(input[0].size()) != config_.input_height_ ||
      static_cast<int>(input[0][0].size()) != config_.input_width_) {
    err_msg_ = std::string("[MiniCNNClassifier::") + func_name +
               "] Invalid input";
    return INVALID_DATA;
  }
  for (const auto &channel : input) {
    if (static_cast<int>(channel.size()) != config_.input_height_) {
      err_msg_ = std::string("[MiniCNNClassifier::") + func_name +
                 "] Invalid input";
      return INVALID_DATA;
    }
    for (const auto &row : channel) {
      if (static_cast<int>(row.size()) != config_.input_width_) {
        err_msg_ = std::string("[MiniCNNClassifier::") + func_name +
                   "] Invalid input";
        return INVALID_DATA;
      }
    }
  }
  return SUCCESS;
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
  Tensor3D feature = relu_output;
  if (config_.conv2_channels_ > 0) {
    Tensor3D conv2_output;
    auto conv2_rc = conv2_.Forward(relu_output, conv2_output);
    if (conv2_rc != Conv2D::SUCCESS) {
      err_msg_ = conv2_.err_msg();
      return INVALID_DATA;
    }
    feature = conv2_output;
    ApplyRelu(feature);
  }
  auto pool_rc = pool_.Forward(feature, pooled_output);
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

} // namespace deeplearning

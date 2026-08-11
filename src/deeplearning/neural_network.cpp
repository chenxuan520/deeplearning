#include "neural_network.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <system_error>
#include <thread>

namespace deeplearning {

NeuralNetwork::NeuralNetwork() = default;

NeuralNetwork::~NeuralNetwork() = default;

NeuralNetwork::NeuralNetwork(const std::vector<int> &layer) { Init(layer); }

NeuralNetwork::RC NeuralNetwork::Init(const std::vector<int> &layer) {
  if (network_status_ != NETWORK_STATUS_UNINIT) {
    err_msg_ = "[NeuralNetwork::InitNetwork] Network has init";
    return ALREADY_INIT;
  }
  if (layer.size() < 2) {
    err_msg_ = "[NeuralNetwork::InitNetwork] Invalid layer size";
    return INVALID_DATA;
  }

  learning_rate_ = 0.1;
  rand_seed_ = 0;
  dropout_step_ = 0;

  softmax_function_ = SoftmaxFactory::Create(SOFTMAX_NONE);
  loss_function_ = LossFactory::Create(LOSS_MSE);
  activate_function_ = ActivateFactory::Create(ACTIVATE_SIGMOID);
  param_init_function_ = ParamInitFactory::Create(PARAM_INIT_ZERO);

  InitParamWithLayer(layer);
  optimizer_function_ = OptimizerFactory::Create(OPTIMIZER_SGD, layer_);
  param_init_function_->InitParam(neuron_weight_, neuron_bias_);

  network_status_ = NETWORK_STATUS_INIT;
  return SUCCESS;
}

NeuralNetwork::RC NeuralNetwork::Train(
    const std::vector<std::vector<double>> &data,
    const std::vector<std::vector<double>> &target,
    std::function<void(NeuralNetwork &network, int epoch_num, bool &early_stop)>
        each_epoch_call,
    int epoch_num, int batch_num, double learning_rate) {
  if (network_status_ != NETWORK_STATUS_INIT) {
    err_msg_ = "[NeuralNetwork::Train] Network not init";
    return NOT_INIT;
  }
  if (data.size() != target.size() || data.empty() || batch_num <= 0 ||
      batch_num > (int)data.size()) {
    err_msg_ = "[NeuralNetwork::Train] Invalid data input in size";
    return INVALID_DATA;
  }

  if (learning_rate != 0) {
    learning_rate_ = learning_rate;
  } else {
    learning_rate_ = (learning_rate_ != 0) ? learning_rate_ : 0.1;
  }

  std::vector<int> index_pos(data.size());
  for (int i = 0; i < (int)data.size(); i++) {
    index_pos[i] = i;
  }
  std::mt19937 shuffle_gen(rand_seed_);
  std::shuffle(index_pos.begin(), index_pos.end(), shuffle_gen);
  int batch_count = ((int)data.size() + batch_num - 1) / batch_num;

  epoch_num = epoch_num == 0 ? (int)data.size() : epoch_num;
  for (int i = 0; i < epoch_num; i++) {
    int batch_start = (i % batch_count) * batch_num;
    int batch_end = std::min(batch_start + batch_num, (int)data.size());
    int B = batch_end - batch_start;
    if (i % batch_count == 0) {
      std::shuffle(index_pos.begin(), index_pos.end(), shuffle_gen);
    }

    if (lr_scheduler_ != nullptr) {
      learning_rate_ = lr_scheduler_->GetLR(i);
    }

    // build batch (按 shuffle 后的索引顺序拷贝)
    std::vector<std::vector<double>> batch_data(B);
    std::vector<std::vector<double>> batch_target(B);
    for (int j = 0; j < B; j++) {
      batch_data[j] = data[index_pos[batch_start + j]];
      batch_target[j] = target[index_pos[batch_start + j]];
    }

    RC rc = SUCCESS;
    if (train_thread_num_ > 1 && B > 1) {
      ResetGradients();
      rc = AccumulateGradientsBatchParallel(batch_data, batch_target,
                                            dropout_step_);
    } else {
      rc = ForwardPropagationBatch(batch_data, /*training=*/true);
      if (rc != SUCCESS) {
        return rc;
      }
      ResetGradients();
      rc = BackPropagationBatch(batch_target);
    }
    if (rc != SUCCESS) {
      return rc;
    }
    if (grad_clip_norm_ > 0.0 || grad_clip_value_ > 0.0) {
      ClipGradients(B);
    }
    rc = ApplyGradient(B);
    if (rc != SUCCESS) {
      return rc;
    }
    dropout_step_++;

    auto early_stop = false;
    if (each_epoch_call != nullptr) {
      each_epoch_call(*this, i, early_stop);
      if (early_stop) {
        break;
      }
    }
  }
  return SUCCESS;
}

NeuralNetwork::RC NeuralNetwork::Predict(const std::vector<double> &data,
                                         std::vector<double> &result) {
  std::vector<std::vector<double>> batch = {data};
  auto rc = ForwardPropagationBatch(batch);
  if (rc != SUCCESS) {
    return rc;
  }
  result = neuron_output_[layer_.size() - 1][0];
  return SUCCESS;
}

NeuralNetwork::RC NeuralNetwork::PredictBatch(
    const std::vector<std::vector<double>> &data,
    std::vector<std::vector<double>> &result) {
  if (data.empty()) {
    result.clear();
    return SUCCESS;
  }
  auto rc = ForwardPropagationBatch(data);
  if (rc != SUCCESS) {
    return rc;
  }
  const auto &out = neuron_output_[layer_.size() - 1];
  int B = static_cast<int>(data.size());
  result.assign(B, std::vector<double>{});
  for (int i = 0; i < B; i++) {
    result[i] = out[i];
  }
  return SUCCESS;
}

NeuralNetwork::RC NeuralNetwork::CalcLoss(
    const std::vector<std::vector<double>> &data,
    const std::vector<std::vector<double>> &target, double &loss) {
  if (network_status_ != NETWORK_STATUS_INIT) {
    err_msg_ = "[NeuralNetwork::Train] Network not init";
    return NOT_INIT;
  }
  if (data.size() != target.size()) {
    err_msg_ = "[NeuralNetwork::Train] Invalid data input in size";
    return INVALID_DATA;
  }

  double loss_sum = 0;
  for (int i = 0; i < (int)data.size(); i++) {
    std::vector<std::vector<double>> batch = {data[i]};
    auto rc = ForwardPropagationBatch(batch);
    if (rc != SUCCESS) {
      return rc;
    }
    loss_sum += loss_function_->AverageLoss(
        target[i], neuron_output_[layer_.size() - 1][0]);
  }
  loss = loss_sum / data.size();
  return SUCCESS;
}

NeuralNetwork::RC NeuralNetwork::ExportNetworkParam(NetworkParam &param,
                                                    NetworkOption &option) {
  if (network_status_ != NETWORK_STATUS_INIT) {
    err_msg_ = "[NeuralNetwork::ExportNetworkParam] Network not init";
    return NOT_INIT;
  }

  param.layer_ = layer_;
  param.neuron_bias_ = neuron_bias_;
  param.neuron_weight_ = neuron_weight_;

  option.learning_rate_ = learning_rate_;
  option.rand_seed_ = rand_seed_;
  option.loss_type_ = loss_function_->GetLossType();
  option.activate_type_ = activate_function_->GetActivateType();
  option.softmax_type_ = softmax_function_->GetSoftmaxType();
  option.optimizer_type_ = optimizer_function_->GetOptimizerType();
  return SUCCESS;
}

NeuralNetwork::RC NeuralNetwork::ImportNetworkParam(
    const NetworkParam &param, const NetworkOption &option) {
  if (network_status_ != NETWORK_STATUS_UNINIT) {
    err_msg_ = "[NeuralNetwork::ImportNetworkParam] Network has init";
    return ALREADY_INIT;
  }
  if (param.layer_.size() < 2) {
    err_msg_ = "[NeuralNetwork::ImportNetworkParam] Invalid layer size";
    return INVALID_DATA;
  }

  layer_ = param.layer_;
  neuron_bias_ = param.neuron_bias_;
  neuron_weight_ = param.neuron_weight_;
  learning_rate_ = option.learning_rate_;
  rand_seed_ = option.rand_seed_;
  dropout_step_ = 0;

  loss_function_ = LossFactory::Create(option.loss_type_);
  activate_function_ = ActivateFactory::Create(option.activate_type_);
  softmax_function_ = SoftmaxFactory::Create(option.softmax_type_);
  param_init_function_ = ParamInitFactory::Create(PARAM_INIT_ZERO);
  optimizer_function_ = OptimizerFactory::Create(option.optimizer_type_, layer_);

  // batch buffer 留到第一次 forward/train 时按需 resize
  int L = (int)layer_.size();
  neuron_output_.assign(L, {});
  neuron_preact_.assign(L, {});
  neuron_delta_.assign(L, {});
  batch_buffer_size_ = 0;

  // grad buffer 形状跟 param 相同, 预分配并清零
  grad_bias_.assign(L, {});
  grad_weight_.assign(L, {});
  for (int i = 0; i < L; i++) {
    grad_bias_[i].assign(layer_[i], 0.0);
    if (i != 0) {
      grad_weight_[i].assign(layer_[i],
                             std::vector<double>(layer_[i - 1], 0.0));
    }
  }

  network_status_ = NETWORK_STATUS_INIT;
  return SUCCESS;
}

NeuralNetwork::RC NeuralNetwork::WidenHiddenLayer(
    int layer_index, int new_width, HiddenLayerWidenMode mode,
    ParamInitType new_param_init) {
  if (network_status_ != NETWORK_STATUS_INIT) {
    err_msg_ = "[NeuralNetwork::WidenHiddenLayer] Network not init";
    return NOT_INIT;
  }
  const int layer_num = static_cast<int>(layer_.size());
  if (layer_index <= 0 || layer_index >= layer_num - 1 ||
      new_width <= layer_[layer_index] ||
      (mode != WIDEN_RANDOM && mode != WIDEN_ZERO_OUTGOING &&
       mode != WIDEN_NET2WIDER) ||
      (mode == WIDEN_NET2WIDER && layer_[layer_index] <= 0)) {
    err_msg_ = "[NeuralNetwork::WidenHiddenLayer] Invalid expansion";
    return INVALID_DATA;
  }

  const int old_width = layer_[layer_index];
  std::vector<int> new_layer = layer_;
  new_layer[layer_index] = new_width;
  auto new_bias = neuron_bias_;
  auto new_weight = neuron_weight_;
  new_bias[layer_index].resize(new_width, 0.0);
  new_weight[layer_index].resize(
      new_width, std::vector<double>(layer_[layer_index - 1], 0.0));
  for (auto &row : new_weight[layer_index + 1]) {
    row.resize(new_width, 0.0);
  }

  if (mode == WIDEN_NET2WIDER) {
    std::vector<int> source(new_width);
    std::vector<int> copy_count(old_width, 1);
    for (int i = 0; i < old_width; i++) {
      source[i] = i;
    }
    for (int i = old_width; i < new_width; i++) {
      source[i] = (i - old_width) % old_width;
      copy_count[source[i]]++;
      new_bias[layer_index][i] = neuron_bias_[layer_index][source[i]];
      new_weight[layer_index][i] = neuron_weight_[layer_index][source[i]];
    }
    for (int out = 0; out < layer_[layer_index + 1]; out++) {
      for (int i = 0; i < new_width; i++) {
        new_weight[layer_index + 1][out][i] =
            neuron_weight_[layer_index + 1][out][source[i]] /
            copy_count[source[i]];
      }
    }
  } else {
    auto initializer = ParamInitFactory::Create(new_param_init);
    if (initializer == nullptr) {
      err_msg_ = "[NeuralNetwork::WidenHiddenLayer] Invalid param init type";
      return INVALID_DATA;
    }
    if (rand_seed_ != 0) {
      initializer->set_seed(rand_seed_);
    }
    std::vector<std::vector<double>> initialized_bias(layer_num);
    std::vector<std::vector<std::vector<double>>> initialized_weight(layer_num);
    for (int i = 0; i < layer_num; i++) {
      initialized_bias[i].assign(new_layer[i], 0.0);
      if (i != 0) {
        initialized_weight[i].assign(
            new_layer[i], std::vector<double>(new_layer[i - 1], 0.0));
      }
    }
    initializer->InitParam(initialized_weight, initialized_bias);
    for (int i = old_width; i < new_width; i++) {
      new_bias[layer_index][i] = initialized_bias[layer_index][i];
      new_weight[layer_index][i] = initialized_weight[layer_index][i];
      for (int out = 0; out < layer_[layer_index + 1]; out++) {
        new_weight[layer_index + 1][out][i] =
            mode == WIDEN_ZERO_OUTGOING
                ? 0.0
                : initialized_weight[layer_index + 1][out][i];
      }
    }
  }

  if (!optimizer_function_->ResetState(new_layer)) {
    err_msg_ =
        "[NeuralNetwork::WidenHiddenLayer] Optimizer cannot reset topology";
    return INVALID_DATA;
  }
  layer_ = std::move(new_layer);
  neuron_bias_ = std::move(new_bias);
  neuron_weight_ = std::move(new_weight);
  neuron_output_.assign(layer_num, {});
  neuron_preact_.assign(layer_num, {});
  neuron_delta_.assign(layer_num, {});
  grad_bias_.assign(layer_num, {});
  grad_weight_.assign(layer_num, {});
  for (int i = 0; i < layer_num; i++) {
    grad_bias_[i].assign(layer_[i], 0.0);
    if (i != 0) {
      grad_weight_[i].assign(
          layer_[i], std::vector<double>(layer_[i - 1], 0.0));
    }
  }
  batch_buffer_size_ = 0;
  dropout_mask_.clear();
  return SUCCESS;
}

NeuralNetwork::RC NeuralNetwork::Clone(const NeuralNetwork &old) {
  if (network_status_ != NETWORK_STATUS_UNINIT) {
    err_msg_ = "[NeuralNetwork::Clone] Network has init";
    return ALREADY_INIT;
  }

  layer_ = old.layer_;
  neuron_bias_ = old.neuron_bias_;
  neuron_weight_ = old.neuron_weight_;
  neuron_delta_ = old.neuron_delta_;
  neuron_output_ = old.neuron_output_;
  neuron_preact_ = old.neuron_preact_;
  grad_bias_ = old.grad_bias_;
  grad_weight_ = old.grad_weight_;
  batch_buffer_size_ = old.batch_buffer_size_;
  learning_rate_ = old.learning_rate_;
  train_thread_num_ = old.train_thread_num_;
  rand_seed_ = old.rand_seed_;
  dropout_rate_ = old.dropout_rate_;
  dropout_step_ = old.dropout_step_;
  network_status_ = old.network_status_;

  loss_function_ = LossFactory::Create(old.loss_function_->GetLossType());
  activate_function_ =
      ActivateFactory::Create(old.activate_function_->GetActivateType());
  softmax_function_ =
      SoftmaxFactory::Create(old.softmax_function_->GetSoftmaxType());
  param_init_function_ =
      ParamInitFactory::Create(old.param_init_function_->GetParamInitType());
  optimizer_function_ =
      OptimizerFactory::Create(old.optimizer_function_->GetOptimizerType(),
                               layer_);

  network_status_ = NETWORK_STATUS_INIT;
  return SUCCESS;
}

std::string NeuralNetwork::err_msg() { return err_msg_; }

double NeuralNetwork::learning_rate() { return learning_rate_; }

int NeuralNetwork::rand_seed() { return rand_seed_; }

NeuralNetwork::NetworkStatus NeuralNetwork::network_status() {
  return network_status_;
}

const std::vector<std::vector<std::vector<double>>> &
NeuralNetwork::neuron_weight() {
  return neuron_weight_;
}

const std::vector<std::vector<double>> &NeuralNetwork::neuron_bias() {
  return neuron_bias_;
}

void NeuralNetwork::set_learning_rate(double rate) { learning_rate_ = rate; }

void NeuralNetwork::set_train_thread_num(int thread_num) {
  train_thread_num_ = std::max(1, thread_num);
}

void NeuralNetwork::set_random_seed(int seed) {
  rand_seed_ = seed;
  dropout_step_ = 0;
  if (param_init_function_ != nullptr && seed != 0) {
    param_init_function_->set_seed(seed);
  }
}

NeuralNetwork::RC NeuralNetwork::set_loss_function(LossType type) {
  loss_function_ = LossFactory::Create(type);
  if (loss_function_ == nullptr) {
    err_msg_ = "[NeuralNetwork::set_loss_function] Invalid loss type";
    return INVALID_DATA;
  }
  return SUCCESS;
}

NeuralNetwork::RC NeuralNetwork::set_activate_function(ActivateType type) {
  activate_function_ = ActivateFactory::Create(type);
  if (activate_function_ == nullptr) {
    err_msg_ = "[NeuralNetwork::set_activate_function] Invalid activate type";
    return INVALID_DATA;
  }
  return SUCCESS;
}

NeuralNetwork::RC NeuralNetwork::set_softmax_function(SoftmaxType type) {
  softmax_function_ = SoftmaxFactory::Create(type);
  if (softmax_function_ == nullptr) {
    err_msg_ = "[NeuralNetwork::set_softmax_function] Invalid softmax type";
    return INVALID_DATA;
  }
  return SUCCESS;
}

NeuralNetwork::RC NeuralNetwork::set_param_init_function(ParamInitType type) {
  param_init_function_ = ParamInitFactory::Create(type);
  if (param_init_function_ == nullptr) {
    err_msg_ = "[NeuralNetwork::set_param_init_function] Invalid param_init type";
    return INVALID_DATA;
  }
  // 已经显式设置过 seed 的就用它, 否则保持 -1 (random_device 模式).
  if (rand_seed_ != 0) {
    param_init_function_->set_seed(rand_seed_);
  }
  param_init_function_->InitParam(neuron_weight_, neuron_bias_);
  return SUCCESS;
}

NeuralNetwork::RC NeuralNetwork::set_optimizer_function(OptimizerType type) {
  optimizer_function_ = OptimizerFactory::Create(type, layer_);
  if (optimizer_function_ == nullptr) {
    err_msg_ = "[NeuralNetwork::set_optimizer_function] Invalid optimizer type";
    return INVALID_DATA;
  }
  return SUCCESS;
}

void NeuralNetwork::InitParamWithLayer(const std::vector<int> &layer) {
  layer_ = layer;
  int L = (int)layer.size();
  // 共享参数
  neuron_bias_.assign(L, {});
  neuron_weight_.assign(L, {});
  // batch 缓冲 (按需 resize)
  neuron_output_.assign(L, {});
  neuron_preact_.assign(L, {});
  neuron_delta_.assign(L, {});
  // 梯度累加缓冲
  grad_bias_.assign(L, {});
  grad_weight_.assign(L, {});

  for (int i = 0; i < L; i++) {
    neuron_bias_[i].assign(layer[i], 0.0);
    grad_bias_[i].assign(layer[i], 0.0);
    if (i != 0) {
      neuron_weight_[i].assign(layer[i],
                               std::vector<double>(layer[i - 1], 0.0));
      grad_weight_[i].assign(layer[i],
                             std::vector<double>(layer[i - 1], 0.0));
    }
  }
  batch_buffer_size_ = 0;
}

void NeuralNetwork::ResizeBatchBuffers(int batch_size) {
  if (batch_buffer_size_ == batch_size) {
    // 检查最内层维度是否还匹配 layer_ (导入时 layer_ 可能变化)
    if ((int)neuron_output_.size() == (int)layer_.size() &&
        (batch_size == 0 ||
         (int)neuron_output_[0].size() == batch_size)) {
      return;
    }
  }
  int L = (int)layer_.size();
  if ((int)neuron_preact_.size() != L) {
    neuron_preact_.assign(L, {});
  }
  for (int l = 0; l < L; l++) {
    neuron_output_[l].assign(batch_size, std::vector<double>(layer_[l], 0.0));
    neuron_preact_[l].assign(batch_size, std::vector<double>(layer_[l], 0.0));
    neuron_delta_[l].assign(batch_size, std::vector<double>(layer_[l], 0.0));
  }
  batch_buffer_size_ = batch_size;
}

void NeuralNetwork::ResetGradients() {
  int L = (int)layer_.size();
  for (int l = 0; l < L; l++) {
    std::fill(grad_bias_[l].begin(), grad_bias_[l].end(), 0.0);
    if (l >= 1) {
      for (int o = 0; o < layer_[l]; o++) {
        std::fill(grad_weight_[l][o].begin(), grad_weight_[l][o].end(), 0.0);
      }
    }
  }
}

void NeuralNetwork::InitGradientBuffer(GradientBuffer &buffer) const {
  int L = (int)layer_.size();
  buffer.bias.assign(L, {});
  buffer.weight.assign(L, {});
  for (int l = 0; l < L; l++) {
    buffer.bias[l].assign(layer_[l], 0.0);
    if (l >= 1) {
      buffer.weight[l].assign(layer_[l],
                              std::vector<double>(layer_[l - 1], 0.0));
    }
  }
}

NeuralNetwork::RC NeuralNetwork::AccumulateGradientsBatchParallel(
    const std::vector<std::vector<double>> &batch_data,
    const std::vector<std::vector<double>> &batch_target, int train_step) {
  int B = (int)batch_data.size();
  if (B == 0 || B != (int)batch_target.size()) {
    err_msg_ =
        "[NeuralNetwork::AccumulateGradientsBatchParallel] invalid batch";
    return INVALID_DATA;
  }

  int worker_num = std::min(train_thread_num_, B);
  int hardware_thread_num =
      static_cast<int>(std::thread::hardware_concurrency());
  if (hardware_thread_num > 0) {
    worker_num = std::min(worker_num, hardware_thread_num);
  }
  std::vector<GradientBuffer> worker_grad(worker_num);
  for (auto &gradient : worker_grad) {
    InitGradientBuffer(gradient);
  }

  auto loss_type = loss_function_->GetLossType();
  auto activate_type = activate_function_->GetActivateType();
  auto softmax_type = softmax_function_->GetSoftmaxType();
  std::vector<RC> worker_rc(worker_num, SUCCESS);
  std::vector<std::string> worker_err(worker_num);
  std::vector<std::thread> threads;
  threads.reserve(worker_num);

  try {
    for (int worker_idx = 0; worker_idx < worker_num; worker_idx++) {
      int begin = B * worker_idx / worker_num;
      int end = B * (worker_idx + 1) / worker_num;
      threads.emplace_back([&, worker_idx, begin, end]() {
        auto worker_loss = LossFactory::Create(loss_type);
        auto worker_activate = ActivateFactory::Create(activate_type);
        auto worker_softmax = SoftmaxFactory::Create(softmax_type);
        if (worker_loss == nullptr || worker_activate == nullptr ||
            worker_softmax == nullptr) {
          worker_err[worker_idx] =
              "[NeuralNetwork::AccumulateGradientsBatchParallel] invalid "
              "strategy";
          worker_rc[worker_idx] = INVALID_DATA;
          return;
        }
        worker_rc[worker_idx] = AccumulateGradientsRange(
            batch_data, batch_target, begin, end, worker_loss, worker_activate,
            worker_softmax, worker_grad[worker_idx], worker_err[worker_idx],
            train_step);
      });
    }
  } catch (const std::system_error &e) {
    for (auto &thread : threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    err_msg_ =
        std::string("[NeuralNetwork::AccumulateGradientsBatchParallel] failed ") +
        "to create worker thread: " + e.what();
    return INVALID_DATA;
  }

  for (auto &thread : threads) {
    thread.join();
  }

  for (int worker_idx = 0; worker_idx < worker_num; worker_idx++) {
    if (worker_rc[worker_idx] != SUCCESS) {
      err_msg_ = worker_err[worker_idx];
      return worker_rc[worker_idx];
    }
  }

  int L = (int)layer_.size();
  for (int worker_idx = 0; worker_idx < worker_num; worker_idx++) {
    for (int l = 1; l < L; l++) {
      int dim_l = layer_[l];
      int dim_lm1 = layer_[l - 1];
      for (int o = 0; o < dim_l; o++) {
        grad_bias_[l][o] += worker_grad[worker_idx].bias[l][o];
        for (int i = 0; i < dim_lm1; i++) {
          grad_weight_[l][o][i] += worker_grad[worker_idx].weight[l][o][i];
        }
      }
    }
  }
  return SUCCESS;
}

NeuralNetwork::RC NeuralNetwork::AccumulateGradientsRange(
    const std::vector<std::vector<double>> &batch_data,
    const std::vector<std::vector<double>> &batch_target, int begin, int end,
    const std::shared_ptr<LossFunction> &loss_function,
    const std::shared_ptr<ActivateFunction> &activate_function,
    const std::shared_ptr<SoftmaxFunction> &softmax_function,
    GradientBuffer &gradient, std::string &err_msg,
    int train_step) const {
  int L = (int)layer_.size();
  int B = end - begin;
  if (L == 0 || B <= 0) {
    err_msg = "[NeuralNetwork::AccumulateGradientsRange] empty";
    return INVALID_DATA;
  }

  std::vector<std::vector<std::vector<double>>> output(L), preact(L), delta(L);
  for (int l = 0; l < L; l++) {
    output[l].assign(B, std::vector<double>(layer_[l], 0.0));
    preact[l].assign(B, std::vector<double>(layer_[l], 0.0));
    delta[l].assign(B, std::vector<double>(layer_[l], 0.0));
  }

  // dropout: 掩码是本函数内的局部变量 (前向采样, 反向复用).
  // 种子按 (rand_seed_, train_step, 样本在 batch 内的槽位) 派生,
  // 每个样本一个生成器跨层连续采样 — 与单线程路径采出的掩码完全一致,
  // 且与 worker 数/batch 切分方式无关.
  bool use_dropout = dropout_rate_ > 0.0;
  std::vector<std::vector<std::vector<double>>> mask;
  if (use_dropout) {
    mask.assign(L, {});
    for (int l = 1; l < L - 1; l++) {
      mask[l].assign(B, std::vector<double>(layer_[l], 1.0));
    }
    for (int b = 0; b < B; b++) {
      std::mt19937 gen(DropoutSeed(rand_seed_, train_step, begin + b));
      for (int l = 1; l < L - 1; l++) {
        FillDropoutMask(gen, mask[l][b].data(), layer_[l]);
      }
    }
  }

  for (int b = 0; b < B; b++) {
    const auto &sample = batch_data[begin + b];
    if ((int)sample.size() != layer_[0]) {
      err_msg =
          "[NeuralNetwork::AccumulateGradientsRange] sample dim mismatch";
      return INVALID_DATA;
    }
    for (int j = 0; j < layer_[0]; j++) {
      output[0][b][j] = sample[j];
    }
  }

  for (int l = 1; l < L; l++) {
    int out_dim = layer_[l];
    int in_dim = layer_[l - 1];
    bool drop_layer = use_dropout && l < L - 1;
    for (int b = 0; b < B; b++) {
      const auto &in_vec = output[l - 1][b];
      auto &out_vec = output[l][b];
      auto &pre_vec = preact[l][b];
      for (int o = 0; o < out_dim; o++) {
        double z = neuron_bias_[l][o];
        const auto &w_row = neuron_weight_[l][o];
        for (int i = 0; i < in_dim; i++) {
          z += w_row[i] * in_vec[i];
        }
        pre_vec[o] = z;
        double a = activate_function->Activate(z);
        if (drop_layer) {
          a *= mask[l][b][o];
        }
        out_vec[o] = a;
      }
    }
  }

  if (softmax_function->GetSoftmaxType() != SOFTMAX_NONE) {
    int now_layer = L - 1;
    int last_layer = L - 2;
    int out_dim = layer_[now_layer];
    int in_dim = layer_[last_layer];
    for (int b = 0; b < B; b++) {
      std::vector<double> logits;
      logits.reserve(out_dim);
      const auto &in_vec = output[last_layer][b];
      for (int o = 0; o < out_dim; o++) {
        double z = neuron_bias_[now_layer][o];
        const auto &w_row = neuron_weight_[now_layer][o];
        for (int i = 0; i < in_dim; i++) {
          z += w_row[i] * in_vec[i];
        }
        logits.push_back(z);
      }
      softmax_function->Normalize(logits, output[now_layer][b]);
    }
  }

  int last = L - 1;
  int out_dim = layer_[last];
  bool use_softmax = (softmax_function->GetSoftmaxType() != SOFTMAX_NONE);
  for (int b = 0; b < B; b++) {
    const auto &target = batch_target[begin + b];
    if ((int)target.size() != out_dim) {
      err_msg =
          "[NeuralNetwork::AccumulateGradientsRange] target dim mismatch";
      return INVALID_DATA;
    }
    for (int o = 0; o < out_dim; o++) {
      double d;
      if (use_softmax) {
        d = softmax_function->CalcDelta(output[last][b][o], target[o],
                                        loss_function);
      } else {
        double dL = loss_function->DerivLoss(target[o], output[last][b][o]) /
                    (double)out_dim;
        d = dL * activate_function->DerivActivate(preact[last][b][o],
                                                  output[last][b][o]);
      }
      delta[last][b][o] = d;
    }
    for (int l = last - 1; l >= 1; l--) {
      int dim_l = layer_[l];
      int dim_lp1 = layer_[l + 1];
      bool drop_layer = use_dropout && l < L - 1;
      for (int o = 0; o < dim_l; o++) {
        double sum = 0.0;
        for (int k = 0; k < dim_lp1; k++) {
          sum += neuron_weight_[l + 1][k][o] * delta[l + 1][b][k];
        }
        double m = 1.0;
        double a = output[l][b][o];
        if (drop_layer) {
          m = mask[l][b][o];
          if (m > 0.0) {
            a /= m; // 还原丢弃前的激活值, 供按 output 求导的激活使用
          }
        }
        delta[l][b][o] =
            sum * activate_function->DerivActivate(preact[l][b][o], a) * m;
      }
    }
    for (int l = 1; l < L; l++) {
      int dim_l = layer_[l];
      int dim_lm1 = layer_[l - 1];
      const auto &in_vec = output[l - 1][b];
      for (int o = 0; o < dim_l; o++) {
        double d = delta[l][b][o];
        gradient.bias[l][o] += d;
        for (int i = 0; i < dim_lm1; i++) {
          gradient.weight[l][o][i] += d * in_vec[i];
        }
      }
    }
  }
  return SUCCESS;
}

NeuralNetwork::RC NeuralNetwork::ForwardPropagationBatch(
    const std::vector<std::vector<double>> &batch_data, bool training) {
  int L = (int)layer_.size();
  int B = (int)batch_data.size();
  if (L == 0) {
    err_msg_ = "[NeuralNetwork::ForwardPropagationBatch] empty layer";
    return INVALID_DATA;
  }
  if (B == 0) {
    err_msg_ = "[NeuralNetwork::ForwardPropagationBatch] empty batch";
    return INVALID_DATA;
  }

  ResizeBatchBuffers(B);

  // 训练且开启 dropout 时, 先为每个样本采样好全部隐藏层掩码:
  // 每个样本一个生成器 (种子按 rand_seed_, dropout_step_, 槽位 派生),
  // 跨层连续采样 — 多线程路径按同样规则采, 两条路径掩码完全一致.
  bool use_dropout = training && dropout_rate_ > 0.0;
  if (use_dropout) {
    ResizeDropoutMask(B);
    for (int b = 0; b < B; b++) {
      std::mt19937 gen(DropoutSeed(rand_seed_, dropout_step_, b));
      for (int l = 1; l < L - 1; l++) {
        FillDropoutMask(gen, dropout_mask_[l][b].data(), layer_[l]);
      }
    }
  }

  // layer 0: 输入原样拷贝
  for (int b = 0; b < B; b++) {
    if ((int)batch_data[b].size() != layer_[0]) {
      err_msg_ = "[NeuralNetwork::ForwardPropagationBatch] sample dim mismatch";
      return INVALID_DATA;
    }
    for (int j = 0; j < layer_[0]; j++) {
      neuron_output_[0][b][j] = batch_data[b][j];
    }
  }

  // 1..L-1: 全连接前向 + 激活, 同时记录 pre-activation
  // 开启 dropout 的隐藏层: 激活值立即乘上掩码因子, 再喂给下一层
  for (int l = 1; l < L; l++) {
    int out_dim = layer_[l];
    int in_dim = layer_[l - 1];
    bool drop_layer = use_dropout && l < L - 1;
    for (int b = 0; b < B; b++) {
      const auto &in_vec = neuron_output_[l - 1][b];
      auto &out_vec = neuron_output_[l][b];
      auto &pre_vec = neuron_preact_[l][b];
      for (int o = 0; o < out_dim; o++) {
        double z = neuron_bias_[l][o];
        const auto &w_row = neuron_weight_[l][o];
        for (int i = 0; i < in_dim; i++) {
          z += w_row[i] * in_vec[i];
        }
        pre_vec[o] = z;
        double a = activate_function_->Activate(z);
        if (drop_layer) {
          a *= dropout_mask_[l][b][o];
        }
        out_vec[o] = a;
      }
    }
  }

  // softmax 路径: 最后一层用 logits 重新算 + softmax 归一化 (覆盖 activation)
  if (softmax_function_->GetSoftmaxType() != SOFTMAX_NONE) {
    auto rc = UpdateNeuronOutputBatchSoftMax();
    if (rc != SUCCESS) {
      return rc;
    }
  }
  return SUCCESS;
}

NeuralNetwork::RC NeuralNetwork::UpdateNeuronOutputBatchSoftMax() {
  int L = (int)layer_.size();
  if (L < 2) {
    err_msg_ = "[NeuralNetwork::UpdateNeuronOutputBatchSoftMax] Invalid layer";
    return INVALID_DATA;
  }
  int now_layer = L - 1;
  int last_layer = L - 2;
  int B = batch_buffer_size_;
  int out_dim = layer_[now_layer];
  int in_dim = layer_[last_layer];

  for (int b = 0; b < B; b++) {
    std::vector<double> logits;
    logits.reserve(out_dim);
    const auto &in_vec = neuron_output_[last_layer][b];
    for (int o = 0; o < out_dim; o++) {
      double z = neuron_bias_[now_layer][o];
      const auto &w_row = neuron_weight_[now_layer][o];
      for (int i = 0; i < in_dim; i++) {
        z += w_row[i] * in_vec[i];
      }
      logits.push_back(z);
    }
    softmax_function_->Normalize(logits, neuron_output_[now_layer][b]);
  }
  return SUCCESS;
}

NeuralNetwork::RC NeuralNetwork::BackPropagationBatch(
    const std::vector<std::vector<double>> &batch_target) {
  int L = (int)layer_.size();
  int B = (int)batch_target.size();
  if (L == 0 || B == 0) {
    err_msg_ = "[NeuralNetwork::BackPropagationBatch] empty";
    return INVALID_DATA;
  }
  if (B != batch_buffer_size_) {
    err_msg_ = "[NeuralNetwork::BackPropagationBatch] batch buffer mismatch";
    return INVALID_DATA;
  }
  int last = L - 1;
  int out_dim = layer_[last];
  bool use_softmax = (softmax_function_->GetSoftmaxType() != SOFTMAX_NONE);

  for (int b = 0; b < B; b++) {
    if ((int)batch_target[b].size() != out_dim) {
      err_msg_ = "[NeuralNetwork::BackPropagationBatch] target dim mismatch";
      return INVALID_DATA;
    }
    // 1) 末层 delta
    for (int o = 0; o < out_dim; o++) {
      double delta;
      if (use_softmax) {
        delta = softmax_function_->CalcDelta(neuron_output_[last][b][o],
                                              batch_target[b][o],
                                              loss_function_);
      } else {
        double dL = loss_function_->DerivLoss(batch_target[b][o],
                                              neuron_output_[last][b][o]) /
                    (double)out_dim;
        delta = dL * activate_function_->DerivActivate(
                         neuron_preact_[last][b][o],
                         neuron_output_[last][b][o]);
      }
      neuron_delta_[last][b][o] = delta;
    }
    // 2) 反向传播到隐藏层 (layer 0 不需要)
    // 开启 dropout 的层: delta 乘上前向时采样的同一份掩码因子;
    // 由于 output 存的是丢弃后的值 (a * m), 先用 output / m 还原出
    // 丢弃前的激活值 a, 供 sigmoid/tanh 这类按 output 求导的激活使用.
    for (int l = last - 1; l >= 1; l--) {
      int dim_l = layer_[l];
      int dim_lp1 = layer_[l + 1];
      bool drop_layer = DropoutMaskActive(l, B);
      for (int o = 0; o < dim_l; o++) {
        double sum = 0.0;
        for (int k = 0; k < dim_lp1; k++) {
          sum += neuron_weight_[l + 1][k][o] * neuron_delta_[l + 1][b][k];
        }
        double m = 1.0;
        double a = neuron_output_[l][b][o];
        if (drop_layer) {
          m = dropout_mask_[l][b][o];
          if (m > 0.0) {
            a /= m;
          }
        }
        neuron_delta_[l][b][o] =
            sum *
            activate_function_->DerivActivate(neuron_preact_[l][b][o], a) * m;
      }
    }
    // 3) 累加梯度: grad_bias += delta; grad_weight[o][i] += delta[o] * in[i]
    for (int l = 1; l < L; l++) {
      int dim_l = layer_[l];
      int dim_lm1 = layer_[l - 1];
      const auto &in_vec = neuron_output_[l - 1][b];
      for (int o = 0; o < dim_l; o++) {
        double d = neuron_delta_[l][b][o];
        grad_bias_[l][o] += d;
        auto &g_row = grad_weight_[l][o];
        for (int i = 0; i < dim_lm1; i++) {
          g_row[i] += d * in_vec[i];
        }
      }
    }
  }
  return SUCCESS;
}

NeuralNetwork::RC NeuralNetwork::ApplyGradient(int batch_size) {
  if (batch_size <= 0) {
    err_msg_ = "[NeuralNetwork::ApplyGradient] invalid batch_size";
    return INVALID_DATA;
  }
  int L = (int)layer_.size();
  double inv_bs = 1.0 / (double)batch_size;

  optimizer_function_->BeforeStep();

  for (int l = 1; l < L; l++) {
    int dim_l = layer_[l];
    int dim_lm1 = layer_[l - 1];
    for (int o = 0; o < dim_l; o++) {
      double avg_db = grad_bias_[l][o] * inv_bs;
      double bias_change = optimizer_function_->CalcChangeValue(
          avg_db, learning_rate_, {l, o}, -1, neuron_bias_[l][o]);
      neuron_bias_[l][o] -= bias_change;

      auto &w_row = neuron_weight_[l][o];
      const auto &g_row = grad_weight_[l][o];
      for (int i = 0; i < dim_lm1; i++) {
        double avg_dw = g_row[i] * inv_bs;
        double w_change = optimizer_function_->CalcChangeValue(
            avg_dw, learning_rate_, {l, o}, i, w_row[i]);
        w_row[i] -= w_change;
      }
    }
  }
  return SUCCESS;
}

void NeuralNetwork::ClipGradients(int batch_size) {
  if (batch_size <= 0) {
    return;
  }
  int L = (int)layer_.size();
  double inv_bs = 1.0 / (double)batch_size;

  // by-value: 逐分量裁剪 (作用在平均梯度上, 等价于裁剪 grad/B 后再 *B)
  if (grad_clip_value_ > 0.0) {
    double thresh = grad_clip_value_;
    for (int l = 1; l < L; l++) {
      int dim_l = layer_[l];
      int dim_lm1 = layer_[l - 1];
      for (int o = 0; o < dim_l; o++) {
        double avg = grad_bias_[l][o] * inv_bs;
        if (avg > thresh) {
          grad_bias_[l][o] = thresh * batch_size;
        } else if (avg < -thresh) {
          grad_bias_[l][o] = -thresh * batch_size;
        }
        auto &g_row = grad_weight_[l][o];
        for (int i = 0; i < dim_lm1; i++) {
          double a = g_row[i] * inv_bs;
          if (a > thresh) {
            g_row[i] = thresh * batch_size;
          } else if (a < -thresh) {
            g_row[i] = -thresh * batch_size;
          }
        }
      }
    }
  }

  // by-norm: 计算全局 L2 范数, 超阈值整体缩放.
  if (grad_clip_norm_ > 0.0) {
    double sq_sum = 0.0;
    for (int l = 1; l < L; l++) {
      int dim_l = layer_[l];
      int dim_lm1 = layer_[l - 1];
      for (int o = 0; o < dim_l; o++) {
        double avg = grad_bias_[l][o] * inv_bs;
        sq_sum += avg * avg;
        const auto &g_row = grad_weight_[l][o];
        for (int i = 0; i < dim_lm1; i++) {
          double a = g_row[i] * inv_bs;
          sq_sum += a * a;
        }
      }
    }
    double norm = std::sqrt(sq_sum);
    if (norm > grad_clip_norm_ && norm > 0.0) {
      double scale = grad_clip_norm_ / norm;
      for (int l = 1; l < L; l++) {
        int dim_l = layer_[l];
        int dim_lm1 = layer_[l - 1];
        for (int o = 0; o < dim_l; o++) {
          grad_bias_[l][o] *= scale;
          auto &g_row = grad_weight_[l][o];
          for (int i = 0; i < dim_lm1; i++) {
            g_row[i] *= scale;
          }
        }
      }
    }
  }
}

void NeuralNetwork::set_gradient_clip_norm(double max_norm) {
  grad_clip_norm_ = max_norm;
}

void NeuralNetwork::set_gradient_clip_value(double max_value) {
  grad_clip_value_ = max_value;
}

NeuralNetwork::RC NeuralNetwork::set_dropout_rate(double rate) {
  if (rate < 0.0 || rate >= 1.0) {
    err_msg_ = "[NeuralNetwork::set_dropout_rate] rate must be in [0, 1)";
    return INVALID_DATA;
  }
  dropout_rate_ = rate;
  return SUCCESS;
}

unsigned int NeuralNetwork::DropoutSeed(int rand_seed, int step, int slot) {
  // splitmix64 混合: 输入差一位也会让输出彻底不同, 避免线性组合
  // 带来的样本间掩码相关性.
  uint64_t x = (uint64_t)(uint32_t)rand_seed;
  x = x * 0x9e3779b97f4a7c15ull + (uint64_t)(uint32_t)step;
  x = x * 0x9e3779b97f4a7c15ull + (uint64_t)(uint32_t)slot;
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  x = x ^ (x >> 31);
  return (unsigned int)(x & 0xffffffffu);
}

void NeuralNetwork::FillDropoutMask(std::mt19937 &gen, double *factors,
                                    int n) const {
  double keep = 1.0 - dropout_rate_;
  // 直接用引擎的 uint32 输出转 [0,1), 不经过 uniform_real_distribution
  // (其实现由各家 stdlib 自定): mt19937 引擎输出是标准规定的, 这样掩码
  // 序列在不同平台/编译器下也完全一致.
  for (int i = 0; i < n; i++) {
    double u = (double)gen() * (1.0 / 4294967296.0);
    factors[i] = (u < keep) ? 1.0 / keep : 0.0;
  }
}

bool NeuralNetwork::DropoutMaskActive(int layer, int batch_size) const {
  return dropout_rate_ > 0.0 && layer < (int)dropout_mask_.size() &&
         (int)dropout_mask_[layer].size() == batch_size;
}

void NeuralNetwork::ResizeDropoutMask(int batch_size) {
  int L = (int)layer_.size();
  if ((int)dropout_mask_.size() == L && L > 1 &&
      (int)dropout_mask_[1].size() == batch_size &&
      (batch_size == 0 ||
       (int)dropout_mask_[1][0].size() == layer_[1])) {
    return;
  }
  dropout_mask_.assign(L, {});
  for (int l = 1; l < L - 1; l++) {
    dropout_mask_[l].assign(batch_size, std::vector<double>(layer_[l], 1.0));
  }
}

void NeuralNetwork::set_lr_scheduler(std::shared_ptr<LRScheduler> scheduler) {
  lr_scheduler_ = std::move(scheduler);
}

NeuralNetwork::RC NeuralNetwork::set_optimizer_function(
    std::shared_ptr<OptimizerFunction> optimizer) {
  if (optimizer == nullptr) {
    err_msg_ = "[NeuralNetwork::set_optimizer_function] null optimizer";
    return INVALID_DATA;
  }
  optimizer_function_ = std::move(optimizer);
  return SUCCESS;
}

std::shared_ptr<OptimizerFunction> NeuralNetwork::optimizer_function() {
  return optimizer_function_;
}

} // namespace deeplearning

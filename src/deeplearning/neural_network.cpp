#include "neural_network.h"

#include <algorithm>

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

    // build batch (按 shuffle 后的索引顺序拷贝)
    std::vector<std::vector<double>> batch_data(B);
    std::vector<std::vector<double>> batch_target(B);
    for (int j = 0; j < B; j++) {
      batch_data[j] = data[index_pos[batch_start + j]];
      batch_target[j] = target[index_pos[batch_start + j]];
    }

    auto rc = ForwardPropagationBatch(batch_data);
    if (rc != SUCCESS) {
      return rc;
    }
    ResetGradients();
    rc = BackPropagationBatch(batch_target);
    if (rc != SUCCESS) {
      return rc;
    }
    rc = ApplyGradient(B);
    if (rc != SUCCESS) {
      return rc;
    }

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

  loss_function_ = LossFactory::Create(option.loss_type_);
  activate_function_ = ActivateFactory::Create(option.activate_type_);
  softmax_function_ = SoftmaxFactory::Create(option.softmax_type_);
  param_init_function_ = ParamInitFactory::Create(PARAM_INIT_ZERO);
  optimizer_function_ = OptimizerFactory::Create(option.optimizer_type_, layer_);

  // batch buffer 留到第一次 forward/train 时按需 resize
  int L = (int)layer_.size();
  neuron_output_.assign(L, {});
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
  grad_bias_ = old.grad_bias_;
  grad_weight_ = old.grad_weight_;
  batch_buffer_size_ = old.batch_buffer_size_;
  learning_rate_ = old.learning_rate_;
  rand_seed_ = old.rand_seed_;
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

void NeuralNetwork::set_random_seed(int seed) { rand_seed_ = seed; }

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
  for (int l = 0; l < L; l++) {
    neuron_output_[l].assign(batch_size, std::vector<double>(layer_[l], 0.0));
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

NeuralNetwork::RC NeuralNetwork::ForwardPropagationBatch(
    const std::vector<std::vector<double>> &batch_data) {
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

  // 1..L-1: 全连接前向 + 激活
  for (int l = 1; l < L; l++) {
    int out_dim = layer_[l];
    int in_dim = layer_[l - 1];
    for (int b = 0; b < B; b++) {
      const auto &in_vec = neuron_output_[l - 1][b];
      auto &out_vec = neuron_output_[l][b];
      for (int o = 0; o < out_dim; o++) {
        double z = neuron_bias_[l][o];
        const auto &w_row = neuron_weight_[l][o];
        for (int i = 0; i < in_dim; i++) {
          z += w_row[i] * in_vec[i];
        }
        out_vec[o] = activate_function_->Activate(z);
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
        delta = dL * activate_function_->DerivActivate(neuron_output_[last][b][o]);
      }
      neuron_delta_[last][b][o] = delta;
    }
    // 2) 反向传播到隐藏层 (layer 0 不需要)
    for (int l = last - 1; l >= 1; l--) {
      int dim_l = layer_[l];
      int dim_lp1 = layer_[l + 1];
      for (int o = 0; o < dim_l; o++) {
        double sum = 0.0;
        for (int k = 0; k < dim_lp1; k++) {
          sum += neuron_weight_[l + 1][k][o] * neuron_delta_[l + 1][b][k];
        }
        neuron_delta_[l][b][o] =
            sum * activate_function_->DerivActivate(neuron_output_[l][b][o]);
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

  for (int l = 1; l < L; l++) {
    int dim_l = layer_[l];
    int dim_lm1 = layer_[l - 1];
    for (int o = 0; o < dim_l; o++) {
      double avg_db = grad_bias_[l][o] * inv_bs;
      double bias_change = optimizer_function_->CalcChangeValue(
          avg_db, learning_rate_, {l, o});
      neuron_bias_[l][o] -= bias_change;

      auto &w_row = neuron_weight_[l][o];
      const auto &g_row = grad_weight_[l][o];
      for (int i = 0; i < dim_lm1; i++) {
        double avg_dw = g_row[i] * inv_bs;
        double w_change = optimizer_function_->CalcChangeValue(
            avg_dw, learning_rate_, {l, o}, i);
        w_row[i] -= w_change;
      }
    }
  }
  return SUCCESS;
}

} // namespace deeplearning

#pragma once
#include "activate/activate_factory.h"
#include "loss/loss_factory.h"
#include "param_init/param_init_factory.h"
#include "softmax/softmax_factory.h"
#include "util/random.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>
namespace deeplearning {

class NeuralNetwork {
public:
  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
  };
  enum NetworkStatus {
    NETWORK_STATUS_UNINIT,
    NETWORK_STATUS_INIT,
  };
  struct NetworkParam {
    std::vector<int> layer_;
    std::vector<std::vector<double>> neuron_bias_;
    std::vector<std::vector<std::vector<double>>> neuron_weight_;
  };
  struct NetworkOption {
    double learning_rate_;
    int rand_seed_;
    LossType loss_type_;
    ActivateType activate_type_;
    SoftmaxType softmax_type_;
  };

public:
  NeuralNetwork() = default;
  ~NeuralNetwork() = default;
  NeuralNetwork(const NeuralNetwork &) = delete;
  NeuralNetwork &operator=(const NeuralNetwork &) = delete;

  NeuralNetwork(const std::vector<int> &layer) { Init(layer); }

  RC Init(const std::vector<int> &layer) {
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
    param_init_function_->InitParam(neuron_weight_, neuron_bias_);

    network_status_ = NETWORK_STATUS_INIT;
    return SUCCESS;
  }

  RC Train(const std::vector<std::vector<double>> &data,
           const std::vector<std::vector<double>> &target,
           std::function<void(NeuralNetwork &network, int epoch_num)>
               each_epoch_call = nullptr,
           int epoch_num = 0, int batch_num = 1, double learning_rate = 0) {
    if (network_status_ != NETWORK_STATUS_INIT) {
      err_msg_ = "[NeuralNetwork::Train] Network not init";
      return NOT_INIT;
    }
    if (data.size() != target.size() || batch_num <= 0) {
      err_msg_ = "[NeuralNetwork::Train] Invalid data input in size";
      return INVALID_DATA;
    }

    // init learning_rate
    if (learning_rate != 0) {
      learning_rate_ = learning_rate;
    } else {
      learning_rate_ = (learning_rate_ != 0) ? learning_rate_ : 0.1;
    }

    // init batch random
    std::vector<int> index_pos(data.size());
    for (int i = 0; i < data.size(); i++) {
      index_pos[i] = i;
    }
    Random::RandomShuffle(index_pos);
    auto max_batch_num = data.size() / batch_num;

    epoch_num = epoch_num == 0 ? data.size() : epoch_num;
    for (int i = 0; i < epoch_num; i++) {

      auto init_batch_num = (i % max_batch_num) * batch_num;
      if (i % max_batch_num == 0) {
        Random::RandomShuffle(index_pos);
      }

      for (int j = (i % max_batch_num) * batch_num;
           j < init_batch_num + batch_num; j += 1) {
        auto data_pos = j;
        auto rc = ForwardPropagation(data[data_pos]);
        if (rc != SUCCESS) {
          return rc;
        }
        rc = BackPropagation(data[data_pos], target[data_pos]);
        if (rc != SUCCESS) {
          return rc;
        }

        // update neuron
        rc = UpdateAllNeuron(batch_num);
        if (rc != SUCCESS) {
          return rc;
        }
      }

      // callback
      if (each_epoch_call != nullptr) {
        each_epoch_call(*this, i);
      }
    }
    return SUCCESS;
  }

  RC Predict(const std::vector<double> &data, std::vector<double> &result) {
    auto rc = ForwardPropagation(data);
    if (rc != SUCCESS) {
      return rc;
    }
    result = neuron_output_[layer_.size() - 1];
    return SUCCESS;
  }

  RC CalcLoss(const std::vector<std::vector<double>> &data,
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
    for (int i = 0; i < data.size(); i++) {
      auto rc = ForwardPropagation(data[i]);
      if (rc != SUCCESS) {
        return rc;
      }
      loss_sum += loss_function_->AverageLoss(
          target[i], neuron_output_[layer_.size() - 1]);
    }
    loss = loss_sum / data.size();
    return SUCCESS;
  }

  RC ExportNetworkParam(NetworkParam &param, NetworkOption &option) {
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
    return SUCCESS;
  }

  RC ImportNetworkParam(const NetworkParam &param,
                        const NetworkOption &option) {
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

    for (int i = 0; i < layer_.size(); i++) {
      neuron_output_.push_back(std::vector<double>(layer_[i], 0));
      neuron_delta_.push_back(std::vector<double>(layer_[i], 0));
    }

    network_status_ = NETWORK_STATUS_INIT;
    return SUCCESS;
  }

  RC Clone(const NeuralNetwork &old) {
    if (network_status_ != NETWORK_STATUS_UNINIT) {
      err_msg_ = "[NeuralNetwork::Clone] Network has init";
      return ALREADY_INIT;
    }

    layer_ = old.layer_;
    neuron_bias_ = old.neuron_bias_;
    neuron_weight_ = old.neuron_weight_;
    neuron_delta_ = old.neuron_delta_;
    neuron_output_ = old.neuron_output_;
    learning_rate_ = old.learning_rate_;
    rand_seed_ = old.rand_seed_;
    network_status_ = old.network_status_;

    loss_function_ = LossFactory::Create(old.loss_function_->GetLossType());
    activate_function_ =
        ActivateFactory::Create(old.activate_function_->GetActivateType());
    softmax_function_ =
        SoftmaxFactory::Create(old.softmax_function_->GetSoftmaxType());

    network_status_ = NETWORK_STATUS_INIT;
    return SUCCESS;
  }

public:
  inline std::string err_msg() { return err_msg_; }
  inline double learning_rate() { return learning_rate_; }
  inline int rand_seed() { return rand_seed_; }
  inline NetworkStatus network_status() { return network_status_; }
  inline const std::vector<std::vector<std::vector<double>>> &neuron_weight() {
    return neuron_weight_;
  }
  inline const std::vector<std::vector<double>> &neuron_bias() {
    return neuron_bias_;
  }
  inline RC set_loss_function(LossType type) {
    loss_function_ = LossFactory::Create(type);
    if (loss_function_ == nullptr) {
      err_msg_ = "[NeuralNetwork::set_loss_function] Invalid loss type";
      return INVALID_DATA;
    }
    return SUCCESS;
  }
  inline RC set_activate_function(ActivateType type) {
    activate_function_ = ActivateFactory::Create(type);
    if (activate_function_ == nullptr) {
      err_msg_ = "[NeuralNetwork::set_activate_function] Invalid loss type";
      return INVALID_DATA;
    }
    return SUCCESS;
  }
  inline RC set_softmax_function(SoftmaxType type) {
    softmax_function_ = SoftmaxFactory::Create(type);
    if (softmax_function_ == nullptr) {
      err_msg_ = "[NeuralNetwork::set_softmax_function] Invalid loss type";
      return INVALID_DATA;
    }
    return SUCCESS;
  }
  inline RC set_param_init_function(ParamInitType type) {
    param_init_function_ = ParamInitFactory::Create(type);
    if (param_init_function_ == nullptr) {
      err_msg_ = "[NeuralNetwork::set_param_init_function] Invalid loss type";
      return INVALID_DATA;
    }
    param_init_function_->InitParam(neuron_weight_, neuron_bias_);
    return SUCCESS;
  }
  inline void set_learning_rate(double rate) { learning_rate_ = rate; }
  inline void set_random_seed(int seed) { rand_seed_ = seed; }

private:
  double CalcDelta(const double deriv_target, const double out) {
    return deriv_target * activate_function_->DerivActivate(out);
  }

  void InitParamWithLayer(const std::vector<int> &layer) {
    layer_ = layer;
    neuron_output_.resize(layer.size());
    neuron_delta_.resize(layer.size());
    neuron_bias_.resize(layer.size());
    neuron_weight_.resize(layer.size());

    for (int i = 0; i < layer.size(); i++) {
      for (int j = 0; j < layer[i]; j++) {
        neuron_bias_[i].push_back(0);
        neuron_delta_[i].push_back(0);
        neuron_output_[i].push_back(0);
        if (i != 0) {
          neuron_weight_[i].push_back(std::vector<double>(layer[i - 1], 0));
        }
      }
    }
  }

  RC UpdateNeuronOutput(const std::pair<int, int> &neuron_pos,
                        const std::vector<double> &input) {
    auto [x, y] = neuron_pos;
    double result = neuron_bias_[x][y];
    if (x >= layer_.size() || x < 0 || y >= layer_[x] || y < 0) {
      err_msg_ = "[NeuralNetwork::UpdateNeuronOutput] Invalid data input";
      return INVALID_DATA;
    }
    if (x == 0) {
      neuron_output_[x][y] = input[y];
      return SUCCESS;
    }
    for (int i = 0; i < layer_[x - 1]; i++) {
      result += neuron_weight_[x][y][i] * neuron_output_[x - 1][i];
    }
    result = activate_function_->Activate(result);
    neuron_output_[x][y] = result;
    return SUCCESS;
  }

  void ClearNeuronDelta() {
    for (int i = 0; i < layer_.size(); i++) {
      for (int j = 0; j < layer_[i]; j++) {
        neuron_delta_[i][j] = 0;
      }
    }
  }

  RC UpdateNeuronDelta(const std::pair<int, int> &neuron_pos,
                       const std::vector<double> &target) {
    auto [x, y] = neuron_pos;
    double result = 0;
    if (x < 0 || x >= layer_.size() || y < 0 || y >= layer_[x]) {
      err_msg_ = "[NeuralNetwork::UpdateNeuronDelta] Invalid data input";
      return INVALID_DATA;
    }
    double deriv_target = 0;
    if (x == layer_.size() - 1) {
      deriv_target =
          (double)(loss_function_->DerivLoss(target[y], neuron_output_[x][y])) /
          (double)target.size();
      result = CalcDelta(deriv_target, neuron_output_[x][y]);
    } else {
      for (int i = 0; i < layer_[x + 1]; i++) {
        deriv_target += neuron_weight_[x + 1][i][y] * neuron_delta_[x + 1][i];
      }
      result = CalcDelta(deriv_target, neuron_output_[x][y]);
    }
    neuron_delta_[x][y] = result;
    return SUCCESS;
  }

  RC UpdateAllNeuron(int batch_num = 1) {
    if (layer_.size() == 0) {
      err_msg_ = "[NeuralNetwork::UpdateAllNeuron] Invalid data input";
      return INVALID_DATA;
    }
    for (int i = 0; i < layer_.size(); i++) {
      for (int j = 0; j < layer_[i]; j++) {
        auto rc = UpdateSingleNeuron({i, j}, batch_num);
        if (rc != SUCCESS) {
          return rc;
        }
      }
    }
    return SUCCESS;
  }

  RC UpdateSingleNeuron(const std::pair<int, int> &neuron_pos,
                        int batch_num = 1) {
    auto [x, y] = neuron_pos;
    if (x >= layer_.size() || x < 0 || y >= layer_[x] || y < 0) {
      err_msg_ = "[NeuralNetwork::UpdateNeuron] Invalid data input";
      return INVALID_DATA;
    }
    if (x == 0) {
      return SUCCESS;
    }
    double averg_delta = neuron_delta_[x][y] / (double)batch_num;
    for (int i = 0; i < layer_[x - 1]; i++) {
      neuron_weight_[x][y][i] -=
          learning_rate_ * neuron_output_[x - 1][i] * averg_delta;
    }
    neuron_bias_[x][y] -= learning_rate_ * averg_delta;
    return SUCCESS;
  }

  RC ForwardPropagation(const std::vector<double> &data) {
    if (layer_.size() == 0 || data.size() != layer_[0]) {
      err_msg_ = "[NeuralNetwork::ForwardPropagation] Invalid data input";
      return INVALID_DATA;
    }
    for (int i = 0; i < layer_.size(); i++) {
      for (int j = 0; j < layer_[i]; j++) {
        auto rc = UpdateNeuronOutput({i, j}, data);
        if (rc != SUCCESS) {
          return rc;
        }
      }
    }
    // calc softmax
    softmax_function_->Normalize(neuron_output_[layer_.size() - 1],
                                 neuron_output_[layer_.size() - 1]);
    return SUCCESS;
  }

  RC BackPropagation(const std::vector<double> &input,
                     const std::vector<double> &target) {
    if (layer_.size() == 0 || target.size() != layer_[layer_.size() - 1]) {
      err_msg_ = "[NeuralNetwork::BackPropagation] Invalid data input";
      return INVALID_DATA;
    }
    // ForwardPropagation has run before
    for (int i = layer_.size() - 1; i >= 0; i--) {
      for (int j = 0; j < layer_[i]; j++) {
        auto rc = UpdateNeuronDelta({i, j}, target);
        if (rc != SUCCESS) {
          return rc;
        }
      }
    }
    return SUCCESS;
  }

private:
  std::shared_ptr<LossFunction> loss_function_ = nullptr;
  std::shared_ptr<ActivateFunction> activate_function_ = nullptr;
  std::shared_ptr<SoftmaxFunction> softmax_function_ = nullptr;
  std::shared_ptr<ParamInitFunction> param_init_function_ = nullptr;

  NetworkStatus network_status_ = NETWORK_STATUS_UNINIT;
  int rand_seed_ = 0;
  double learning_rate_ = 0.1;
  std::vector<int> layer_;
  std::vector<std::vector<double>> neuron_bias_;
  std::vector<std::vector<std::vector<double>>> neuron_weight_;
  std::vector<std::vector<double>> neuron_output_;
  std::vector<std::vector<double>> neuron_delta_;
  std::string err_msg_;
};

} // namespace deeplearning

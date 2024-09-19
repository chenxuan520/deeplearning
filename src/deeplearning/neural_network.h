#pragma once
#include "activate/activate_factory.h"
#include "loss/loss_factory.h"
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
  };

public:
  NeuralNetwork() = default;
  NeuralNetwork(const std::vector<int> &layer, double learning_rate = 0.1) {
    Init(layer, learning_rate);
  }

  RC Init(const std::vector<int> &layer, double learning_rate = 0.1) {
    if (layer.size() < 2) {
      err_msg_ = "[NeuralNetwork::InitNetwork] Invalid layer size";
      return INVALID_DATA;
    }

    learning_rate_ = learning_rate;
    layer_ = layer;
    neuron_output_.resize(layer.size());
    neuron_delta_.resize(layer.size());
    neuron_bias_.resize(layer.size());
    neuron_weight_.resize(layer.size());
    loss_function_ = LossFactory::Create(LOSS_MSE);
    activate_function_ = ActivateFactory::Create(ACTIVATE_SIGMOID);
    rand_seed_ = 0;

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

    network_status_ = NETWORK_STATUS_INIT;
    return SUCCESS;
  }

  RC Train(const std::vector<std::vector<double>> &data,
           const std::vector<std::vector<double>> &target, int epoch_num = 1,
           int batch_num = 1,
           std::function<void(const NeuralNetwork &network, int epoch_num,
                              double train_loss)>
               each_train_end_call = nullptr) {
    if (network_status_ != NETWORK_STATUS_INIT) {
      err_msg_ = "[NeuralNetwork::Train] Network not init";
      return NOT_INIT;
    }
    if (data.size() != target.size() || batch_num < 1) {
      err_msg_ = "[NeuralNetwork::Train] Invalid data input in size";
      return INVALID_DATA;
    }

    // init batch random
    Random rand(0, batch_num, rand_seed_);

    for (int i = 0; i < epoch_num; i++) {
      for (int j = 0; j < data.size(); j += batch_num) {
        auto temp_pos = j + rand.CreateRandom();
        auto data_pos = temp_pos < data.size() ? temp_pos : j;
        auto rc = ForwardPropagation(data[data_pos]);
        if (rc != SUCCESS) {
          return rc;
        }
        rc = BackPropagation(data[data_pos], target[data_pos]);
        if (rc != SUCCESS) {
          return rc;
        }
        if (each_train_end_call != nullptr) {
          each_train_end_call(
              *this, i,
              loss_function_->AverageLoss(target[data_pos],
                                          neuron_output_[layer_.size() - 1]));
        }
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
    return SUCCESS;
  }

  RC ImportNetworkParam(const NetworkParam &param,
                        const NetworkOption &option) {
    if (network_status_ != NETWORK_STATUS_UNINIT) {
      err_msg_ = "[NeuralNetwork::ImportNetworkParam] Network has init";
      return NOT_INIT;
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

    network_status_ = NETWORK_STATUS_INIT;
    return SUCCESS;
  }

public:
  inline std::string err_msg() { return err_msg_; }
  inline double learning_rate() { return learning_rate_; }
  inline int rand_seed() { return rand_seed_; }
  inline NetworkStatus network_status() { return network_status_; }
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
  inline void set_learning_rate(double rate) { learning_rate_ = rate; }
  inline void set_random_seed(int seed) { rand_seed_ = seed; }

private:
  double CalcDelta(const double deriv_target, const double out) {
    return deriv_target * activate_function_->DerivActivate(out);
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

  RC UpdateNeuron(const std::pair<int, int> &neuron_pos) {
    auto [x, y] = neuron_pos;
    if (x >= layer_.size() || x < 0 || y >= layer_[x] || y < 0) {
      err_msg_ = "[NeuralNetwork::UpdateNeuron] Invalid data input";
      return INVALID_DATA;
    }
    if (x == 0) {
      return SUCCESS;
    }
    for (int i = 0; i < layer_[x - 1]; i++) {
      neuron_weight_[x][y][i] -=
          learning_rate_ * neuron_output_[x - 1][i] * neuron_delta_[x][y];
    }
    neuron_bias_[x][y] -= learning_rate_ * neuron_delta_[x][y];
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
    for (int i = 0; i < layer_.size(); i++) {
      for (int j = 0; j < layer_[i]; j++) {
        auto rc = UpdateNeuron({i, j});
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

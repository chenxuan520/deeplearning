#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class LossFunction {
public:
  virtual double AverageLoss(const std::vector<double> &target,
                             const std::vector<double> &output) = 0;
  virtual double Loss(double target, double output) = 0;
  virtual double DerivLoss(double target, double output) = 0;
};

class MSELoss : public LossFunction {
public:
  double AverageLoss(const std::vector<double> &target,
                     const std::vector<double> &output) {
    double result = 0;
    if (target.size() != output.size()) {
      return -1;
    }
    for (int i = 0; i < target.size(); i++) {
      result += Loss(target[i], output[i]);
    }
    result /= target.size();
    return result;
  }
  double Loss(double target, double output) {
    return (target - output) * (target - output);
  }
  double DerivLoss(double target, double output) {
    return -2 * (target - output);
  }
};

class ActivateFunction {
public:
  virtual double Activate(const double &input) = 0;
  virtual double DerivActivate(const double &output) = 0;
};

class SigmoidActivate : public ActivateFunction {
public:
  double Activate(const double &input) { return 1 / (1 + exp(-input)); }
  double DerivActivate(const double &output) { return output * (1 - output); }
};

class NeuralNetwork {
public:
  enum RC {
    SUCCESS,
    INVALID_DATA,
  };

public:
  NeuralNetwork(const std::vector<int> &layer, double learning_rate = 0.1)
      : layer_(layer), learning_rate_(learning_rate) {
    neuron_output_.resize(layer.size());
    neuron_delta_.resize(layer.size());
    neuron_bias_.resize(layer.size());
    neuron_weight_.resize(layer.size());
    loss_function_ = std::make_shared<MSELoss>();
    activate_function_ = std::make_shared<SigmoidActivate>();

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
  RC Train(const std::vector<std::vector<double>> &data,
           const std::vector<std::vector<double>> &target, int train_time = 1,
           int batch_num = 1,
           std::function<void(double train_loss)> train_end_call = nullptr) {
    if (data.size() != target.size()) {
      err_msg_ = "[NeuralNetwork::Train] Invalid data input";
      return INVALID_DATA;
    }
    for (int i = 0; i < train_time; i++) {
      for (int j = 0; j < data.size(); j++) {
        auto rc = ForwardPropagation(data[j]);
        if (rc != SUCCESS) {
          return rc;
        }
        rc = BackPropagation(data[j], target[j]);
        if (rc != SUCCESS) {
          return rc;
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

public:
  inline std::string err_msg() { return err_msg_; }

private:
  double CalcDelta(const double deriv_target, const double out) {
    return deriv_target * activate_function_->DerivActivate(out);
  }

  RC UpdateNeuronOutput(const std::pair<int, int> &neuron_pos,
                        const std::vector<double> &input) {
    auto [x, y] = neuron_pos;
    double result = neuron_bias_[x][y];
    if (x >= layer_.size() || x < 0 || y >= input.size() || y >= layer_[x] ||
        y < 0) {
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
      deriv_target = loss_function_->DerivLoss(target[y], neuron_output_[x][y]);
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
  double learning_rate_;
  std::vector<int> layer_;
  std::vector<std::vector<double>> neuron_output_;
  std::vector<std::vector<double>> neuron_bias_;
  std::vector<std::vector<std::vector<double>>> neuron_weight_;
  std::vector<std::vector<double>> neuron_delta_;
  std::string err_msg_;
};

int main() {
  NeuralNetwork network((std::vector<int>() = {2, 1, 1}), 0.2);
  std::vector<std::vector<double>> data = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  std::vector<std::vector<double>> target = {{0}, {0}, {1}, {1}};
  auto rc = network.Train(data, target, 2000);
  if (rc != NeuralNetwork::SUCCESS) {
    std::cout << "Train failed" << std::endl;
    return -1;
  }
  std::cout << "Train success" << std::endl;
  std::vector<double> test_data = {1, 1.2}, result;
  rc = network.Predict(test_data, result);
  if (rc != NeuralNetwork::SUCCESS) {
    std::cout << "Predict failed" << std::endl;
    return -1;
  }
  std::cout << "Predict: " << result[0] << std::endl;
  return 0;
}

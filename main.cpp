#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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
           const std::vector<std::vector<double>> &target, int train_time) {
    if (data.size() != target.size()) {
      err_msg_ = "[NeuralNetwork::Train] Invalid data input";
      return INVALID_DATA;
    }
    for (int i = 0; i < train_time; i++) {
      for (int j = 0; j < data.size(); j++) {
        auto rc = BackPropagation(data[j], target[j]);
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
  inline std::string err_msg() { return err_msg_; }

private:
  double Sigmoid(const double x) {
    // 激活函数
    return 1 / (1 + exp(-x));
  }
  double DerivSigmoid(const double x) {
    // 激活函数求导
    double y = Sigmoid(x);
    return y * (1 - y);
  }
  double SumMSEloss(const std::vector<double> &target) {
    double result = 0;
    for (int i = 0; i < target.size(); i++) {
      result += MSEloss(target[i], neuron_output_[layer_.size() - 1][i]);
    }
    return result;
  }
  double MSEloss(const double x1, const double x2) {
    return ((x1 - x2) * (x1 - x2)) / 2;
  }
  double DerivMSEloss(const double x1, const double x2) { return -(x1 - x2); }
  double CalcDelta(const double deriv_target, const double out) {
    return deriv_target * out * (1 - out);
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
    result = Sigmoid(result);
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
      deriv_target = DerivMSEloss(target[y], neuron_output_[x][y]);
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
    auto rc = ForwardPropagation(input);
    if (rc != SUCCESS) {
      return rc;
    }
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
  double learning_rate_;
  std::vector<int> layer_;
  std::vector<std::vector<double>> neuron_output_;
  std::vector<std::vector<double>> neuron_bias_;
  std::vector<std::vector<std::vector<double>>> neuron_weight_;
  std::vector<std::vector<double>> neuron_delta_;
  std::string err_msg_;
};

int main() {
  NeuralNetwork network((std::vector<int>() = {2, 1, 1}), 1);
  std::vector<std::vector<double>> data = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  std::vector<std::vector<double>> target = {{0}, {0}, {1}, {1}};
  auto rc = network.Train(data, target, 1000);
  if (rc != NeuralNetwork::SUCCESS) {
    std::cout << "Train failed" << std::endl;
    return -1;
  }
  std::cout << "Train success" << std::endl;
  std::vector<double> test_data = {-1, 0}, result;
  rc = network.Predict(test_data, result);
  if (rc != NeuralNetwork::SUCCESS) {
    std::cout << "Predict failed" << std::endl;
    return -1;
  }
  std::cout << "Predict: " << result[0] << std::endl;
  return 0;
}

#pragma once
#include "neural_network.h"
#include <fstream>

namespace deeplearning {

class NeuralNetworkLoader {
public:
  enum RC {
    SUCCESS,
    EXPORT_ERROR,
    INPORT_ERROR,
  };

public:
  static RC ExportParamToFile(const NeuralNetwork::NetworkParam &param,
                              const std::string &filename) {
    std::ofstream ofs(filename, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) {
      return INPORT_ERROR;
    }
    ParamSizeMsg msg(param);
    auto is_success = ofs.write((const char *)&msg, sizeof(msg)).good();
    if (!is_success) {
      return EXPORT_ERROR;
    }
    // write layer
    for (int i = 0; i < param.layer_.size(); i++) {
      auto is_success =
          ofs.write((const char *)&param.layer_[i], sizeof(int)).good();
      if (!is_success) {
        return EXPORT_ERROR;
      }
    }
    // write neuron bias
    for (int i = 0; i < param.neuron_bias_.size(); i++) {
      for (int j = 0; j < param.neuron_bias_[i].size(); j++) {
        auto is_success =
            ofs.write((const char *)&param.neuron_bias_[i][j], sizeof(double))
                .good();
        if (!is_success) {
          return EXPORT_ERROR;
        }
      }
    }
    // write neuron weight
    for (int i = 0; i < param.neuron_weight_.size(); i++) {
      for (int j = 0; j < param.neuron_weight_[i].size(); j++) {
        for (int k = 0; k < param.neuron_weight_[i][j].size(); k++) {
          auto is_success =
              ofs.write((const char *)&param.neuron_weight_[i][j][k],
                        sizeof(double))
                  .good();
          if (!is_success) {
            return EXPORT_ERROR;
          }
        }
      }
    }

    ofs.close();
    return SUCCESS;
  }

  static RC ImportParamFromFile(NeuralNetwork::NetworkParam &param,
                                const std::string &filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
      return INPORT_ERROR;
    }
    ParamSizeMsg msg;
    auto is_success = ifs.read((char *)&msg, sizeof(msg)).good();
    if (!is_success) {
      return INPORT_ERROR;
    }

    // read other
    param.rand_seed_ = msg.rand_seed_;
    param.learning_rate_ = msg.learning_rate_;

    // read layer
    param.layer_.resize(msg.layer_size_);
    for (int i = 0; i < msg.layer_size_; i++) {
      auto is_success = ifs.read((char *)&param.layer_[i], sizeof(int)).good();
      if (!is_success) {
        return INPORT_ERROR;
      }
    }
    // read neuron bias
    param.neuron_bias_.resize(msg.neuron_bias_size_);
    for (int i = 0; i < msg.neuron_bias_size_; i++) {
      param.neuron_bias_[i].resize(param.layer_[i]);
      for (int j = 0; j < param.layer_[i]; j++) {
        auto is_success =
            ifs.read((char *)&param.neuron_bias_[i][j], sizeof(double)).good();
        if (!is_success) {
          return INPORT_ERROR;
        }
      }
    }
    // read neuron weight
    param.neuron_weight_.resize(msg.neuron_weight_size_);
    for (int i = 1; i < msg.neuron_weight_size_; i++) {
      param.neuron_weight_[i].resize(param.layer_[i]);
      for (int j = 0; j < param.layer_[i]; j++) {
        param.neuron_weight_[i][j].resize(param.layer_[i - 1]);
        for (int k = 0; k < param.layer_[i - 1]; k++) {
          auto is_success =
              ifs.read((char *)&param.neuron_weight_[i][j][k], sizeof(double))
                  .good();
          if (!is_success) {
            return INPORT_ERROR;
          }
        }
      }
    }
    ifs.close();
    return SUCCESS;
  }

private:
  struct ParamSizeMsg {
    double learning_rate_;
    int rand_seed_;
    int layer_size_;
    int neuron_bias_size_;
    int neuron_weight_size_;
    ParamSizeMsg() = default;
    ParamSizeMsg(const NeuralNetwork::NetworkParam &param) {
      rand_seed_ = param.rand_seed_;
      learning_rate_ = param.learning_rate_;
      layer_size_ = param.layer_.size();
      neuron_bias_size_ = param.neuron_bias_.size();

      neuron_weight_size_ = param.neuron_weight_.size();
    }
  };
};

} // namespace deeplearning

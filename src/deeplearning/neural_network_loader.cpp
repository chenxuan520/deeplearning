#include "neural_network_loader.h"

namespace deeplearning {

NeuralNetworkLoader::RC NeuralNetworkLoader::ExportParamToFile(
    const NeuralNetwork::NetworkParam &param,
    const NeuralNetwork::NetworkOption &option, const std::string &filename) {
  std::ofstream ofs(filename, std::ios::binary | std::ios::trunc);
  if (!ofs.is_open()) {
    return INPORT_ERROR;
  }

  auto is_success = ofs.write((const char *)&option, sizeof(option)).good();
  if (!is_success) {
    ofs.close();
    return EXPORT_ERROR;
  }

  ParamSizeMsg msg(param);
  is_success = ofs.write((const char *)&msg, sizeof(msg)).good();
  if (!is_success) {
    ofs.close();
    return EXPORT_ERROR;
  }

  for (int i = 0; i < param.layer_.size(); i++) {
    is_success = ofs.write((const char *)&param.layer_[i], sizeof(int)).good();
    if (!is_success) {
      ofs.close();
      return EXPORT_ERROR;
    }
  }

  for (int i = 0; i < param.neuron_bias_.size(); i++) {
    for (int j = 0; j < param.neuron_bias_[i].size(); j++) {
      is_success =
          ofs.write((const char *)&param.neuron_bias_[i][j], sizeof(double)).good();
      if (!is_success) {
        ofs.close();
        return EXPORT_ERROR;
      }
    }
  }

  for (int i = 1; i < param.neuron_weight_.size(); i++) {
    for (int j = 0; j < param.neuron_weight_[i].size(); j++) {
      for (int k = 0; k < param.neuron_weight_[i][j].size(); k++) {
        is_success = ofs.write((const char *)&param.neuron_weight_[i][j][k],
                               sizeof(double))
                         .good();
        if (!is_success) {
          ofs.close();
          return EXPORT_ERROR;
        }
      }
    }
  }

  ofs.close();
  return SUCCESS;
}

NeuralNetworkLoader::RC NeuralNetworkLoader::ImportParamFromFile(
    NeuralNetwork::NetworkParam &param, NeuralNetwork::NetworkOption &option,
    const std::string &filename) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs.is_open()) {
    return INPORT_ERROR;
  }

  auto is_success = ifs.read((char *)&option, sizeof(option)).good();
  if (!is_success) {
    ifs.close();
    return INPORT_ERROR;
  }

  ParamSizeMsg msg;
  is_success = ifs.read((char *)&msg, sizeof(msg)).good();
  if (!is_success) {
    ifs.close();
    return INPORT_ERROR;
  }

  param.layer_.resize(msg.layer_size_);
  for (int i = 0; i < msg.layer_size_; i++) {
    is_success = ifs.read((char *)&param.layer_[i], sizeof(int)).good();
    if (!is_success) {
      ifs.close();
      return INPORT_ERROR;
    }
  }

  param.neuron_bias_.resize(msg.neuron_bias_size_);
  for (int i = 0; i < msg.neuron_bias_size_; i++) {
    param.neuron_bias_[i].resize(param.layer_[i]);
    for (int j = 0; j < param.layer_[i]; j++) {
      is_success = ifs.read((char *)&param.neuron_bias_[i][j], sizeof(double)).good();
      if (!is_success) {
        ifs.close();
        return INPORT_ERROR;
      }
    }
  }

  param.neuron_weight_.resize(msg.neuron_weight_size_);
  for (int i = 1; i < msg.neuron_weight_size_; i++) {
    param.neuron_weight_[i].resize(param.layer_[i]);
    for (int j = 0; j < param.layer_[i]; j++) {
      param.neuron_weight_[i][j].resize(param.layer_[i - 1]);
      for (int k = 0; k < param.layer_[i - 1]; k++) {
        is_success = ifs.read((char *)&param.neuron_weight_[i][j][k], sizeof(double))
                         .good();
        if (!is_success) {
          ifs.close();
          return INPORT_ERROR;
        }
      }
    }
  }

  ifs.close();
  return SUCCESS;
}

NeuralNetworkLoader::ParamSizeMsg::ParamSizeMsg() = default;

NeuralNetworkLoader::ParamSizeMsg::ParamSizeMsg(
    const NeuralNetwork::NetworkParam &param) {
  layer_size_ = param.layer_.size();
  neuron_bias_size_ = param.neuron_bias_.size();
  neuron_weight_size_ = param.neuron_weight_.size();
}

} // namespace deeplearning

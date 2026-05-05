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
                              const NeuralNetwork::NetworkOption &option,
                              const std::string &filename);

  static RC ImportParamFromFile(NeuralNetwork::NetworkParam &param,
                                NeuralNetwork::NetworkOption &option,
                                const std::string &filename);

private:
  struct ParamSizeMsg {
    double learning_rate_;
    int rand_seed_;
    int layer_size_;
    int neuron_bias_size_;
    int neuron_weight_size_;
    ParamSizeMsg();
    ParamSizeMsg(const NeuralNetwork::NetworkParam &param);
  };
};

} // namespace deeplearning

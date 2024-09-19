#pragma once

#include "../deeplearning/neural_network_loader.h"
#include "test.h"
#include <cstdlib>

using namespace std;
using namespace deeplearning;

NeuralNetwork::NetworkParam demo_param = {};
NeuralNetwork::NetworkOption demo_option = {};

INIT(LoaderParam) {
  NeuralNetwork network((vector<int>() = {1, 2, 3}), 0.3);
  auto rc = network.ExportNetworkParam(demo_param, demo_option);
  if (rc != NeuralNetwork::SUCCESS) {
    DEBUG("Export failed");
    exit(-1);
  }
}

TEST(Loader, ExportAndInport) {
  const string file_path = "demo.param";
  // delete file
  DEFER([=]() { remove(file_path.c_str()); });

  auto rc = deeplearning::NeuralNetworkLoader::ExportParamToFile(demo_param, demo_option,
                                                   file_path);
  MUST_EQUAL(rc, NeuralNetworkLoader::SUCCESS);

  NeuralNetwork::NetworkParam param;
  NeuralNetwork::NetworkOption option;
  rc = NeuralNetworkLoader::ImportParamFromFile(param, option, file_path);
  MUST_EQUAL(rc, NeuralNetworkLoader::SUCCESS);

  MUST_EQUAL(option.rand_seed_, demo_option.rand_seed_);
  MUST_EQUAL(option.learning_rate_, demo_option.learning_rate_);
  MUST_EQUAL(option.activate_type_, demo_option.activate_type_);
  MUST_EQUAL(option.loss_type_, demo_option.loss_type_);

  MUST_EQUAL(param.layer_.size(), demo_param.layer_.size());
  for (int i = 0; i < param.layer_.size(); i++) {
    MUST_EQUAL(param.layer_[i], demo_param.layer_[i]);
  }

  MUST_EQUAL(param.neuron_bias_.size(), demo_param.neuron_bias_.size());
  for (int i = 0; i < param.neuron_bias_.size(); i++) {
    MUST_EQUAL(param.neuron_bias_[i].size(), demo_param.neuron_bias_[i].size());
    for (int j = 0; j < param.neuron_bias_[i].size(); j++) {
      MUST_EQUAL(param.neuron_bias_[i][j], demo_param.neuron_bias_[i][j]);
    }
  }

  MUST_EQUAL(param.neuron_weight_.size(), demo_param.neuron_weight_.size());
  for (int i = 0; i < param.neuron_weight_.size(); i++) {
    MUST_EQUAL(param.neuron_weight_[i].size(),
               demo_param.neuron_weight_[i].size());
    for (int j = 0; j < param.neuron_weight_[i].size(); j++) {
      MUST_EQUAL(param.neuron_weight_[i][j].size(),
                 demo_param.neuron_weight_[i][j].size());
      for (int k = 0; k < param.neuron_weight_[i][j].size(); k++) {
        MUST_EQUAL(param.neuron_weight_[i][j][k],
                   demo_param.neuron_weight_[i][j][k]);
      }
    }
  }
}

#pragma once

#include "../deeplearning/neural_network_loader.h"
#include "test.h"
#include <cstdlib>
#include <vector>

using namespace std;
using namespace deeplearning;

NeuralNetwork::NetworkParam demo_param = {};

INIT(LoaderParam) {
  NeuralNetwork network((vector<int>() = {1, 2, 3}), 0.3);
  auto rc = network.ExportNetworkParam(demo_param);
  if (rc != NeuralNetwork::SUCCESS) {
    DEBUG("Export failed");
    exit(-1);
  }
}

TEST(Loader, ExportAndInport) {
  const string file_path = "demo.param";
  // delete file
  DEFER([=]() { remove(file_path.c_str()); });

  auto rc = NeuralNetworkLoader::ExportParamToFile(demo_param, file_path);
  MUST_EQUAL(rc, NeuralNetworkLoader::SUCCESS);

  NeuralNetwork::NetworkParam param;
  rc = NeuralNetworkLoader::ImportParamFromFile(param, file_path);
  MUST_EQUAL(rc, NeuralNetworkLoader::SUCCESS);

  MUST_EQUAL(param.rand_seed_, demo_param.rand_seed_);
  MUST_EQUAL(param.learning_rate_, demo_param.learning_rate_);

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

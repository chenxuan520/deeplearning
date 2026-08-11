#pragma once

#include "neural_network.h"
#include "optimizer/adam_optimizer.h"
#include "test.h"
#include "transformer/mini_transformer_lm.h"
#include "transformer/mini_transformer_lm_loader.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace model_expansion_test_detail {

using namespace deeplearning;

inline bool Near(double lhs, double rhs, double eps = 1e-9) {
  return std::fabs(lhs - rhs) <= eps;
}

inline NeuralNetwork::RC BuildKnownMlp(NeuralNetwork &network) {
  NeuralNetwork::NetworkParam param;
  param.layer_ = {2, 2, 2};
  param.neuron_bias_ = {{0.0, 0.0}, {0.2, -0.1}, {0.3, -0.4}};
  param.neuron_weight_ = {
      {},
      {{0.5, -0.25}, {0.75, 0.4}},
      {{0.8, -0.6}, {-0.3, 0.9}},
  };
  NeuralNetwork::NetworkOption option = {
      0.01, 17, LOSS_MSE, ACTIVATE_TANH, SOFTMAX_NONE, OPTIMIZER_ADAM};
  return network.ImportNetworkParam(param, option);
}

inline bool SameMatrix(const std::vector<std::vector<double>> &lhs,
                       const std::vector<std::vector<double>> &rhs,
                       double eps = 1e-9) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (int row = 0; row < static_cast<int>(lhs.size()); row++) {
    if (lhs[row].size() != rhs[row].size()) {
      return false;
    }
    for (int col = 0; col < static_cast<int>(lhs[row].size()); col++) {
      if (!Near(lhs[row][col], rhs[row][col], eps)) {
        return false;
      }
    }
  }
  return true;
}

inline bool CopyWithoutTail(const std::string &source,
                            const std::string &destination, int tail_size) {
  std::ifstream ifs(source, std::ios::binary | std::ios::ate);
  if (!ifs.is_open()) {
    return false;
  }
  const auto file_size = ifs.tellg();
  if (file_size < tail_size) {
    return false;
  }
  std::vector<char> content(static_cast<size_t>(file_size) - tail_size);
  ifs.seekg(0);
  if (!ifs.read(content.data(), content.size()).good()) {
    return false;
  }
  std::ofstream ofs(destination, std::ios::binary | std::ios::trunc);
  return ofs.is_open() && ofs.write(content.data(), content.size()).good();
}

class TopologyResetUnsupportedOptimizer : public OptimizerFunction {
public:
  explicit TopologyResetUnsupportedOptimizer(const std::vector<int> &layer)
      : OptimizerFunction(layer) {}

  double CalcChangeValue(double delta, double learning_rate,
                         const std::pair<int, int> &, int = -1,
                         double = 0.0) override {
    return delta * learning_rate;
  }

  OptimizerType GetOptimizerType() override { return OPTIMIZER_SGD; }
};

} // namespace model_expansion_test_detail

TEST(ModelExpansion, MlpNet2WiderPreservesOutput) {
  using namespace model_expansion_test_detail;
  NeuralNetwork network;
  MUST_EQUAL(BuildKnownMlp(network), NeuralNetwork::SUCCESS);
  const std::vector<std::vector<double>> input = {
      {0.2, -0.7}, {1.1, 0.3}, {-0.4, 0.8}};
  std::vector<std::vector<double>> before;
  MUST_EQUAL(network.PredictBatch(input, before), NeuralNetwork::SUCCESS);

  MUST_EQUAL(network.WidenHiddenLayer(1, 4, NeuralNetwork::WIDEN_NET2WIDER),
             NeuralNetwork::SUCCESS);
  std::vector<std::vector<double>> after;
  MUST_EQUAL(network.PredictBatch(input, after), NeuralNetwork::SUCCESS);
  MUST_TRUE(SameMatrix(before, after, 1e-12), "MLP output changed");

  const auto &weight = network.neuron_weight();
  const auto &bias = network.neuron_bias();
  MUST_EQUAL(weight[1].size(), 4);
  MUST_EQUAL(weight[2][0].size(), 4);
  MUST_TRUE(weight[1][2] == weight[1][0], "clone 0 incoming mismatch");
  MUST_TRUE(weight[1][3] == weight[1][1], "clone 1 incoming mismatch");
  MUST_TRUE(Near(bias[1][2], bias[1][0]), "clone 0 bias mismatch");
  MUST_TRUE(Near(bias[1][3], bias[1][1]), "clone 1 bias mismatch");
  MUST_TRUE(Near(weight[2][0][0], 0.4), "source outgoing not split");
  MUST_TRUE(Near(weight[2][0][2], 0.4), "clone outgoing not split");
}

TEST(ModelExpansion, MlpZeroOutgoingResetsOptimizerAndKeepsConfig) {
  using namespace model_expansion_test_detail;
  NeuralNetwork network;
  MUST_EQUAL(network.Init({2, 3, 2}), NeuralNetwork::SUCCESS);
  network.set_random_seed(23);
  MUST_EQUAL(network.set_param_init_function(PARAM_INIT_XAVIER),
             NeuralNetwork::SUCCESS);
  MUST_EQUAL(network.set_activate_function(ACTIVATE_TANH),
             NeuralNetwork::SUCCESS);
  MUST_EQUAL(network.set_optimizer_function(OPTIMIZER_ADAM),
             NeuralNetwork::SUCCESS);
  auto adam = std::dynamic_pointer_cast<AdamOptimizer>(
      network.optimizer_function());
  MUST_TRUE(adam != nullptr, "Adam cast failed");
  adam->set_beta1(0.8);
  adam->set_weight_decay(0.03);

  const std::vector<std::vector<double>> input = {{0.2, 0.4}, {-0.3, 0.8}};
  const std::vector<std::vector<double>> target = {{0.1, 0.9}, {0.8, 0.2}};
  MUST_EQUAL(network.Train(input, target, nullptr, 2, 2, 0.01),
             NeuralNetwork::SUCCESS);
  MUST_EQUAL(adam->step(), 2);
  std::vector<std::vector<double>> before;
  MUST_EQUAL(network.PredictBatch(input, before), NeuralNetwork::SUCCESS);

  MUST_EQUAL(network.WidenHiddenLayer(
                 1, 5, NeuralNetwork::WIDEN_ZERO_OUTGOING, PARAM_INIT_HE),
             NeuralNetwork::SUCCESS);
  MUST_EQUAL(adam->step(), 0);
  MUST_TRUE(Near(adam->beta1(), 0.8), "Adam beta1 was lost");
  MUST_TRUE(Near(adam->weight_decay(), 0.03), "weight decay was lost");
  for (const auto &row : network.neuron_weight()[2]) {
    MUST_TRUE(row[3] == 0.0 && row[4] == 0.0,
              "new outgoing weights must start at zero");
  }
  std::vector<std::vector<double>> after;
  MUST_EQUAL(network.PredictBatch(input, after), NeuralNetwork::SUCCESS);
  MUST_TRUE(SameMatrix(before, after, 1e-12), "MLP output changed");
  MUST_EQUAL(network.Train(input, target, nullptr, 1, 2, 0.01),
             NeuralNetwork::SUCCESS);
  MUST_EQUAL(adam->step(), 1);
}

TEST(ModelExpansion, MlpRandomKeepsOldParameterRegion) {
  using namespace model_expansion_test_detail;
  NeuralNetwork network;
  MUST_EQUAL(BuildKnownMlp(network), NeuralNetwork::SUCCESS);
  const auto old_weight = network.neuron_weight();
  MUST_EQUAL(network.WidenHiddenLayer(1, 3, NeuralNetwork::WIDEN_RANDOM,
                                      PARAM_INIT_XAVIER),
             NeuralNetwork::SUCCESS);
  const auto &new_weight = network.neuron_weight();
  for (int neuron = 0; neuron < 2; neuron++) {
    MUST_TRUE(new_weight[1][neuron] == old_weight[1][neuron],
              "old incoming weight changed");
  }
  for (int out = 0; out < 2; out++) {
    for (int in = 0; in < 2; in++) {
      MUST_TRUE(new_weight[2][out][in] == old_weight[2][out][in],
                "old outgoing weight changed");
    }
  }
  MUST_TRUE(new_weight[2][0][2] != 0.0 || new_weight[2][1][2] != 0.0,
            "random outgoing weights were not initialized");
}

TEST(ModelExpansion, MlpRejectsUnsafeExpansion) {
  using namespace model_expansion_test_detail;
  NeuralNetwork zero_width;
  MUST_EQUAL(zero_width.Init({2, 0, 1}), NeuralNetwork::SUCCESS);
  MUST_EQUAL(zero_width.WidenHiddenLayer(
                 1, 2, NeuralNetwork::WIDEN_NET2WIDER),
             NeuralNetwork::INVALID_DATA);

  NeuralNetwork custom_optimizer;
  MUST_EQUAL(custom_optimizer.Init({2, 2, 1}), NeuralNetwork::SUCCESS);
  auto optimizer = std::make_shared<TopologyResetUnsupportedOptimizer>(
      std::vector<int>{2, 2, 1});
  MUST_EQUAL(custom_optimizer.set_optimizer_function(optimizer),
             NeuralNetwork::SUCCESS);
  const auto old_weight = custom_optimizer.neuron_weight();
  MUST_EQUAL(custom_optimizer.WidenHiddenLayer(
                 1, 3, NeuralNetwork::WIDEN_ZERO_OUTGOING),
             NeuralNetwork::INVALID_DATA);
  MUST_TRUE(custom_optimizer.neuron_weight() == old_weight,
            "failed expansion changed parameters");
}

TEST(ModelExpansion, MlpBuiltInOptimizersTrainAfterDeepWiden) {
  const std::vector<OptimizerType> optimizer_types = {
      OPTIMIZER_SGD, OPTIMIZER_MOMENTUM, OPTIMIZER_RMSPROP, OPTIMIZER_ADAMW};
  const std::vector<std::vector<double>> input = {{0.2, 0.4}, {-0.3, 0.8}};
  const std::vector<std::vector<double>> target = {{0.1, 0.9}, {0.8, 0.2}};
  for (OptimizerType type : optimizer_types) {
    NeuralNetwork network;
    MUST_EQUAL(network.Init({2, 3, 4, 2}), NeuralNetwork::SUCCESS);
    network.set_random_seed(43);
    MUST_EQUAL(network.set_param_init_function(PARAM_INIT_XAVIER),
               NeuralNetwork::SUCCESS);
    MUST_EQUAL(network.set_optimizer_function(type), NeuralNetwork::SUCCESS);
    MUST_EQUAL(network.set_dropout_rate(0.2), NeuralNetwork::SUCCESS);
    MUST_EQUAL(network.Train(input, target, nullptr, 1, 2, 0.01),
               NeuralNetwork::SUCCESS);
    MUST_EQUAL(network.WidenHiddenLayer(
                   2, 6, NeuralNetwork::WIDEN_ZERO_OUTGOING),
               NeuralNetwork::SUCCESS);
    network.set_train_thread_num(2);
    MUST_EQUAL(network.Train(input, target, nullptr, 1, 2, 0.01),
               NeuralNetwork::SUCCESS);
  }
}

TEST(ModelExpansion, TransformerVocabularyMappingPreservesOldLogits) {
  using namespace model_expansion_test_detail;
  MiniTransformerLM model;
  model.set_random_seed(19);
  model.set_use_positional_encoding(false);
  model.set_scale_embedding(false);
  MUST_EQUAL(model.Init(3, 2, 1, 4, 0), MiniTransformerLM::SUCCESS);
  MUST_EQUAL(model.token_embedding().set_embedding_table(
                 {{1.0, 0.5}, {-0.5, 0.25}, {0.75, -1.0}}),
             TokenEmbedding::SUCCESS);
  MUST_EQUAL(model.set_output_weight(
                 {{0.4, -0.2}, {0.1, 0.8}, {-0.7, 0.3}}),
             MiniTransformerLM::SUCCESS);
  MUST_EQUAL(model.set_output_bias({0.2, -0.1, 0.5}),
             MiniTransformerLM::SUCCESS);

  MiniTransformerLM::Matrix before;
  MUST_EQUAL(model.Forward({0, 2, 1}, before), MiniTransformerLM::SUCCESS);
  const std::vector<int> mapping = {4, 1, 3};
  MUST_EQUAL(model.ExpandVocabulary(5, mapping), MiniTransformerLM::SUCCESS);
  MUST_EQUAL(model.vocab_size(), 5);
  MiniTransformerLM::Matrix after;
  MUST_EQUAL(model.Forward({4, 3, 1}, after), MiniTransformerLM::SUCCESS);
  for (int pos = 0; pos < static_cast<int>(before.size()); pos++) {
    for (int old_id = 0; old_id < 3; old_id++) {
      MUST_TRUE(Near(before[pos][old_id], after[pos][mapping[old_id]], 1e-12),
                "mapped old logit changed");
    }
  }
  MUST_EQUAL(model.TrainNextToken({{4, 0, 3}}, {2}, nullptr, 1, 0.01),
             MiniTransformerLM::SUCCESS);
}

TEST(ModelExpansion, TransformerRejectsInvalidVocabularyMapping) {
  MiniTransformerLM model;
  MUST_EQUAL(model.Init(3, 2, 1, 4, 0), MiniTransformerLM::SUCCESS);
  MUST_EQUAL(model.ExpandVocabulary(5, {0, 0, 2}),
             MiniTransformerLM::INVALID_DATA);
  MUST_EQUAL(model.vocab_size(), 3);
}

TEST(ModelExpansion, TransformerZeroResidualBlockPreservesAndTrains) {
  using namespace model_expansion_test_detail;
  MiniTransformerLM model;
  model.set_random_seed(29);
  model.set_backbone_type(MiniTransformerLM::BACKBONE_DECODER);
  model.set_use_positional_encoding(false);
  model.set_scale_embedding(false);
  MUST_EQUAL(model.Init(3, 4, 1, 8, 0), MiniTransformerLM::SUCCESS);
  MiniTransformerLM::Matrix before;
  MUST_EQUAL(model.Forward({0, 1, 2}, before), MiniTransformerLM::SUCCESS);

  MUST_EQUAL(model.AppendBlock(MiniTransformerLM::APPEND_ZERO_RESIDUAL),
             MiniTransformerLM::SUCCESS);
  MUST_EQUAL(model.block_num(), 1);
  MUST_EQUAL(model.encoder().blocks().size(), 1);
  MUST_EQUAL(model.decoder().blocks().size(), 1);
  const auto &block = model.decoder().blocks()[0];
  MUST_TRUE(block.depth_residual_scale() == 0.0,
            "new residual gate must start at zero");
  MUST_TRUE(block.depth_residual_trainable(),
            "new residual gate must be trainable");
  MiniTransformerLM::Matrix after;
  MUST_EQUAL(model.Forward({0, 1, 2}, after), MiniTransformerLM::SUCCESS);
  MUST_TRUE(SameMatrix(before, after, 0.0), "Transformer output changed");

  MUST_EQUAL(model.TrainNextToken({{0, 1, 2}}, {0}, nullptr, 1, 0.01),
             MiniTransformerLM::SUCCESS);
  const double learned_scale =
      model.decoder().blocks()[0].depth_residual_scale();
  MUST_TRUE(std::isfinite(learned_scale), "residual gate became non-finite");
  MUST_TRUE(learned_scale != 0.0,
            "residual gate did not learn");
}

TEST(ModelExpansion, TransformerParallelResidualGateMatchesBatch) {
  using namespace model_expansion_test_detail;
  MiniTransformerLM batch_model;
  batch_model.set_random_seed(41);
  batch_model.set_use_positional_encoding(false);
  batch_model.set_scale_embedding(false);
  MUST_EQUAL(batch_model.Init(3, 4, 1, 8, 0), MiniTransformerLM::SUCCESS);
  MUST_EQUAL(batch_model.AppendBlock(MiniTransformerLM::APPEND_ZERO_RESIDUAL),
             MiniTransformerLM::SUCCESS);
  MiniTransformerLM parallel_model = batch_model;
  const std::vector<std::vector<int>> samples = {{0, 1}, {1, 2}};
  const std::vector<int> targets = {2, 0};
  MUST_EQUAL(batch_model.TrainNextTokenBatch(samples, targets, 2, nullptr, 2,
                                              0.01),
             MiniTransformerLM::SUCCESS);
  MUST_EQUAL(parallel_model.TrainNextTokenBatchParallel(
                 samples, targets, 2, 2, nullptr, 2, 0.01),
             MiniTransformerLM::SUCCESS);
  const double batch_scale =
      batch_model.encoder().blocks()[0].depth_residual_scale();
  const double parallel_scale =
      parallel_model.encoder().blocks()[0].depth_residual_scale();
  MUST_TRUE(std::isfinite(batch_scale) && std::isfinite(parallel_scale),
            "residual gate became non-finite");
  MUST_TRUE(Near(batch_scale, parallel_scale, 1e-12),
            "parallel residual gate update mismatch");
}

TEST(ModelExpansion, TransformerCopiesBlockAndPersistsResidualGate) {
  using namespace model_expansion_test_detail;
  MiniTransformerLM copied;
  copied.set_random_seed(31);
  MUST_EQUAL(copied.Init(3, 4, 1, 8, 1), MiniTransformerLM::SUCCESS);
  const auto encoder_query =
      copied.encoder().blocks()[0].self_attention().query_weight();
  const auto decoder_query =
      copied.decoder().blocks()[0].self_attention().query_weight();
  MUST_EQUAL(copied.AppendBlock(MiniTransformerLM::APPEND_COPY_LAST),
             MiniTransformerLM::SUCCESS);
  MUST_TRUE(copied.encoder().blocks()[1].self_attention().query_weight() ==
                encoder_query,
            "encoder block copy mismatch");
  MUST_TRUE(copied.decoder().blocks()[1].self_attention().query_weight() ==
                decoder_query,
            "decoder block copy mismatch");

  MiniTransformerLM model;
  model.set_random_seed(37);
  model.set_use_positional_encoding(false);
  model.set_scale_embedding(false);
  MUST_EQUAL(model.Init(3, 4, 1, 8, 0), MiniTransformerLM::SUCCESS);
  MUST_EQUAL(model.AppendBlock(MiniTransformerLM::APPEND_ZERO_RESIDUAL),
             MiniTransformerLM::SUCCESS);
  MUST_EQUAL(model.decoder().mutable_block(0)->set_depth_residual(0.25, false),
             TransformerBlock::SUCCESS);
  MiniTransformerLM::Matrix before;
  MUST_EQUAL(model.Forward({0, 2}, before), MiniTransformerLM::SUCCESS);
  const char *filename = "model_expansion_transformer.param";
  DEFER([=]() { std::remove(filename); });
  MUST_EQUAL(MiniTransformerLMLoader::ExportModelToFile(model, filename),
             MiniTransformerLMLoader::SUCCESS);
  MiniTransformerLM loaded;
  MUST_EQUAL(MiniTransformerLMLoader::ImportModelFromFile(loaded, filename),
             MiniTransformerLMLoader::SUCCESS);
  MUST_TRUE(loaded.encoder().blocks()[0].depth_residual_scale() == 0.0,
            "residual gate was not loaded");
  MUST_TRUE(loaded.encoder().blocks()[0].depth_residual_trainable(),
            "residual trainable flag was not loaded");
  MUST_TRUE(loaded.decoder().blocks()[0].depth_residual_scale() == 0.25,
            "inactive backbone residual gate was not loaded");
  MUST_TRUE(!loaded.decoder().blocks()[0].depth_residual_trainable(),
            "inactive backbone trainable flag was not loaded");
  MiniTransformerLM::Matrix after;
  MUST_EQUAL(loaded.Forward({0, 2}, after), MiniTransformerLM::SUCCESS);
  MUST_TRUE(SameMatrix(before, after, 0.0), "reloaded output changed");
}

TEST(ModelExpansion, TransformerLoaderSupportsLegacyAndIsTransactional) {
  using namespace model_expansion_test_detail;
  MiniTransformerLM model;
  model.set_random_seed(47);
  MUST_EQUAL(model.Init(3, 4, 1, 8, 0), MiniTransformerLM::SUCCESS);
  MUST_EQUAL(model.AppendBlock(MiniTransformerLM::APPEND_ZERO_RESIDUAL),
             MiniTransformerLM::SUCCESS);
  const std::string valid = "model_expansion_valid.param";
  const std::string legacy = "model_expansion_legacy.param";
  const std::string corrupt = "model_expansion_corrupt.param";
  DEFER([=]() {
    std::remove(valid.c_str());
    std::remove(legacy.c_str());
    std::remove(corrupt.c_str());
  });
  MUST_EQUAL(MiniTransformerLMLoader::ExportModelToFile(model, valid),
             MiniTransformerLMLoader::SUCCESS);

  // magic(7) + version/count(8) + two stacks' scale/flag(24)
  MUST_TRUE(CopyWithoutTail(valid, legacy, 39),
            "failed to create legacy checkpoint");
  MiniTransformerLM legacy_loaded;
  MUST_EQUAL(MiniTransformerLMLoader::ImportModelFromFile(legacy_loaded, legacy),
             MiniTransformerLMLoader::SUCCESS);
  MUST_TRUE(legacy_loaded.encoder().blocks()[0].depth_residual_scale() == 1.0,
            "legacy checkpoint did not use the default residual gate");
  MUST_TRUE(!legacy_loaded.encoder().blocks()[0].depth_residual_trainable(),
            "legacy checkpoint gate must be frozen");

  MUST_TRUE(CopyWithoutTail(valid, corrupt, 1),
            "failed to create corrupt checkpoint");
  MiniTransformerLM retryable;
  MUST_EQUAL(MiniTransformerLMLoader::ImportModelFromFile(retryable, corrupt),
             MiniTransformerLMLoader::INPORT_ERROR);
  MUST_EQUAL(retryable.vocab_size(), 0);
  MUST_EQUAL(MiniTransformerLMLoader::ImportModelFromFile(retryable, valid),
             MiniTransformerLMLoader::SUCCESS);
}

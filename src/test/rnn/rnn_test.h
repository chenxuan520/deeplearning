#pragma once

#include "rnn/mini_rnn_lm.h"
#include "rnn/simple_rnn.h"
#include "test.h"
#include "transformer/character_dataset.h"
#include "transformer/character_tokenizer.h"

#include <cmath>
#include <vector>

namespace {

using namespace deeplearning;

bool RnnNearlyEqual(double lhs, double rhs, double eps = 1e-6) {
  return std::fabs(lhs - rhs) <= eps;
}

} // namespace

TEST(SimpleRNN, ForwardWithKnownWeights) {
  SimpleRNN rnn;
  MUST_EQUAL(rnn.Init(2, 2), SimpleRNN::SUCCESS);
  MUST_EQUAL(rnn.set_input_weight({{1.0, 0.0}, {0.0, 1.0}}), SimpleRNN::SUCCESS);
  MUST_EQUAL(rnn.set_hidden_weight({{0.0, 0.0}, {0.0, 0.0}}), SimpleRNN::SUCCESS);
  MUST_EQUAL(rnn.set_bias({0.0, 0.0}), SimpleRNN::SUCCESS);

  SimpleRNN::Matrix hidden;
  MUST_EQUAL(rnn.Forward({{1.0, 0.0}, {0.0, 1.0}}, hidden), SimpleRNN::SUCCESS);
  MUST_EQUAL(hidden.size(), 2);
  MUST_TRUE(RnnNearlyEqual(hidden[0][0], std::tanh(1.0)), "h0[0] mismatch");
  MUST_TRUE(RnnNearlyEqual(hidden[0][1], 0.0), "h0[1] mismatch");
  MUST_TRUE(RnnNearlyEqual(hidden[1][0], 0.0), "h1[0] mismatch");
  MUST_TRUE(RnnNearlyEqual(hidden[1][1], std::tanh(1.0)), "h1[1] mismatch");
}

TEST(MiniRNNLM, TrainAndGenerateToyCorpus) {
  CharacterTokenizer tokenizer;
  MUST_EQUAL(tokenizer.Init("abc"), CharacterTokenizer::SUCCESS);

  std::vector<int> corpus_token_ids;
  MUST_EQUAL(tokenizer.Encode("abcabcabcabcabc", corpus_token_ids),
             CharacterTokenizer::SUCCESS);

  CharacterDataset dataset;
  MUST_EQUAL(dataset.Init(corpus_token_ids, 3), CharacterDataset::SUCCESS);
  std::vector<std::vector<int>> input_samples;
  std::vector<int> target_tokens;
  MUST_EQUAL(dataset.BuildNextTokenSamples(input_samples, target_tokens),
             CharacterDataset::SUCCESS);

  MiniRNNLM model;
  MiniRNNLM::Config config;
  config.vocab_size_ = tokenizer.vocab_size();
  config.hidden_dim_ = 12;
  config.rand_seed_ = 3;
  config.gradient_clip_norm_ = 1.0;
  MUST_EQUAL(model.Init(config), MiniRNNLM::SUCCESS);

  auto rc = model.TrainNextToken(input_samples, target_tokens, nullptr, 220, 0.03);
  MUST_TRUE(rc == MiniRNNLM::SUCCESS, model.err_msg());

  double perplexity = 0.0;
  MUST_EQUAL(model.CalcPerplexity(input_samples, target_tokens, perplexity),
             MiniRNNLM::SUCCESS);
  DEBUG("mini rnn ppl=" << perplexity);
  MUST_TRUE(perplexity < 1.2, "mini rnn should fit toy corpus");

  std::vector<int> prompt_token_ids;
  MUST_EQUAL(tokenizer.Encode("abc", prompt_token_ids), CharacterTokenizer::SUCCESS);
  int next_token = -1;
  MUST_EQUAL(model.PredictNextToken(prompt_token_ids, next_token),
             MiniRNNLM::SUCCESS);
  MUST_EQUAL(next_token, 0);

  std::vector<int> generated_token_ids;
  MUST_EQUAL(model.Generate(prompt_token_ids, 6, generated_token_ids),
             MiniRNNLM::SUCCESS);
  std::string generated_text;
  MUST_EQUAL(tokenizer.Decode(generated_token_ids, generated_text),
             CharacterTokenizer::SUCCESS);
  MUST_TRUE(generated_text == "abcabcabc", "generated text mismatch");
}

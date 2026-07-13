#pragma once

#include "test.h"
#include "transformer/character_tokenizer.h"
#include "transformer/character_dataset.h"
#include "transformer/layer_norm.h"
#include "transformer/mini_transformer_lm.h"
#include "transformer/mini_transformer_lm_loader.h"
#include "transformer/positional_encoding.h"
#include "transformer/self_attention.h"
#include "transformer/token_embedding.h"
#include "transformer/transformer_block.h"
#include "transformer/transformer_decoder.h"
#include "transformer/transformer_encoder.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace std;
using namespace deeplearning;

namespace {

bool NearlyEqual(double lhs, double rhs, double eps = 1e-6) {
  return std::fabs(lhs - rhs) <= eps;
}

vector<vector<double>> IdentityMatrix(int dim) {
  vector<vector<double>> matrix(dim, vector<double>(dim, 0));
  for (int i = 0; i < dim; i++) {
    matrix[i][i] = 1.0;
  }
  return matrix;
}

double RowMean(const vector<double> &row) {
  double sum = 0;
  for (double value : row) {
    sum += value;
  }
  return sum / row.size();
}

double RowVariance(const vector<double> &row) {
  double mean = RowMean(row);
  double variance = 0;
  for (double value : row) {
    double diff = value - mean;
    variance += diff * diff;
  }
  return variance / row.size();
}

} // namespace

TEST(PositionalEncoding, ApplySinusoidalEncoding) {
  vector<vector<double>> sequence = {{0.0, 0.0, 0.0, 0.0},
                                     {0.0, 0.0, 0.0, 0.0}};
  MUST_TRUE(PositionalEncoding::Apply(sequence), "apply encoding failed");

  MUST_TRUE(NearlyEqual(sequence[0][0], 0.0), "position 0 dim 0 mismatch");
  MUST_TRUE(NearlyEqual(sequence[0][1], 1.0), "position 0 dim 1 mismatch");
  MUST_TRUE(NearlyEqual(sequence[0][2], 0.0), "position 0 dim 2 mismatch");
  MUST_TRUE(NearlyEqual(sequence[0][3], 1.0), "position 0 dim 3 mismatch");

  MUST_TRUE(NearlyEqual(sequence[1][0], std::sin(1.0)),
            "position 1 dim 0 mismatch");
  MUST_TRUE(NearlyEqual(sequence[1][1], std::cos(1.0)),
            "position 1 dim 1 mismatch");
}

TEST(LayerNorm, NormalizeEachToken) {
  LayerNorm layer_norm;
  MUST_EQUAL(layer_norm.Init(3), LayerNorm::SUCCESS);

  vector<vector<double>> input = {{1.0, 2.0, 3.0}, {2.0, 4.0, 6.0}};
  vector<vector<double>> output;
  MUST_EQUAL(layer_norm.Forward(input, output), LayerNorm::SUCCESS);

  MUST_TRUE(NearlyEqual(RowMean(output[0]), 0.0, 1e-5), "row 0 mean mismatch");
  MUST_TRUE(NearlyEqual(RowMean(output[1]), 0.0, 1e-5), "row 1 mean mismatch");
  MUST_TRUE(NearlyEqual(RowVariance(output[0]), 1.0, 1e-4),
            "row 0 variance mismatch");
  MUST_TRUE(NearlyEqual(RowVariance(output[1]), 1.0, 1e-4),
            "row 1 variance mismatch");
}

TEST(SelfAttention, ForwardWithCausalMask) {
  SelfAttention attention;
  attention.set_random_seed(0);
  MUST_EQUAL(attention.Init(2, 1), SelfAttention::SUCCESS);
  MUST_EQUAL(attention.set_query_weight(IdentityMatrix(2)),
             SelfAttention::SUCCESS);
  MUST_EQUAL(attention.set_key_weight(IdentityMatrix(2)),
             SelfAttention::SUCCESS);
  MUST_EQUAL(attention.set_value_weight(IdentityMatrix(2)),
             SelfAttention::SUCCESS);
  MUST_EQUAL(attention.set_output_weight(IdentityMatrix(2)),
             SelfAttention::SUCCESS);

  vector<vector<double>> input = {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
  vector<vector<double>> mask = {{1.0, 0.0, 0.0},
                                 {1.0, 1.0, 0.0},
                                 {1.0, 1.0, 1.0}};
  vector<vector<double>> output;
  MUST_EQUAL(attention.Forward(input, output, &mask), SelfAttention::SUCCESS);

  const double scale = std::sqrt(2.0);
  const double expected_w0 = std::exp(0.0);
  const double expected_w1 = std::exp(1.0 / scale);
  const double expected_sum = expected_w0 + expected_w1;

  MUST_TRUE(NearlyEqual(output[0][0], 1.0), "token 0 dim 0 mismatch");
  MUST_TRUE(NearlyEqual(output[0][1], 0.0), "token 0 dim 1 mismatch");
  MUST_TRUE(NearlyEqual(output[1][0], expected_w0 / expected_sum, 1e-6),
            "token 1 dim 0 mismatch");
  MUST_TRUE(NearlyEqual(output[1][1], expected_w1 / expected_sum, 1e-6),
            "token 1 dim 1 mismatch");
}

TEST(TransformerBlock, ForwardSingleTokenSequence) {
  TransformerBlock block;
  block.set_random_seed(0);
  MUST_EQUAL(block.Init(2, 1, 4), TransformerBlock::SUCCESS);
  MUST_EQUAL(block.self_attention().set_query_weight(IdentityMatrix(2)),
             SelfAttention::SUCCESS);
  MUST_EQUAL(block.self_attention().set_key_weight(IdentityMatrix(2)),
             SelfAttention::SUCCESS);
  MUST_EQUAL(block.self_attention().set_value_weight(IdentityMatrix(2)),
             SelfAttention::SUCCESS);
  MUST_EQUAL(block.self_attention().set_output_weight(IdentityMatrix(2)),
             SelfAttention::SUCCESS);

  MUST_EQUAL(block.set_feed_forward_weight_1(vector<vector<double>>(4, vector<double>(2, 0.0))),
             TransformerBlock::SUCCESS);
  MUST_EQUAL(block.set_feed_forward_bias_1(vector<double>(4, 0.0)),
             TransformerBlock::SUCCESS);
  MUST_EQUAL(block.set_feed_forward_weight_2(vector<vector<double>>(2, vector<double>(4, 0.0))),
             TransformerBlock::SUCCESS);
  MUST_EQUAL(block.set_feed_forward_bias_2(vector<double>(2, 0.0)),
             TransformerBlock::SUCCESS);

  vector<vector<double>> input = {{1.0, 2.0}};
  vector<vector<double>> output;
  MUST_EQUAL(block.Forward(input, output), TransformerBlock::SUCCESS);

  MUST_EQUAL(output.size(), 1);
  MUST_EQUAL(output[0].size(), 2);
  MUST_TRUE(std::isfinite(output[0][0]), "output[0][0] not finite");
  MUST_TRUE(std::isfinite(output[0][1]), "output[0][1] not finite");
  MUST_TRUE(NearlyEqual(output[0][0], -1.0, 1e-3), "output dim 0 mismatch");
  MUST_TRUE(NearlyEqual(output[0][1], 1.0, 1e-3), "output dim 1 mismatch");
}

TEST(TokenEmbedding, EncodeTokenIds) {
  TokenEmbedding embedding;
  embedding.set_random_seed(0);
  MUST_EQUAL(embedding.Init(3, 2), TokenEmbedding::SUCCESS);
  MUST_EQUAL(embedding.set_embedding_table({{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}}),
             TokenEmbedding::SUCCESS);

  vector<vector<double>> output;
  MUST_EQUAL(embedding.Encode({2, 0, 1}, output), TokenEmbedding::SUCCESS);
  MUST_EQUAL(output.size(), 3);
  MUST_TRUE(NearlyEqual(output[0][0], 1.0), "token 2 dim 0 mismatch");
  MUST_TRUE(NearlyEqual(output[0][1], 1.0), "token 2 dim 1 mismatch");
  MUST_TRUE(NearlyEqual(output[1][0], 1.0), "token 0 dim 0 mismatch");
  MUST_TRUE(NearlyEqual(output[1][1], 0.0), "token 0 dim 1 mismatch");
}

TEST(TransformerEncoder, ZeroBlockIsPassThrough) {
  TransformerEncoder encoder;
  MUST_EQUAL(encoder.Init(0, 2, 1, 4), TransformerEncoder::SUCCESS);

  vector<vector<double>> input = {{1.0, 2.0}, {3.0, 4.0}};
  vector<vector<double>> output;
  MUST_EQUAL(encoder.Forward(input, output), TransformerEncoder::SUCCESS);
  MUST_EQUAL(output.size(), input.size());
  MUST_TRUE(NearlyEqual(output[1][0], 3.0), "pass through mismatch");
  MUST_TRUE(NearlyEqual(output[1][1], 4.0), "pass through mismatch");
}

TEST(TransformerDecoder, ZeroBlockIsPassThrough) {
  TransformerDecoder decoder;
  MUST_EQUAL(decoder.Init(0, 2, 1, 4), TransformerDecoder::SUCCESS);

  vector<vector<double>> input = {{1.0, 2.0}, {3.0, 4.0}};
  vector<vector<double>> output;
  MUST_EQUAL(decoder.Forward(input, output), TransformerDecoder::SUCCESS);
  MUST_EQUAL(output.size(), input.size());
  MUST_TRUE(NearlyEqual(output[0][0], 1.0), "pass through mismatch");
  MUST_TRUE(NearlyEqual(output[1][1], 4.0), "pass through mismatch");
}

TEST(CharacterTokenizer, EncodeDecodeRoundTrip) {
  CharacterTokenizer tokenizer;
  MUST_EQUAL(tokenizer.Init("abc "), CharacterTokenizer::SUCCESS);

  vector<int> token_ids;
  MUST_EQUAL(tokenizer.Encode("cab", token_ids), CharacterTokenizer::SUCCESS);
  MUST_EQUAL(token_ids.size(), 3);
  MUST_EQUAL(token_ids[0], 2);
  MUST_EQUAL(token_ids[1], 0);
  MUST_EQUAL(token_ids[2], 1);

  string decoded;
  MUST_EQUAL(tokenizer.Decode(token_ids, decoded), CharacterTokenizer::SUCCESS);
  MUST_TRUE(decoded == "cab", "decode mismatch");
}

TEST(CharacterDataset, BuildNextTokenSamples) {
  CharacterDataset dataset;
  MUST_EQUAL(dataset.Init({0, 1, 2, 0, 1}, 3), CharacterDataset::SUCCESS);

  vector<vector<int>> input_samples;
  vector<int> target_tokens;
  MUST_EQUAL(dataset.BuildNextTokenSamples(input_samples, target_tokens),
             CharacterDataset::SUCCESS);
  MUST_EQUAL(dataset.sample_size(), 2);
  MUST_EQUAL(input_samples.size(), 2);
  MUST_EQUAL(target_tokens.size(), 2);
  MUST_EQUAL(input_samples[0][0], 0);
  MUST_EQUAL(input_samples[0][1], 1);
  MUST_EQUAL(input_samples[0][2], 2);
  MUST_EQUAL(target_tokens[0], 0);
  MUST_EQUAL(target_tokens[1], 1);
}

TEST(MiniTransformerLM, PredictConfiguredNextToken) {
  MiniTransformerLM model;
  model.set_random_seed(0);
  model.set_use_positional_encoding(false);
  model.set_scale_embedding(false);
  MUST_EQUAL(model.Init(3, 3, 1, 6, 0), MiniTransformerLM::SUCCESS);
  MUST_EQUAL(model.token_embedding().set_embedding_table(IdentityMatrix(3)),
             TokenEmbedding::SUCCESS);
  MUST_EQUAL(model.set_output_weight({{0.0, 0.0, 1.0},
                                      {1.0, 0.0, 0.0},
                                      {0.0, 1.0, 0.0}}),
             MiniTransformerLM::SUCCESS);
  MUST_EQUAL(model.set_output_bias(vector<double>(3, 0.0)),
             MiniTransformerLM::SUCCESS);

  int next_token_id = -1;
  MUST_EQUAL(model.PredictNextToken({0, 1}, next_token_id),
             MiniTransformerLM::SUCCESS);
  MUST_EQUAL(next_token_id, 2);

  MiniTransformerLM::Matrix logits;
  MUST_EQUAL(model.Forward({0, 1, 2}, logits), MiniTransformerLM::SUCCESS);
  MUST_EQUAL(logits.size(), 3);
  MUST_EQUAL(logits[2].size(), 3);
  MUST_TRUE(logits[2][0] > logits[2][1], "logit order mismatch");
  MUST_TRUE(logits[2][0] > logits[2][2], "logit order mismatch");
}

TEST(MiniTransformerLM, GenerateConfiguredSequence) {
  MiniTransformerLM model;
  model.set_random_seed(0);
  model.set_use_positional_encoding(false);
  model.set_scale_embedding(false);
  MUST_EQUAL(model.Init(3, 3, 1, 6, 0), MiniTransformerLM::SUCCESS);
  MUST_EQUAL(model.token_embedding().set_embedding_table(IdentityMatrix(3)),
             TokenEmbedding::SUCCESS);
  MUST_EQUAL(model.set_output_weight({{0.0, 0.0, 1.0},
                                      {1.0, 0.0, 0.0},
                                      {0.0, 1.0, 0.0}}),
             MiniTransformerLM::SUCCESS);
  MUST_EQUAL(model.set_output_bias(vector<double>(3, 0.0)),
             MiniTransformerLM::SUCCESS);

  vector<int> generated_token_ids;
  MUST_EQUAL(model.Generate({0, 1}, 6, generated_token_ids),
             MiniTransformerLM::SUCCESS);
  MUST_EQUAL(generated_token_ids.size(), 8);
  MUST_EQUAL(generated_token_ids[0], 0);
  MUST_EQUAL(generated_token_ids[1], 1);
  MUST_EQUAL(generated_token_ids[2], 2);
  MUST_EQUAL(generated_token_ids[3], 0);
  MUST_EQUAL(generated_token_ids[4], 1);
  MUST_EQUAL(generated_token_ids[5], 2);
  MUST_EQUAL(generated_token_ids[6], 0);
  MUST_EQUAL(generated_token_ids[7], 1);
}

TEST(MiniTransformerLM, SampleTopKOneMatchesGreedy) {
  MiniTransformerLM model;
  model.set_random_seed(0);
  model.set_use_positional_encoding(false);
  model.set_scale_embedding(false);
  MUST_EQUAL(model.Init(3, 3, 1, 6, 0), MiniTransformerLM::SUCCESS);
  MUST_EQUAL(model.token_embedding().set_embedding_table(IdentityMatrix(3)),
             TokenEmbedding::SUCCESS);
  MUST_EQUAL(model.set_output_weight({{0.0, 0.0, 1.0},
                                      {1.0, 0.0, 0.0},
                                      {0.0, 1.0, 0.0}}),
             MiniTransformerLM::SUCCESS);
  MUST_EQUAL(model.set_output_bias(vector<double>(3, 0.0)),
             MiniTransformerLM::SUCCESS);

  MiniTransformerLM::SamplingOption option;
  option.top_k_ = 1;

  vector<int> greedy_generated_token_ids;
  MUST_EQUAL(model.Generate({0, 1}, 6, greedy_generated_token_ids),
             MiniTransformerLM::SUCCESS);
  vector<int> sampled_generated_token_ids;
  MUST_EQUAL(model.GenerateSample({0, 1}, 6, sampled_generated_token_ids, option),
             MiniTransformerLM::SUCCESS);

  MUST_EQUAL(sampled_generated_token_ids.size(), greedy_generated_token_ids.size());
  for (int i = 0; i < static_cast<int>(greedy_generated_token_ids.size());
       i++) {
    MUST_EQUAL(sampled_generated_token_ids[i], greedy_generated_token_ids[i]);
  }
}

TEST(MiniTransformerLM, RejectInvalidSamplingOption) {
  MiniTransformerLM model;
  model.set_random_seed(0);
  model.set_use_positional_encoding(false);
  model.set_scale_embedding(false);
  MUST_EQUAL(model.Init(3, 3, 1, 6, 0), MiniTransformerLM::SUCCESS);
  MUST_EQUAL(model.token_embedding().set_embedding_table(IdentityMatrix(3)),
             TokenEmbedding::SUCCESS);

  MiniTransformerLM::SamplingOption option;
  option.temperature_ = 0;

  int token_id = -1;
  MUST_EQUAL(model.SampleNextToken({0, 1}, token_id, option),
             MiniTransformerLM::INVALID_DATA);
}

TEST(MiniTransformerLM, CalcNextTokenLossUsesEveryPosition) {
  MiniTransformerLM model;
  model.set_random_seed(0);
  model.set_use_positional_encoding(false);
  model.set_scale_embedding(false);
  MUST_EQUAL(model.Init(3, 3, 1, 6, 0), MiniTransformerLM::SUCCESS);
  MUST_EQUAL(model.token_embedding().set_embedding_table(IdentityMatrix(3)),
             TokenEmbedding::SUCCESS);
  MUST_EQUAL(model.set_output_weight(IdentityMatrix(3)),
             MiniTransformerLM::SUCCESS);
  MUST_EQUAL(model.set_output_bias({0.0, 1.0, 2.0}),
             MiniTransformerLM::SUCCESS);

  double average_loss = 0.0;
  MUST_EQUAL(model.CalcNextTokenLoss({{0, 1}}, {2}, average_loss),
             MiniTransformerLM::SUCCESS);

  const double pos0_loss =
      std::log(std::exp(1.0) + std::exp(1.0) + std::exp(2.0)) - 1.0;
  const double pos1_loss =
      std::log(std::exp(0.0) + std::exp(2.0) + std::exp(2.0)) - 2.0;
  MUST_TRUE(NearlyEqual(average_loss, (pos0_loss + pos1_loss) / 2.0, 1e-9),
            "loss should average every next-token position");
}

TEST(MiniTransformerLM, TrainAndReloadCharacterModel) {
  const string corpus = "abcabcabcabcabcabc";
  CharacterTokenizer tokenizer;
  MUST_EQUAL(tokenizer.Init(CharacterTokenizer::BuildVocabularyFromText(corpus)),
             CharacterTokenizer::SUCCESS);

  vector<int> corpus_token_ids;
  MUST_EQUAL(tokenizer.Encode(corpus, corpus_token_ids),
             CharacterTokenizer::SUCCESS);

  CharacterDataset dataset;
  MUST_EQUAL(dataset.Init(corpus_token_ids, 2), CharacterDataset::SUCCESS);

  vector<vector<int>> input_samples;
  vector<int> target_tokens;
  MUST_EQUAL(dataset.BuildNextTokenSamples(input_samples, target_tokens),
             CharacterDataset::SUCCESS);

  MiniTransformerLM model;
  model.set_random_seed(0);
  model.set_use_positional_encoding(false);
  model.set_scale_embedding(true);
  MUST_EQUAL(model.Init(tokenizer.vocab_size(), 6, 1, 12, 0),
             MiniTransformerLM::SUCCESS);
  MUST_EQUAL(model.TrainNextToken(input_samples, target_tokens, nullptr, 250, 0.1),
             MiniTransformerLM::SUCCESS);

  vector<int> prompt_token_ids;
  MUST_EQUAL(tokenizer.Encode("ab", prompt_token_ids),
             CharacterTokenizer::SUCCESS);

  vector<int> generated_token_ids;
  MUST_EQUAL(model.Generate(prompt_token_ids, 6, generated_token_ids),
             MiniTransformerLM::SUCCESS);
  string generated_text;
  MUST_EQUAL(tokenizer.Decode(generated_token_ids, generated_text),
             CharacterTokenizer::SUCCESS);
  MUST_TRUE(generated_text == "abcabcab", "trained generation mismatch");

  const string filename = "mini_transformer_lm_test.param";
  DEFER([=]() { remove(filename.c_str()); });
  MUST_EQUAL(MiniTransformerLMLoader::ExportModelToFile(model, filename),
             MiniTransformerLMLoader::SUCCESS);

  MiniTransformerLM loaded_model;
  MUST_EQUAL(MiniTransformerLMLoader::ImportModelFromFile(loaded_model, filename),
             MiniTransformerLMLoader::SUCCESS);

  vector<int> loaded_generated_token_ids;
  MUST_EQUAL(loaded_model.Generate(prompt_token_ids, 6, loaded_generated_token_ids),
             MiniTransformerLM::SUCCESS);
  string loaded_generated_text;
  MUST_EQUAL(tokenizer.Decode(loaded_generated_token_ids, loaded_generated_text),
             CharacterTokenizer::SUCCESS);
  MUST_TRUE(loaded_generated_text == generated_text,
            "reload generation mismatch");
}

TEST(MiniTransformerLM, TrainBatchCharacterModel) {
  const string corpus = "abcabcabcabcabcabc";
  CharacterTokenizer tokenizer;
  MUST_EQUAL(tokenizer.Init(CharacterTokenizer::BuildVocabularyFromText(corpus)),
             CharacterTokenizer::SUCCESS);

  vector<int> corpus_token_ids;
  MUST_EQUAL(tokenizer.Encode(corpus, corpus_token_ids),
             CharacterTokenizer::SUCCESS);

  CharacterDataset dataset;
  MUST_EQUAL(dataset.Init(corpus_token_ids, 2), CharacterDataset::SUCCESS);

  vector<vector<int>> input_samples;
  vector<int> target_tokens;
  MUST_EQUAL(dataset.BuildNextTokenSamples(input_samples, target_tokens),
             CharacterDataset::SUCCESS);

  MiniTransformerLM model;
  model.set_random_seed(0);
  model.set_use_positional_encoding(false);
  model.set_scale_embedding(true);
  MUST_EQUAL(model.Init(tokenizer.vocab_size(), 6, 1, 12, 0),
             MiniTransformerLM::SUCCESS);
  MUST_EQUAL(model.TrainNextTokenBatch(input_samples, target_tokens, 4, nullptr,
                                       250, 0.1),
             MiniTransformerLM::SUCCESS);

  double average_loss = 0.0;
  MUST_EQUAL(model.CalcNextTokenLoss(input_samples, target_tokens, average_loss),
             MiniTransformerLM::SUCCESS);
  MUST_TRUE(average_loss < 0.1, "batch trained loss too high");

  vector<int> prompt_token_ids;
  MUST_EQUAL(tokenizer.Encode("ab", prompt_token_ids),
             CharacterTokenizer::SUCCESS);

  vector<int> generated_token_ids;
  MUST_EQUAL(model.Generate(prompt_token_ids, 6, generated_token_ids),
             MiniTransformerLM::SUCCESS);
  string generated_text;
  MUST_EQUAL(tokenizer.Decode(generated_token_ids, generated_text),
             CharacterTokenizer::SUCCESS);
  MUST_TRUE(generated_text == "abcabcab", "batch generation mismatch");
}

TEST(MiniTransformerLM, TrainBatchSingleBlockDecoderCharacterModel) {
  const string corpus = "abcabcabcabcabcabc";
  CharacterTokenizer tokenizer;
  MUST_EQUAL(tokenizer.Init(CharacterTokenizer::BuildVocabularyFromText(corpus)),
             CharacterTokenizer::SUCCESS);

  vector<int> corpus_token_ids;
  MUST_EQUAL(tokenizer.Encode(corpus, corpus_token_ids),
             CharacterTokenizer::SUCCESS);

  CharacterDataset dataset;
  MUST_EQUAL(dataset.Init(corpus_token_ids, 2), CharacterDataset::SUCCESS);

  vector<vector<int>> input_samples;
  vector<int> target_tokens;
  MUST_EQUAL(dataset.BuildNextTokenSamples(input_samples, target_tokens),
             CharacterDataset::SUCCESS);

  MiniTransformerLM model;
  model.set_random_seed(0);
  model.set_backbone_type(MiniTransformerLM::BACKBONE_DECODER);
  model.set_use_positional_encoding(true);
  model.set_scale_embedding(true);
  model.set_max_context_size(2);
  MUST_EQUAL(model.Init(tokenizer.vocab_size(), 6, 1, 12, 1),
             MiniTransformerLM::SUCCESS);
  auto query_weight_before =
      model.decoder().blocks()[0].self_attention().query_weight();
  MUST_EQUAL(model.TrainNextTokenBatch(input_samples, target_tokens, 4, nullptr,
                                       1200, 0.01),
             MiniTransformerLM::SUCCESS);

  auto query_weight_after =
      model.decoder().blocks()[0].self_attention().query_weight();
  bool has_query_weight_change = false;
  for (int i = 0; i < static_cast<int>(query_weight_before.size()); i++) {
    for (int j = 0; j < static_cast<int>(query_weight_before[i].size()); j++) {
      if (!NearlyEqual(query_weight_before[i][j], query_weight_after[i][j],
                       1e-9)) {
        has_query_weight_change = true;
      }
    }
  }
  MUST_TRUE(has_query_weight_change,
            "batch single block attention weight did not update");

  vector<int> prompt_token_ids;
  MUST_EQUAL(tokenizer.Encode("ab", prompt_token_ids),
             CharacterTokenizer::SUCCESS);

  vector<int> generated_token_ids;
  MUST_EQUAL(model.Generate(prompt_token_ids, 6, generated_token_ids),
             MiniTransformerLM::SUCCESS);
  string generated_text;
  MUST_EQUAL(tokenizer.Decode(generated_token_ids, generated_text),
             CharacterTokenizer::SUCCESS);
  MUST_TRUE(generated_text == "abcabcab",
            "batch single block generation mismatch");
}

TEST(MiniTransformerLM, CalcPerplexityAfterTraining) {
  const string corpus = "abcabcabcabcabcabc";
  CharacterTokenizer tokenizer;
  MUST_EQUAL(tokenizer.Init(CharacterTokenizer::BuildVocabularyFromText(corpus)),
             CharacterTokenizer::SUCCESS);

  vector<int> corpus_token_ids;
  MUST_EQUAL(tokenizer.Encode(corpus, corpus_token_ids),
             CharacterTokenizer::SUCCESS);

  CharacterDataset dataset;
  MUST_EQUAL(dataset.Init(corpus_token_ids, 2), CharacterDataset::SUCCESS);

  vector<vector<int>> input_samples;
  vector<int> target_tokens;
  MUST_EQUAL(dataset.BuildNextTokenSamples(input_samples, target_tokens),
             CharacterDataset::SUCCESS);

  MiniTransformerLM model;
  model.set_random_seed(0);
  model.set_use_positional_encoding(false);
  model.set_scale_embedding(true);
  MUST_EQUAL(model.Init(tokenizer.vocab_size(), 6, 1, 12, 0),
             MiniTransformerLM::SUCCESS);
  MUST_EQUAL(model.TrainNextToken(input_samples, target_tokens, nullptr, 250, 0.1),
             MiniTransformerLM::SUCCESS);

  double average_loss = 0;
  MUST_EQUAL(model.CalcNextTokenLoss(input_samples, target_tokens, average_loss),
             MiniTransformerLM::SUCCESS);
  double perplexity = 0;
  MUST_EQUAL(model.CalcPerplexity(input_samples, target_tokens, perplexity),
             MiniTransformerLM::SUCCESS);
  MUST_TRUE(average_loss < 0.1, "trained loss too high");
  MUST_TRUE(perplexity < 1.2, "trained perplexity too high");
  MUST_TRUE(NearlyEqual(perplexity, std::exp(average_loss), 1e-9),
            "perplexity formula mismatch");
}

TEST(MiniTransformerLM, TrainSingleBlockCharacterModel) {
  const string corpus = "abcabcabcabcabcabc";
  CharacterTokenizer tokenizer;
  MUST_EQUAL(tokenizer.Init(CharacterTokenizer::BuildVocabularyFromText(corpus)),
             CharacterTokenizer::SUCCESS);

  vector<int> corpus_token_ids;
  MUST_EQUAL(tokenizer.Encode(corpus, corpus_token_ids),
             CharacterTokenizer::SUCCESS);

  CharacterDataset dataset;
  MUST_EQUAL(dataset.Init(corpus_token_ids, 2), CharacterDataset::SUCCESS);

  vector<vector<int>> input_samples;
  vector<int> target_tokens;
  MUST_EQUAL(dataset.BuildNextTokenSamples(input_samples, target_tokens),
             CharacterDataset::SUCCESS);

  MiniTransformerLM model;
  model.set_random_seed(0);
  model.set_backbone_type(MiniTransformerLM::BACKBONE_DECODER);
  model.set_use_positional_encoding(true);
  model.set_scale_embedding(true);
  model.set_max_context_size(2);
  MUST_EQUAL(model.Init(tokenizer.vocab_size(), 6, 1, 12, 1),
             MiniTransformerLM::SUCCESS);
  auto query_weight_before = model.decoder().blocks()[0].self_attention().query_weight();
  MUST_EQUAL(model.TrainNextToken(input_samples, target_tokens, nullptr, 1000, 0.01),
             MiniTransformerLM::SUCCESS);
  auto query_weight_after = model.decoder().blocks()[0].self_attention().query_weight();
  bool has_query_weight_change = false;
  for (int i = 0; i < static_cast<int>(query_weight_before.size()); i++) {
    for (int j = 0; j < static_cast<int>(query_weight_before[i].size()); j++) {
      if (!NearlyEqual(query_weight_before[i][j], query_weight_after[i][j], 1e-9)) {
        has_query_weight_change = true;
      }
    }
  }
  MUST_TRUE(has_query_weight_change, "single block attention weight did not update");

  vector<int> prompt_token_ids;
  MUST_EQUAL(tokenizer.Encode("ab", prompt_token_ids),
             CharacterTokenizer::SUCCESS);

  vector<int> generated_token_ids;
  MUST_EQUAL(model.Generate(prompt_token_ids, 6, generated_token_ids),
             MiniTransformerLM::SUCCESS);
  string generated_text;
  MUST_EQUAL(tokenizer.Decode(generated_token_ids, generated_text),
             CharacterTokenizer::SUCCESS);
  MUST_TRUE(generated_text == "abcabcab", "single block generation mismatch");

  const string filename = "mini_transformer_lm_block_test.param";
  DEFER([=]() { remove(filename.c_str()); });
  MUST_EQUAL(MiniTransformerLMLoader::ExportModelToFile(model, filename),
             MiniTransformerLMLoader::SUCCESS);

  MiniTransformerLM loaded_model;
  MUST_EQUAL(MiniTransformerLMLoader::ImportModelFromFile(loaded_model, filename),
             MiniTransformerLMLoader::SUCCESS);
  vector<int> loaded_generated_token_ids;
  MUST_EQUAL(loaded_model.Generate(prompt_token_ids, 6, loaded_generated_token_ids),
             MiniTransformerLM::SUCCESS);
  string loaded_generated_text;
  MUST_EQUAL(tokenizer.Decode(loaded_generated_token_ids, loaded_generated_text),
             CharacterTokenizer::SUCCESS);
  MUST_TRUE(loaded_generated_text == generated_text,
            "single block reload generation mismatch");
}

TEST(MiniTransformerLM, TrainTwoBlockDecoderCharacterModel) {
  const string corpus = "abcabcabcabcabcabc";
  CharacterTokenizer tokenizer;
  MUST_EQUAL(tokenizer.Init(CharacterTokenizer::BuildVocabularyFromText(corpus)),
             CharacterTokenizer::SUCCESS);

  vector<int> corpus_token_ids;
  MUST_EQUAL(tokenizer.Encode(corpus, corpus_token_ids),
             CharacterTokenizer::SUCCESS);

  CharacterDataset dataset;
  MUST_EQUAL(dataset.Init(corpus_token_ids, 2), CharacterDataset::SUCCESS);

  vector<vector<int>> input_samples;
  vector<int> target_tokens;
  MUST_EQUAL(dataset.BuildNextTokenSamples(input_samples, target_tokens),
             CharacterDataset::SUCCESS);

  MiniTransformerLM model;
  model.set_random_seed(0);
  model.set_backbone_type(MiniTransformerLM::BACKBONE_DECODER);
  model.set_use_positional_encoding(true);
  model.set_scale_embedding(true);
  model.set_max_context_size(2);
  MUST_EQUAL(model.Init(tokenizer.vocab_size(), 6, 1, 12, 2),
             MiniTransformerLM::SUCCESS);
  auto train_callback = [](int, double average_loss, bool &early_stop) {
    if (average_loss < 0.02) {
      early_stop = true;
    }
  };
  MUST_EQUAL(model.TrainNextToken(input_samples, target_tokens, train_callback, 1500,
                                  0.01),
             MiniTransformerLM::SUCCESS);

  vector<int> prompt_token_ids;
  MUST_EQUAL(tokenizer.Encode("ab", prompt_token_ids),
             CharacterTokenizer::SUCCESS);

  const string filename = "mini_transformer_lm_two_block_test.param";
  DEFER([=]() { remove(filename.c_str()); });
  MUST_EQUAL(MiniTransformerLMLoader::ExportModelToFile(model, filename),
             MiniTransformerLMLoader::SUCCESS);

  MiniTransformerLM loaded_model;
  MUST_EQUAL(MiniTransformerLMLoader::ImportModelFromFile(loaded_model, filename),
             MiniTransformerLMLoader::SUCCESS);

  vector<int> generated_token_ids;
  MUST_EQUAL(loaded_model.Generate(prompt_token_ids, 6, generated_token_ids),
             MiniTransformerLM::SUCCESS);
  string generated_text;
  MUST_EQUAL(tokenizer.Decode(generated_token_ids, generated_text),
             CharacterTokenizer::SUCCESS);
  MUST_TRUE(generated_text == "abcabcab", "two block decoder generation mismatch");
}

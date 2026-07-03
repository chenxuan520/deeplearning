#include "word2vec.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace deeplearning {

namespace {

constexpr double kSigmoidClamp = 20.0;

} // namespace

Word2Vec::RC Word2Vec::Init(int vocab_size, const Config &config) {
  if (is_init_) {
    err_msg_ = "[Word2Vec::Init] Word2Vec already init";
    return ALREADY_INIT;
  }
  if (vocab_size <= 0 || config.embed_dim <= 0 || config.window_size <= 0 ||
      config.negative_num <= 0 || config.epochs <= 0 || config.learning_rate <= 0) {
    err_msg_ = "[Word2Vec::Init] Invalid config";
    return INVALID_DATA;
  }

  vocab_size_ = vocab_size;
  config_ = config;
  random_counter_ = config.rand_seed;
  word_freq_.assign(vocab_size_, 0);
  noise_distribution_.clear();
  noise_prefix_.clear();
  noise_total_ = 0.0;

  InitEmbeddings();
  is_init_ = true;
  return SUCCESS;
}

void Word2Vec::InitEmbeddings() {
  input_embeddings_.assign(vocab_size_,
                           EmbeddingVector(config_.embed_dim, 0.0));
  output_embeddings_.assign(vocab_size_,
                            EmbeddingVector(config_.embed_dim, 0.0));

  const double limit =
      std::sqrt(6.0 / (static_cast<double>(vocab_size_) + config_.embed_dim));
  std::mt19937 gen(config_.rand_seed);
  std::uniform_real_distribution<double> dist(-limit, limit);
  for (auto &embedding : input_embeddings_) {
    for (double &value : embedding) {
      value = dist(gen);
    }
  }
  for (auto &embedding : output_embeddings_) {
    for (double &value : embedding) {
      value = dist(gen);
    }
  }
}

Word2Vec::RC Word2Vec::Train(const std::vector<int> &token_ids,
                             TrainStats *stats) {
  if (!is_init_) {
    err_msg_ = "[Word2Vec::Train] Word2Vec not init";
    return NOT_INIT;
  }
  if (token_ids.empty()) {
    err_msg_ = "[Word2Vec::Train] Empty token sequence";
    return INVALID_DATA;
  }

  return Train(std::vector<std::vector<int>>{token_ids}, stats);
}

Word2Vec::RC Word2Vec::Train(const std::vector<std::vector<int>> &sentences,
                             TrainStats *stats) {
  if (!is_init_) {
    err_msg_ = "[Word2Vec::Train] Word2Vec not init";
    return NOT_INIT;
  }
  if (sentences.empty()) {
    err_msg_ = "[Word2Vec::Train] Empty sentences";
    return INVALID_DATA;
  }

  BuildWordFrequencies(sentences);
  BuildNoiseDistribution();

  double loss_sum = 0.0;
  long long pair_count = 0;

  for (int epoch = 0; epoch < config_.epochs; epoch++) {
    const double progress =
        config_.epochs <= 1
            ? 1.0
            : static_cast<double>(epoch) / (config_.epochs - 1);
    const double learning_rate =
        config_.learning_rate * (1.0 - progress) +
        config_.min_learning_rate * progress;

    for (const auto &sentence : sentences) {
      if (sentence.empty()) {
        continue;
      }

      if (config_.mode == Mode::SKIP_GRAM) {
        std::vector<int> centers;
        std::vector<int> contexts;
        BuildSkipGramPairs(sentence, centers, contexts);
        for (size_t i = 0; i < centers.size(); i++) {
          loss_sum += TrainSkipGramPair(centers[i], contexts[i], learning_rate);
          pair_count++;
        }
      } else {
        std::vector<std::vector<int>> context_batch;
        std::vector<int> centers;
        BuildCBOWSamples(sentence, context_batch, centers);
        for (size_t i = 0; i < centers.size(); i++) {
          loss_sum += TrainCBOWSample(context_batch[i], centers[i], learning_rate);
          pair_count++;
        }
      }
    }
  }

  if (stats != nullptr) {
    stats->average_loss =
        pair_count > 0 ? loss_sum / static_cast<double>(pair_count) : 0.0;
    stats->pair_count = pair_count;
  }
  return SUCCESS;
}

void Word2Vec::BuildWordFrequencies(
    const std::vector<std::vector<int>> &sentences) {
  std::fill(word_freq_.begin(), word_freq_.end(), 0);
  for (const auto &sentence : sentences) {
    for (int token_id : sentence) {
      if (token_id < 0 || token_id >= vocab_size_) {
        continue;
      }
      word_freq_[token_id]++;
    }
  }
}

void Word2Vec::BuildNoiseDistribution() {
  noise_distribution_.assign(vocab_size_, 0.0);
  noise_prefix_.assign(vocab_size_, 0.0);
  noise_total_ = 0.0;

  for (int token_id = 0; token_id < vocab_size_; token_id++) {
    const double freq = static_cast<double>(word_freq_[token_id]);
    const double weight = freq > 0.0 ? std::pow(freq, 0.75) : 0.0;
    noise_distribution_[token_id] = weight;
    noise_total_ += weight;
  }

  if (noise_total_ <= 0.0) {
    for (int token_id = 0; token_id < vocab_size_; token_id++) {
      noise_distribution_[token_id] = 1.0;
    }
    noise_total_ = static_cast<double>(vocab_size_);
  }

  double prefix = 0.0;
  for (int token_id = 0; token_id < vocab_size_; token_id++) {
    prefix += noise_distribution_[token_id];
    noise_prefix_[token_id] = prefix;
  }
}

void Word2Vec::BuildSkipGramPairs(const std::vector<int> &token_ids,
                                  std::vector<int> &centers,
                                  std::vector<int> &contexts) const {
  centers.clear();
  contexts.clear();

  const int seq_len = static_cast<int>(token_ids.size());
  for (int center_pos = 0; center_pos < seq_len; center_pos++) {
    const int left = std::max(0, center_pos - config_.window_size);
    const int right =
        std::min(seq_len - 1, center_pos + config_.window_size);
    for (int context_pos = left; context_pos <= right; context_pos++) {
      if (context_pos == center_pos) {
        continue;
      }
      centers.push_back(token_ids[center_pos]);
      contexts.push_back(token_ids[context_pos]);
    }
  }
}

void Word2Vec::BuildCBOWSamples(
    const std::vector<int> &token_ids, std::vector<std::vector<int>> &contexts,
    std::vector<int> &centers) const {
  contexts.clear();
  centers.clear();

  const int seq_len = static_cast<int>(token_ids.size());
  for (int center_pos = 0; center_pos < seq_len; center_pos++) {
    const int left = std::max(0, center_pos - config_.window_size);
    const int right =
        std::min(seq_len - 1, center_pos + config_.window_size);
    std::vector<int> context_ids;
    for (int context_pos = left; context_pos <= right; context_pos++) {
      if (context_pos == center_pos) {
        continue;
      }
      context_ids.push_back(token_ids[context_pos]);
    }
    if (context_ids.empty()) {
      continue;
    }
    contexts.push_back(context_ids);
    centers.push_back(token_ids[center_pos]);
  }
}

double Word2Vec::Dot(const EmbeddingVector &a, const EmbeddingVector &b) const {
  double sum = 0.0;
  for (int dim = 0; dim < config_.embed_dim; dim++) {
    sum += a[dim] * b[dim];
  }
  return sum;
}

double Word2Vec::Sigmoid(double x) const {
  if (x > kSigmoidClamp) {
    return 1.0;
  }
  if (x < -kSigmoidClamp) {
    return 0.0;
  }
  return 1.0 / (1.0 + std::exp(-x));
}

int Word2Vec::SampleNegative(int center_id, int context_id) {
  std::mt19937 gen(config_.rand_seed + random_counter_++);
  std::uniform_real_distribution<double> dist(0.0, noise_total_);
  const double target = dist(gen);

  for (int attempt = 0; attempt < 16; attempt++) {
    const auto upper = std::upper_bound(noise_prefix_.begin(),
                                        noise_prefix_.end(), target);
    int sampled_id = static_cast<int>(upper - noise_prefix_.begin());
    if (sampled_id >= vocab_size_) {
      sampled_id = vocab_size_ - 1;
    }
    if (sampled_id != center_id && sampled_id != context_id) {
      return sampled_id;
    }
  }

  for (int token_id = 0; token_id < vocab_size_; token_id++) {
    if (token_id != center_id && token_id != context_id) {
      return token_id;
    }
  }
  return center_id;
}

void Word2Vec::UpdatePositivePair(EmbeddingVector &input_vec,
                                  EmbeddingVector &output_vec,
                                  double learning_rate) {
  const double score = Dot(input_vec, output_vec);
  const double sigmoid_score = Sigmoid(score);
  const double grad_scale = 1.0 - sigmoid_score;

  for (int dim = 0; dim < config_.embed_dim; dim++) {
    const double grad = grad_scale * learning_rate;
    input_vec[dim] += grad * output_vec[dim];
    output_vec[dim] += grad * input_vec[dim];
  }
}

void Word2Vec::UpdateNegativePair(EmbeddingVector &input_vec,
                                  EmbeddingVector &output_vec,
                                  double learning_rate) {
  const double score = Dot(input_vec, output_vec);
  const double sigmoid_score = Sigmoid(score);
  const double grad_scale = -sigmoid_score;

  for (int dim = 0; dim < config_.embed_dim; dim++) {
    const double grad = grad_scale * learning_rate;
    input_vec[dim] += grad * output_vec[dim];
    output_vec[dim] += grad * input_vec[dim];
  }
}

double Word2Vec::TrainSkipGramPair(int center_id, int context_id,
                                   double learning_rate) {
  EmbeddingVector &input_vec = input_embeddings_[center_id];
  EmbeddingVector &output_vec = output_embeddings_[context_id];

  const double score = Dot(input_vec, output_vec);
  double loss = -std::log(std::max(Sigmoid(score), 1e-12));
  UpdatePositivePair(input_vec, output_vec, learning_rate);

  for (int neg_idx = 0; neg_idx < config_.negative_num; neg_idx++) {
    const int negative_id = SampleNegative(center_id, context_id);
    EmbeddingVector &negative_vec = output_embeddings_[negative_id];
    const double negative_score = Dot(input_vec, negative_vec);
    loss += -std::log(std::max(1.0 - Sigmoid(negative_score), 1e-12));
    UpdateNegativePair(input_vec, negative_vec, learning_rate);
  }
  return loss;
}

Word2Vec::EmbeddingVector
Word2Vec::AverageInputVectors(const std::vector<int> &context_ids) const {
  EmbeddingVector averaged(config_.embed_dim, 0.0);
  if (context_ids.empty()) {
    return averaged;
  }

  for (int context_id : context_ids) {
    const EmbeddingVector &context_vec = input_embeddings_[context_id];
    for (int dim = 0; dim < config_.embed_dim; dim++) {
      averaged[dim] += context_vec[dim];
    }
  }

  const double inv = 1.0 / static_cast<double>(context_ids.size());
  for (double &value : averaged) {
    value *= inv;
  }
  return averaged;
}

double Word2Vec::TrainCBOWSample(const std::vector<int> &context_ids,
                                 int center_id, double learning_rate) {
  EmbeddingVector hidden = AverageInputVectors(context_ids);
  EmbeddingVector &output_vec = output_embeddings_[center_id];

  const double score = Dot(hidden, output_vec);
  double loss = -std::log(std::max(Sigmoid(score), 1e-12));

  const double sigmoid_score = Sigmoid(score);
  const double grad_scale = 1.0 - sigmoid_score;
  const double inv_context = 1.0 / static_cast<double>(context_ids.size());

  for (int dim = 0; dim < config_.embed_dim; dim++) {
    const double grad = grad_scale * learning_rate;
    output_vec[dim] += grad * hidden[dim];
    for (int context_id : context_ids) {
      input_embeddings_[context_id][dim] += grad * output_vec[dim] * inv_context;
    }
  }

  for (int neg_idx = 0; neg_idx < config_.negative_num; neg_idx++) {
    const int negative_id = SampleNegative(center_id, center_id);
    EmbeddingVector &negative_vec = output_embeddings_[negative_id];
    const double negative_score = Dot(hidden, negative_vec);
    loss += -std::log(std::max(1.0 - Sigmoid(negative_score), 1e-12));

    const double negative_grad_scale = -Sigmoid(negative_score);
    for (int dim = 0; dim < config_.embed_dim; dim++) {
      const double grad = negative_grad_scale * learning_rate;
      negative_vec[dim] += grad * hidden[dim];
      for (int context_id : context_ids) {
        input_embeddings_[context_id][dim] +=
            grad * negative_vec[dim] * inv_context;
      }
    }
  }

  return loss;
}

double Word2Vec::CosineSimilarity(int word_a, int word_b) const {
  if (!is_init_ || word_a < 0 || word_b < 0 || word_a >= vocab_size_ ||
      word_b >= vocab_size_) {
    return 0.0;
  }

  const EmbeddingVector &a = input_embeddings_[word_a];
  const EmbeddingVector &b = input_embeddings_[word_b];
  double dot = 0.0;
  double norm_a = 0.0;
  double norm_b = 0.0;
  for (int dim = 0; dim < config_.embed_dim; dim++) {
    dot += a[dim] * b[dim];
    norm_a += a[dim] * a[dim];
    norm_b += b[dim] * b[dim];
  }
  if (norm_a <= 0.0 || norm_b <= 0.0) {
    return 0.0;
  }
  return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

std::vector<std::pair<int, double>>
Word2Vec::MostSimilar(int word_id, int top_k) const {
  std::vector<std::pair<int, double>> result;
  if (!is_init_ || word_id < 0 || word_id >= vocab_size_ || top_k <= 0) {
    return result;
  }

  for (int candidate_id = 0; candidate_id < vocab_size_; candidate_id++) {
    if (candidate_id == word_id) {
      continue;
    }
    result.emplace_back(candidate_id, CosineSimilarity(word_id, candidate_id));
  }

  const int keep = std::min(top_k, static_cast<int>(result.size()));
  std::partial_sort(result.begin(), result.begin() + keep, result.end(),
                    [](const auto &lhs, const auto &rhs) {
                      return lhs.second > rhs.second;
                    });
  result.resize(keep);
  return result;
}

const Word2Vec::EmbeddingVector &Word2Vec::InputVector(int word_id) const {
  static const EmbeddingVector kEmpty;
  if (!is_init_ || word_id < 0 || word_id >= vocab_size_) {
    return kEmpty;
  }
  return input_embeddings_[word_id];
}

const Word2Vec::Matrix &Word2Vec::input_embeddings() const {
  return input_embeddings_;
}

const Word2Vec::Matrix &Word2Vec::output_embeddings() const {
  return output_embeddings_;
}

std::string Word2Vec::err_msg() const { return err_msg_; }

int Word2Vec::vocab_size() const { return vocab_size_; }

int Word2Vec::embed_dim() const { return config_.embed_dim; }

Word2Vec::Mode Word2Vec::mode() const { return config_.mode; }

} // namespace deeplearning

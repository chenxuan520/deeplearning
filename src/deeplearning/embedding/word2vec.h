#pragma once

#include <string>
#include <utility>
#include <vector>

namespace deeplearning {

class Word2Vec {
public:
  using Matrix = std::vector<std::vector<double>>;

  enum class Mode { SKIP_GRAM, CBOW };

  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
  };

  struct Config {
    int embed_dim = 32;
    int window_size = 2;
    int negative_num = 5;
    int epochs = 10;
    double learning_rate = 0.05;
    double min_learning_rate = 0.0001;
    int rand_seed = 0;
    Mode mode = Mode::SKIP_GRAM;
  };

  struct TrainStats {
    double average_loss = 0.0;
    long long pair_count = 0;
  };

public:
  RC Init(int vocab_size, const Config &config);
  RC Train(const std::vector<std::vector<int>> &sentences,
           TrainStats *stats = nullptr);
  RC Train(const std::vector<int> &token_ids, TrainStats *stats = nullptr);

  double CosineSimilarity(int word_a, int word_b) const;
  std::vector<std::pair<int, double>> MostSimilar(int word_id, int top_k) const;

  const std::vector<double> &InputVector(int word_id) const;
  const Matrix &input_embeddings() const;
  const Matrix &output_embeddings() const;

  std::string err_msg() const;
  int vocab_size() const;
  int embed_dim() const;
  Mode mode() const;

private:
  using EmbeddingVector = std::vector<double>;

  void InitEmbeddings();
  void BuildWordFrequencies(const std::vector<std::vector<int>> &sentences);
  void BuildNoiseDistribution();
  void BuildSkipGramPairs(const std::vector<int> &token_ids,
                          std::vector<int> &centers,
                          std::vector<int> &contexts) const;
  void BuildCBOWSamples(const std::vector<int> &token_ids,
                        std::vector<std::vector<int>> &contexts,
                        std::vector<int> &centers) const;

  double Dot(const EmbeddingVector &a, const EmbeddingVector &b) const;
  double Sigmoid(double x) const;
  int SampleNegative(int center_id, int context_id);
  double TrainSkipGramPair(int center_id, int context_id, double learning_rate);
  double TrainCBOWSample(const std::vector<int> &context_ids, int center_id,
                         double learning_rate);
  void UpdatePositivePair(EmbeddingVector &input_vec, EmbeddingVector &output_vec,
                          double learning_rate);
  void UpdateNegativePair(EmbeddingVector &input_vec, EmbeddingVector &output_vec,
                          double learning_rate);
  EmbeddingVector AverageInputVectors(
      const std::vector<int> &context_ids) const;

private:
  Config config_;
  int vocab_size_ = 0;
  Matrix input_embeddings_;
  Matrix output_embeddings_;
  std::vector<long long> word_freq_;
  std::vector<double> noise_distribution_;
  std::vector<double> noise_prefix_;
  double noise_total_ = 0.0;
  int random_counter_ = 0;
  std::string err_msg_;
  bool is_init_ = false;
};

} // namespace deeplearning

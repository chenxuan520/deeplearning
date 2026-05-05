#include "token_embedding.h"

#include <cmath>
#include <random>

namespace deeplearning {

TokenEmbedding::RC TokenEmbedding::Init(int vocab_size, int model_dim) {
  if (is_init_) {
    err_msg_ = "[TokenEmbedding::Init] TokenEmbedding has init";
    return ALREADY_INIT;
  }
  if (vocab_size <= 0 || model_dim <= 0) {
    err_msg_ = "[TokenEmbedding::Init] Invalid vocab size or model dim";
    return INVALID_DATA;
  }

  vocab_size_ = vocab_size;
  model_dim_ = model_dim;
  embedding_table_.assign(vocab_size_, std::vector<double>(model_dim_, 0));

  const double limit = std::sqrt(6.0 / (vocab_size_ + model_dim_));
  std::mt19937 gen(rand_seed_);
  std::uniform_real_distribution<double> dist(-limit, limit);
  for (auto &token_embedding : embedding_table_) {
    for (double &value : token_embedding) {
      value = dist(gen);
    }
  }

  is_init_ = true;
  return SUCCESS;
}

TokenEmbedding::RC TokenEmbedding::Encode(const std::vector<int> &token_ids,
                                          Matrix &output) {
  if (!is_init_) {
    err_msg_ = "[TokenEmbedding::Encode] TokenEmbedding not init";
    return NOT_INIT;
  }
  if (token_ids.empty()) {
    err_msg_ = "[TokenEmbedding::Encode] Invalid token input";
    return INVALID_DATA;
  }

  output.clear();
  output.reserve(token_ids.size());
  for (int token_id : token_ids) {
    if (token_id < 0 || token_id >= vocab_size_) {
      err_msg_ = "[TokenEmbedding::Encode] Invalid token input";
      return INVALID_DATA;
    }
    output.push_back(embedding_table_[token_id]);
  }
  return SUCCESS;
}

void TokenEmbedding::set_random_seed(int seed) { rand_seed_ = seed; }

TokenEmbedding::RC
TokenEmbedding::set_embedding_table(const Matrix &embedding_table) {
  if (!is_init_) {
    err_msg_ = "[TokenEmbedding::set_embedding_table] TokenEmbedding not init";
    return NOT_INIT;
  }
  if (embedding_table.size() != vocab_size_) {
    err_msg_ =
        "[TokenEmbedding::set_embedding_table] Invalid embedding table size";
    return INVALID_DATA;
  }
  for (const auto &token_embedding : embedding_table) {
    if (token_embedding.size() != model_dim_) {
      err_msg_ =
          "[TokenEmbedding::set_embedding_table] Invalid embedding table size";
      return INVALID_DATA;
    }
  }

  embedding_table_ = embedding_table;
  return SUCCESS;
}

std::string TokenEmbedding::err_msg() { return err_msg_; }

int TokenEmbedding::vocab_size() { return vocab_size_; }

int TokenEmbedding::model_dim() { return model_dim_; }

TokenEmbedding::Matrix &TokenEmbedding::mutable_embedding_table() {
  return embedding_table_;
}

const TokenEmbedding::Matrix &TokenEmbedding::embedding_table() const {
  return embedding_table_;
}

} // namespace deeplearning

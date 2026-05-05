#pragma once

#include <string>
#include <vector>

namespace deeplearning {

class TokenEmbedding {
public:
  using Matrix = std::vector<std::vector<double>>;

  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
  };

public:
  RC Init(int vocab_size, int model_dim);
  RC Encode(const std::vector<int> &token_ids, Matrix &output);

  void set_random_seed(int seed);
  RC set_embedding_table(const Matrix &embedding_table);

  std::string err_msg();
  int vocab_size();
  int model_dim();
  Matrix &mutable_embedding_table();
  const Matrix &embedding_table() const;

private:
  int vocab_size_ = 0;
  int model_dim_ = 0;
  int rand_seed_ = 0;
  Matrix embedding_table_;
  std::string err_msg_;
  bool is_init_ = false;
};

} // namespace deeplearning

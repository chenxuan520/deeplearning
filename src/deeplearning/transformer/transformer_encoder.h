#pragma once

#include "transformer_block.h"

#include <string>
#include <vector>

namespace deeplearning {

class TransformerEncoder {
public:
  using Matrix = std::vector<std::vector<double>>;

  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
  };

public:
  RC Init(int block_num, int model_dim, int head_num, int feed_forward_dim);
  RC Forward(const Matrix &input, Matrix &output,
             const Matrix *mask = nullptr);
  RC Backward(const Matrix &grad_output, Matrix &grad_input,
              double learning_rate);

  void set_random_seed(int seed);
  TransformerBlock *mutable_block(int index);
  const std::vector<TransformerBlock> &blocks() const;
  std::string err_msg();

private:
  int block_num_ = 0;
  int model_dim_ = 0;
  int head_num_ = 0;
  int feed_forward_dim_ = 0;
  int rand_seed_ = 0;
  std::vector<TransformerBlock> blocks_;
  std::string err_msg_;
  bool is_init_ = false;
};

} // namespace deeplearning

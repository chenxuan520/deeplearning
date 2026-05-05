#include "transformer_decoder.h"

namespace deeplearning {
namespace {

TransformerDecoder::Matrix CreateCausalMask(int sequence_length) {
  TransformerDecoder::Matrix mask(sequence_length,
                                  std::vector<double>(sequence_length, 0));
  for (int i = 0; i < sequence_length; i++) {
    for (int j = 0; j <= i; j++) {
      mask[i][j] = 1.0;
    }
  }
  return mask;
}

} // namespace

TransformerDecoder::RC TransformerDecoder::Init(int block_num, int model_dim,
                                                int head_num,
                                                int feed_forward_dim) {
  if (is_init_) {
    err_msg_ = "[TransformerDecoder::Init] TransformerDecoder has init";
    return ALREADY_INIT;
  }
  if (block_num < 0 || model_dim <= 0 || head_num <= 0 ||
      feed_forward_dim <= 0) {
    err_msg_ = "[TransformerDecoder::Init] Invalid config";
    return INVALID_DATA;
  }

  block_num_ = block_num;
  model_dim_ = model_dim;
  head_num_ = head_num;
  feed_forward_dim_ = feed_forward_dim;
  blocks_.reserve(block_num_);
  for (int i = 0; i < block_num_; i++) {
    TransformerBlock block;
    block.set_random_seed(rand_seed_ + i);
    if (block.Init(model_dim_, head_num_, feed_forward_dim_) !=
        TransformerBlock::SUCCESS) {
      err_msg_ = block.err_msg();
      return INVALID_DATA;
    }
    blocks_.push_back(block);
  }

  is_init_ = true;
  return SUCCESS;
}

TransformerDecoder::RC TransformerDecoder::Forward(const Matrix &input,
                                                   Matrix &output) {
  if (!is_init_) {
    err_msg_ = "[TransformerDecoder::Forward] TransformerDecoder not init";
    return NOT_INIT;
  }
  if (input.empty()) {
    err_msg_ = "[TransformerDecoder::Forward] Invalid data input";
    return INVALID_DATA;
  }
  for (const auto &token : input) {
    if (token.size() != model_dim_) {
      err_msg_ = "[TransformerDecoder::Forward] Invalid data input";
      return INVALID_DATA;
    }
  }

  output = input;
  Matrix mask = CreateCausalMask(input.size());
  for (auto &block : blocks_) {
    Matrix block_output;
    if (block.Forward(output, block_output, &mask) != TransformerBlock::SUCCESS) {
      err_msg_ = block.err_msg();
      return INVALID_DATA;
    }
    output = block_output;
  }
  return SUCCESS;
}

TransformerDecoder::RC
TransformerDecoder::Backward(const Matrix &grad_output, Matrix &grad_input,
                             double learning_rate) {
  if (!is_init_) {
    err_msg_ = "[TransformerDecoder::Backward] TransformerDecoder not init";
    return NOT_INIT;
  }
  if (grad_output.empty()) {
    err_msg_ = "[TransformerDecoder::Backward] Invalid data input";
    return INVALID_DATA;
  }

  grad_input = grad_output;
  for (int i = block_num_ - 1; i >= 0; i--) {
    Matrix block_grad_input;
    if (blocks_[i].Backward(grad_input, block_grad_input, learning_rate) !=
        TransformerBlock::SUCCESS) {
      err_msg_ = blocks_[i].err_msg();
      return INVALID_DATA;
    }
    grad_input = block_grad_input;
  }
  return SUCCESS;
}

void TransformerDecoder::set_random_seed(int seed) { rand_seed_ = seed; }

TransformerBlock *TransformerDecoder::mutable_block(int index) {
  if (!is_init_ || index < 0 || index >= blocks_.size()) {
    return nullptr;
  }
  return &blocks_[index];
}

const std::vector<TransformerBlock> &TransformerDecoder::blocks() const {
  return blocks_;
}

std::string TransformerDecoder::err_msg() { return err_msg_; }

} // namespace deeplearning

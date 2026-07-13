#include "character_dataset.h"

namespace deeplearning {

CharacterDataset::RC CharacterDataset::Init(const std::vector<int> &token_ids,
                                            int context_size) {
  if (is_init_) {
    err_msg_ = "[CharacterDataset::Init] CharacterDataset has init";
    return ALREADY_INIT;
  }
  if (token_ids.size() < 2 || context_size <= 0) {
    err_msg_ = "[CharacterDataset::Init] Invalid token input";
    return INVALID_DATA;
  }

  token_ids_ = token_ids;
  context_size_ = context_size;
  is_init_ = true;
  return SUCCESS;
}

CharacterDataset::RC CharacterDataset::BuildNextTokenSamples(
    std::vector<std::vector<int>> &input_samples,
    std::vector<int> &target_tokens) {
  if (!is_init_) {
    err_msg_ = "[CharacterDataset::BuildNextTokenSamples] CharacterDataset not init";
    return NOT_INIT;
  }
  if (token_ids_.size() <= static_cast<size_t>(context_size_)) {
    err_msg_ = "[CharacterDataset::BuildNextTokenSamples] Invalid token input";
    return INVALID_DATA;
  }

  input_samples.clear();
  target_tokens.clear();
  for (int i = 0; i + context_size_ < static_cast<int>(token_ids_.size()); i++) {
    input_samples.push_back(std::vector<int>(token_ids_.begin() + i,
                                             token_ids_.begin() + i + context_size_));
    target_tokens.push_back(token_ids_[i + context_size_]);
  }
  return SUCCESS;
}

std::string CharacterDataset::err_msg() { return err_msg_; }

int CharacterDataset::context_size() { return context_size_; }

int CharacterDataset::sample_size() {
  if (!is_init_ || token_ids_.size() <= static_cast<size_t>(context_size_)) {
    return 0;
  }
  return token_ids_.size() - context_size_;
}

} // namespace deeplearning

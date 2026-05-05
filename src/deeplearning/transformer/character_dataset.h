#pragma once

#include <string>
#include <vector>

namespace deeplearning {

class CharacterDataset {
public:
  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
  };

public:
  RC Init(const std::vector<int> &token_ids, int context_size);
  RC BuildNextTokenSamples(std::vector<std::vector<int>> &input_samples,
                           std::vector<int> &target_tokens);

  std::string err_msg();
  int context_size();
  int sample_size();

private:
  std::vector<int> token_ids_;
  int context_size_ = 0;
  std::string err_msg_;
  bool is_init_ = false;
};

} // namespace deeplearning

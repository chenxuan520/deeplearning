#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace deeplearning {

class CharacterTokenizer {
public:
  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
  };

public:
  static std::string BuildVocabularyFromText(const std::string &text);

public:
  RC Init(const std::string &vocabulary);
  RC Encode(const std::string &text, std::vector<int> &token_ids);
  RC Decode(const std::vector<int> &token_ids, std::string &text);

  std::string err_msg();
  int vocab_size() const;
  const std::string &vocabulary() const;

private:
  std::string vocabulary_;
  std::unordered_map<char, int> char_to_id_;
  std::string err_msg_;
  bool is_init_ = false;
};

} // namespace deeplearning

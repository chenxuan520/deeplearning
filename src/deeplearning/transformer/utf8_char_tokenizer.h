#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace deeplearning {

class Utf8CharTokenizer {
public:
  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
  };

public:
  static bool SplitUtf8(const std::string &text,
                        std::vector<std::string> &tokens);
  static std::vector<std::string>
  BuildVocabularyFromText(const std::string &text);

public:
  RC Init(const std::vector<std::string> &vocabulary);
  RC Encode(const std::string &text, std::vector<int> &token_ids);
  RC Decode(const std::vector<int> &token_ids, std::string &text);

  int Lookup(const std::string &token) const;
  std::string err_msg() const;
  int vocab_size() const;
  const std::vector<std::string> &vocabulary() const;

private:
  std::vector<std::string> vocabulary_;
  std::unordered_map<std::string, int> token_to_id_;
  std::string err_msg_;
  bool is_init_ = false;
};

} // namespace deeplearning

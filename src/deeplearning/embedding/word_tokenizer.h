#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace deeplearning {

class WordTokenizer {
public:
  enum RC {
    SUCCESS,
    INVALID_DATA,
    NOT_INIT,
    ALREADY_INIT,
  };

public:
  struct Config {
    bool add_unknown_token = false;
    int max_vocab_size = 0;
  };

public:
  RC InitFromText(const std::string &text, const Config &config);
  RC InitFromText(const std::string &text, bool add_unknown_token = false);
  RC InitFromVocabulary(const std::vector<std::string> &vocabulary);
  RC TokenizeSentences(const std::string &text,
                       std::vector<std::vector<int>> &sentences);
  RC TokenizeFlat(const std::string &text, std::vector<int> &token_ids);
  RC TokenizeFlatWithUnknown(const std::string &text,
                             std::vector<int> &token_ids,
                             int &unknown_count);
  RC Decode(const std::vector<int> &token_ids, std::string &text);

  int Lookup(const std::string &word) const;
  const std::string &Word(int token_id) const;

  std::string err_msg() const;
  int vocab_size() const;
  const std::vector<std::string> &vocabulary() const;

private:
  static bool IsWordChar(unsigned char ch);
  static std::string NormalizeToken(const std::string &word);

  RC AddWord(const std::string &word, int &token_id);

private:
  std::vector<std::string> vocabulary_;
  std::unordered_map<std::string, int> word_to_id_;
  std::string err_msg_;
  bool is_init_ = false;
};

} // namespace deeplearning

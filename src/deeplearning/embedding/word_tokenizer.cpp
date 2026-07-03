#include "word_tokenizer.h"

#include <cctype>

namespace deeplearning {

bool WordTokenizer::IsWordChar(unsigned char ch) {
  return std::isalnum(ch) != 0 || ch == '_';
}

std::string WordTokenizer::NormalizeToken(const std::string &word) {
  std::string normalized;
  normalized.reserve(word.size());
  for (unsigned char ch : word) {
    normalized.push_back(static_cast<char>(std::tolower(ch)));
  }
  return normalized;
}

WordTokenizer::RC WordTokenizer::AddWord(const std::string &word, int &token_id) {
  if (word.empty()) {
    err_msg_ = "[WordTokenizer::AddWord] Empty word";
    return INVALID_DATA;
  }

  auto found = word_to_id_.find(word);
  if (found != word_to_id_.end()) {
    token_id = found->second;
    return SUCCESS;
  }

  token_id = static_cast<int>(vocabulary_.size());
  vocabulary_.push_back(word);
  word_to_id_[word] = token_id;
  return SUCCESS;
}

WordTokenizer::RC WordTokenizer::InitFromText(const std::string &text) {
  if (is_init_) {
    err_msg_ = "[WordTokenizer::InitFromText] WordTokenizer already init";
    return ALREADY_INIT;
  }
  if (text.empty()) {
    err_msg_ = "[WordTokenizer::InitFromText] Empty text";
    return INVALID_DATA;
  }

  vocabulary_.clear();
  word_to_id_.clear();

  std::string current;
  for (unsigned char ch : text) {
    if (IsWordChar(ch)) {
      current.push_back(static_cast<char>(ch));
      continue;
    }
    if (!current.empty()) {
      int token_id = 0;
      auto rc = AddWord(NormalizeToken(current), token_id);
      if (rc != SUCCESS) {
        return rc;
      }
      current.clear();
    }
  }
  if (!current.empty()) {
    int token_id = 0;
    auto rc = AddWord(NormalizeToken(current), token_id);
    if (rc != SUCCESS) {
      return rc;
    }
  }

  if (vocabulary_.empty()) {
    err_msg_ = "[WordTokenizer::InitFromText] No words found";
    return INVALID_DATA;
  }

  is_init_ = true;
  return SUCCESS;
}

WordTokenizer::RC
WordTokenizer::TokenizeSentences(const std::string &text,
                                 std::vector<std::vector<int>> &sentences) {
  if (!is_init_) {
    err_msg_ = "[WordTokenizer::TokenizeSentences] WordTokenizer not init";
    return NOT_INIT;
  }

  sentences.clear();
  std::vector<int> current_sentence;
  std::string current;

  auto flush_word = [&]() -> WordTokenizer::RC {
    if (current.empty()) {
      return SUCCESS;
    }
    const std::string word = NormalizeToken(current);
    auto found = word_to_id_.find(word);
    if (found == word_to_id_.end()) {
      err_msg_ = "[WordTokenizer::TokenizeSentences] Unknown word: " + word;
      return INVALID_DATA;
    }
    current_sentence.push_back(found->second);
    current.clear();
    return SUCCESS;
  };

  for (unsigned char ch : text) {
    if (IsWordChar(ch)) {
      current.push_back(static_cast<char>(ch));
      continue;
    }

    auto rc = flush_word();
    if (rc != SUCCESS) {
      return rc;
    }

    if (ch == '\n' || ch == '\r') {
      if (!current_sentence.empty()) {
        sentences.push_back(current_sentence);
        current_sentence.clear();
      }
    }
  }

  auto rc = flush_word();
  if (rc != SUCCESS) {
    return rc;
  }
  if (!current_sentence.empty()) {
    sentences.push_back(current_sentence);
  }
  if (sentences.empty()) {
    err_msg_ = "[WordTokenizer::TokenizeSentences] No sentences found";
    return INVALID_DATA;
  }
  return SUCCESS;
}

WordTokenizer::RC WordTokenizer::TokenizeFlat(const std::string &text,
                                              std::vector<int> &token_ids) {
  std::vector<std::vector<int>> sentences;
  auto rc = TokenizeSentences(text, sentences);
  if (rc != SUCCESS) {
    return rc;
  }

  token_ids.clear();
  for (const auto &sentence : sentences) {
    token_ids.insert(token_ids.end(), sentence.begin(), sentence.end());
  }
  return SUCCESS;
}

int WordTokenizer::Lookup(const std::string &word) const {
  if (!is_init_) {
    return -1;
  }
  const std::string normalized = NormalizeToken(word);
  auto found = word_to_id_.find(normalized);
  if (found == word_to_id_.end()) {
    return -1;
  }
  return found->second;
}

const std::string &WordTokenizer::Word(int token_id) const {
  static const std::string kEmpty;
  if (!is_init_ || token_id < 0 ||
      token_id >= static_cast<int>(vocabulary_.size())) {
    return kEmpty;
  }
  return vocabulary_[token_id];
}

std::string WordTokenizer::err_msg() const { return err_msg_; }

int WordTokenizer::vocab_size() const {
  return static_cast<int>(vocabulary_.size());
}

const std::vector<std::string> &WordTokenizer::vocabulary() const {
  return vocabulary_;
}

} // namespace deeplearning

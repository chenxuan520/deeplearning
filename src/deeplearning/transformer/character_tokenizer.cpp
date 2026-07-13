#include "character_tokenizer.h"

namespace deeplearning {

std::string CharacterTokenizer::BuildVocabularyFromText(const std::string &text) {
  std::string vocabulary;
  std::unordered_map<char, int> char_to_id;
  for (char ch : text) {
    if (char_to_id.count(ch) != 0) {
      continue;
    }
    char_to_id[ch] = vocabulary.size();
    vocabulary.push_back(ch);
  }
  return vocabulary;
}

CharacterTokenizer::RC CharacterTokenizer::Init(const std::string &vocabulary) {
  if (is_init_) {
    err_msg_ = "[CharacterTokenizer::Init] CharacterTokenizer has init";
    return ALREADY_INIT;
  }
  if (vocabulary.empty()) {
    err_msg_ = "[CharacterTokenizer::Init] Invalid vocabulary";
    return INVALID_DATA;
  }

  vocabulary_ = vocabulary;
  for (int i = 0; i < static_cast<int>(vocabulary_.size()); i++) {
    if (char_to_id_.count(vocabulary_[i]) != 0) {
      err_msg_ = "[CharacterTokenizer::Init] Duplicate char in vocabulary";
      return INVALID_DATA;
    }
    char_to_id_[vocabulary_[i]] = i;
  }
  is_init_ = true;
  return SUCCESS;
}

CharacterTokenizer::RC CharacterTokenizer::Encode(const std::string &text,
                                                  std::vector<int> &token_ids) {
  if (!is_init_) {
    err_msg_ = "[CharacterTokenizer::Encode] CharacterTokenizer not init";
    return NOT_INIT;
  }
  if (text.empty()) {
    err_msg_ = "[CharacterTokenizer::Encode] Invalid text input";
    return INVALID_DATA;
  }

  token_ids.clear();
  token_ids.reserve(text.size());
  for (char ch : text) {
    auto it = char_to_id_.find(ch);
    if (it == char_to_id_.end()) {
      err_msg_ = "[CharacterTokenizer::Encode] Invalid text input";
      return INVALID_DATA;
    }
    token_ids.push_back(it->second);
  }
  return SUCCESS;
}

CharacterTokenizer::RC
CharacterTokenizer::Decode(const std::vector<int> &token_ids, std::string &text) {
  if (!is_init_) {
    err_msg_ = "[CharacterTokenizer::Decode] CharacterTokenizer not init";
    return NOT_INIT;
  }
  if (token_ids.empty()) {
    err_msg_ = "[CharacterTokenizer::Decode] Invalid token input";
    return INVALID_DATA;
  }

  text.clear();
  text.reserve(token_ids.size());
  for (int token_id : token_ids) {
    if (token_id < 0 || token_id >= static_cast<int>(vocabulary_.size())) {
      err_msg_ = "[CharacterTokenizer::Decode] Invalid token input";
      return INVALID_DATA;
    }
    text.push_back(vocabulary_[token_id]);
  }
  return SUCCESS;
}

std::string CharacterTokenizer::err_msg() { return err_msg_; }

int CharacterTokenizer::vocab_size() const { return vocabulary_.size(); }

const std::string &CharacterTokenizer::vocabulary() const { return vocabulary_; }

} // namespace deeplearning

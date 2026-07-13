#include "utf8_char_tokenizer.h"

#include <cstdint>

namespace deeplearning {
namespace {

bool IsContinuationByte(unsigned char ch) { return (ch & 0xC0) == 0x80; }

} // namespace

bool Utf8CharTokenizer::SplitUtf8(const std::string &text,
                                  std::vector<std::string> &tokens) {
  tokens.clear();
  if (text.empty()) {
    return false;
  }

  for (int pos = 0; pos < static_cast<int>(text.size());) {
    const unsigned char first = static_cast<unsigned char>(text[pos]);
    int length = 0;
    uint32_t codepoint = 0;
    if ((first & 0x80) == 0) {
      length = 1;
      codepoint = first;
    } else if ((first & 0xE0) == 0xC0) {
      length = 2;
      codepoint = first & 0x1F;
    } else if ((first & 0xF0) == 0xE0) {
      length = 3;
      codepoint = first & 0x0F;
    } else if ((first & 0xF8) == 0xF0) {
      length = 4;
      codepoint = first & 0x07;
    } else {
      return false;
    }

    if (pos + length > static_cast<int>(text.size())) {
      return false;
    }
    for (int i = 1; i < length; i++) {
      const unsigned char ch = static_cast<unsigned char>(text[pos + i]);
      if (!IsContinuationByte(ch)) {
        return false;
      }
      codepoint = (codepoint << 6) | (ch & 0x3F);
    }

    if ((length == 2 && codepoint < 0x80) ||
        (length == 3 && codepoint < 0x800) ||
        (length == 4 && codepoint < 0x10000) ||
        (codepoint >= 0xD800 && codepoint <= 0xDFFF) ||
        codepoint > 0x10FFFF) {
      return false;
    }

    tokens.push_back(text.substr(pos, length));
    pos += length;
  }
  return !tokens.empty();
}

std::vector<std::string>
Utf8CharTokenizer::BuildVocabularyFromText(const std::string &text) {
  std::vector<std::string> tokens;
  if (!SplitUtf8(text, tokens)) {
    return {};
  }

  std::vector<std::string> vocabulary;
  std::unordered_map<std::string, int> token_to_id;
  for (const auto &token : tokens) {
    if (token_to_id.count(token) != 0) {
      continue;
    }
    token_to_id[token] = static_cast<int>(vocabulary.size());
    vocabulary.push_back(token);
  }
  return vocabulary;
}

Utf8CharTokenizer::RC
Utf8CharTokenizer::Init(const std::vector<std::string> &vocabulary) {
  if (is_init_) {
    err_msg_ = "[Utf8CharTokenizer::Init] Utf8CharTokenizer has init";
    return ALREADY_INIT;
  }
  if (vocabulary.empty()) {
    err_msg_ = "[Utf8CharTokenizer::Init] Invalid vocabulary";
    return INVALID_DATA;
  }

  vocabulary_.clear();
  token_to_id_.clear();
  for (const auto &token : vocabulary) {
    std::vector<std::string> split_tokens;
    if (!SplitUtf8(token, split_tokens) || split_tokens.size() != 1) {
      err_msg_ = "[Utf8CharTokenizer::Init] Invalid vocabulary";
      return INVALID_DATA;
    }
    if (token_to_id_.count(token) != 0) {
      err_msg_ = "[Utf8CharTokenizer::Init] Duplicate token in vocabulary";
      return INVALID_DATA;
    }
    token_to_id_[token] = static_cast<int>(vocabulary_.size());
    vocabulary_.push_back(token);
  }

  is_init_ = true;
  return SUCCESS;
}

Utf8CharTokenizer::RC
Utf8CharTokenizer::Encode(const std::string &text, std::vector<int> &token_ids) {
  if (!is_init_) {
    err_msg_ = "[Utf8CharTokenizer::Encode] Utf8CharTokenizer not init";
    return NOT_INIT;
  }

  std::vector<std::string> tokens;
  if (!SplitUtf8(text, tokens)) {
    err_msg_ = "[Utf8CharTokenizer::Encode] Invalid text input";
    return INVALID_DATA;
  }

  token_ids.clear();
  token_ids.reserve(tokens.size());
  for (const auto &token : tokens) {
    auto it = token_to_id_.find(token);
    if (it == token_to_id_.end()) {
      err_msg_ = "[Utf8CharTokenizer::Encode] Invalid text input";
      return INVALID_DATA;
    }
    token_ids.push_back(it->second);
  }
  return SUCCESS;
}

Utf8CharTokenizer::RC
Utf8CharTokenizer::Decode(const std::vector<int> &token_ids, std::string &text) {
  if (!is_init_) {
    err_msg_ = "[Utf8CharTokenizer::Decode] Utf8CharTokenizer not init";
    return NOT_INIT;
  }
  if (token_ids.empty()) {
    err_msg_ = "[Utf8CharTokenizer::Decode] Invalid token input";
    return INVALID_DATA;
  }

  text.clear();
  for (int token_id : token_ids) {
    if (token_id < 0 || token_id >= static_cast<int>(vocabulary_.size())) {
      err_msg_ = "[Utf8CharTokenizer::Decode] Invalid token input";
      return INVALID_DATA;
    }
    text += vocabulary_[token_id];
  }
  return SUCCESS;
}

int Utf8CharTokenizer::Lookup(const std::string &token) const {
  if (!is_init_) {
    return -1;
  }
  auto found = token_to_id_.find(token);
  if (found == token_to_id_.end()) {
    return -1;
  }
  return found->second;
}

std::string Utf8CharTokenizer::err_msg() const { return err_msg_; }

int Utf8CharTokenizer::vocab_size() const {
  return static_cast<int>(vocabulary_.size());
}

const std::vector<std::string> &Utf8CharTokenizer::vocabulary() const {
  return vocabulary_;
}

} // namespace deeplearning

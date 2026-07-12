#include "word_tokenizer.h"

#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <utility>

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

WordTokenizer::RC WordTokenizer::InitFromText(const std::string &text,
                                             const Config &config) {
  if (is_init_) {
    err_msg_ = "[WordTokenizer::InitFromText] WordTokenizer already init";
    return ALREADY_INIT;
  }
  if (text.empty()) {
    err_msg_ = "[WordTokenizer::InitFromText] Empty text";
    return INVALID_DATA;
  }
  if (config.max_vocab_size < 0) {
    err_msg_ = "[WordTokenizer::InitFromText] Invalid max vocab size";
    return INVALID_DATA;
  }
  if (config.unknown_policy == UNKNOWN_MAP_TO_UNK &&
      !config.add_unknown_token) {
    err_msg_ =
        "[WordTokenizer::InitFromText] UNKNOWN_MAP_TO_UNK requires <unk>";
    return INVALID_DATA;
  }

  vocabulary_.clear();
  word_to_id_.clear();
  unknown_policy_ = config.unknown_policy;
  if (config.add_unknown_token) {
    int token_id = 0;
    auto rc = AddWord("<unk>", token_id);
    if (rc != SUCCESS) {
      return rc;
    }
  }

  std::string current;
  bool has_word = false;
  if (config.max_vocab_size > 0) {
    struct WordStats {
      int count = 0;
      int first_pos = 0;
    };
    std::unordered_map<std::string, WordStats> word_stats;
    int next_pos = 0;

    auto count_word = [&]() {
      if (current.empty()) {
        return;
      }
      const std::string word = NormalizeToken(current);
      auto found = word_stats.find(word);
      if (found == word_stats.end()) {
        word_stats[word] = WordStats{1, next_pos};
        next_pos++;
      } else {
        found->second.count++;
      }
      has_word = true;
      current.clear();
    };

    for (unsigned char ch : text) {
      if (IsWordChar(ch)) {
        current.push_back(static_cast<char>(ch));
        continue;
      }
      count_word();
    }
    count_word();

    if (!has_word) {
      err_msg_ = "[WordTokenizer::InitFromText] No words found";
      return INVALID_DATA;
    }

    std::vector<std::pair<std::string, WordStats>> ranked_words;
    ranked_words.reserve(word_stats.size());
    for (const auto &entry : word_stats) {
      ranked_words.push_back(entry);
    }
    std::sort(ranked_words.begin(), ranked_words.end(),
              [](const auto &lhs, const auto &rhs) {
                if (lhs.second.count != rhs.second.count) {
                  return lhs.second.count > rhs.second.count;
                }
                return lhs.second.first_pos < rhs.second.first_pos;
              });

    const int keep_num = std::min(config.max_vocab_size,
                                  static_cast<int>(ranked_words.size()));
    for (int i = 0; i < keep_num; i++) {
      int token_id = 0;
      auto rc = AddWord(ranked_words[i].first, token_id);
      if (rc != SUCCESS) {
        return rc;
      }
    }
  } else {
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
        has_word = true;
        current.clear();
      }
    }
    if (!current.empty()) {
      int token_id = 0;
      auto rc = AddWord(NormalizeToken(current), token_id);
      if (rc != SUCCESS) {
        return rc;
      }
      has_word = true;
    }
  }

  if (!has_word) {
    err_msg_ = "[WordTokenizer::InitFromText] No words found";
    return INVALID_DATA;
  }

  is_init_ = true;
  return SUCCESS;
}

WordTokenizer::RC WordTokenizer::InitFromText(const std::string &text,
                                             bool add_unknown_token) {
  Config config;
  config.add_unknown_token = add_unknown_token;
  config.unknown_policy =
      add_unknown_token ? UNKNOWN_MAP_TO_UNK : UNKNOWN_DROP;
  return InitFromText(text, config);
}

WordTokenizer::RC
WordTokenizer::InitFromVocabulary(const std::vector<std::string> &vocabulary,
                                  const Config &config) {
  if (is_init_) {
    err_msg_ = "[WordTokenizer::InitFromVocabulary] WordTokenizer already init";
    return ALREADY_INIT;
  }
  if (vocabulary.empty()) {
    err_msg_ = "[WordTokenizer::InitFromVocabulary] Empty vocabulary";
    return INVALID_DATA;
  }
  if (config.unknown_policy == UNKNOWN_MAP_TO_UNK) {
    bool has_unknown = false;
    for (const auto &word : vocabulary) {
      if (NormalizeToken(word) == "<unk>") {
        has_unknown = true;
        break;
      }
    }
    if (!has_unknown) {
      err_msg_ =
          "[WordTokenizer::InitFromVocabulary] UNKNOWN_MAP_TO_UNK requires <unk>";
      return INVALID_DATA;
    }
  }

  vocabulary_.clear();
  word_to_id_.clear();
  unknown_policy_ = config.unknown_policy;
  for (const auto &word : vocabulary) {
    const std::string normalized_word = NormalizeToken(word);
    if (word_to_id_.count(normalized_word) != 0) {
      err_msg_ =
          "[WordTokenizer::InitFromVocabulary] Duplicate word: " + normalized_word;
      return INVALID_DATA;
    }
    int token_id = 0;
    auto rc = AddWord(normalized_word, token_id);
    if (rc != SUCCESS) {
      return rc;
    }
  }

  is_init_ = true;
  return SUCCESS;
}

WordTokenizer::RC
WordTokenizer::InitFromVocabulary(const std::vector<std::string> &vocabulary) {
  Config config;
  config.unknown_policy = UNKNOWN_MAP_TO_UNK;
  return InitFromVocabulary(vocabulary, config);
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

WordTokenizer::RC
WordTokenizer::TokenizeFlatWithUnknown(const std::string &text,
                                       std::vector<int> &token_ids,
                                       int &unknown_count) {
  if (!is_init_) {
    err_msg_ =
        "[WordTokenizer::TokenizeFlatWithUnknown] WordTokenizer not init";
    return NOT_INIT;
  }
  const int unknown_id = Lookup("<unk>");
  if (unknown_id < 0) {
    err_msg_ =
        "[WordTokenizer::TokenizeFlatWithUnknown] Missing <unk> token";
    return INVALID_DATA;
  }

  token_ids.clear();
  unknown_count = 0;
  std::string current;
  auto flush_word = [&]() {
    if (current.empty()) {
      return;
    }
    const std::string word = NormalizeToken(current);
    auto found = word_to_id_.find(word);
    if (found == word_to_id_.end()) {
      token_ids.push_back(unknown_id);
      unknown_count++;
    } else {
      token_ids.push_back(found->second);
    }
    current.clear();
  };

  for (unsigned char ch : text) {
    if (IsWordChar(ch)) {
      current.push_back(static_cast<char>(ch));
    } else {
      flush_word();
    }
  }
  flush_word();

  if (token_ids.empty()) {
    err_msg_ = "[WordTokenizer::TokenizeFlatWithUnknown] No words found";
    return INVALID_DATA;
  }
  return SUCCESS;
}

WordTokenizer::RC
WordTokenizer::TokenizeFlatWithPolicy(const std::string &text,
                                      std::vector<int> &token_ids,
                                      int &unknown_count) {
  if (!is_init_) {
    err_msg_ =
        "[WordTokenizer::TokenizeFlatWithPolicy] WordTokenizer not init";
    return NOT_INIT;
  }
  if (unknown_policy_ == UNKNOWN_MAP_TO_UNK) {
    return TokenizeFlatWithUnknown(text, token_ids, unknown_count);
  }

  token_ids.clear();
  unknown_count = 0;
  std::string current;
  auto flush_word = [&]() {
    if (current.empty()) {
      return;
    }
    const std::string word = NormalizeToken(current);
    auto found = word_to_id_.find(word);
    if (found == word_to_id_.end()) {
      unknown_count++;
    } else {
      token_ids.push_back(found->second);
    }
    current.clear();
  };

  for (unsigned char ch : text) {
    if (IsWordChar(ch)) {
      current.push_back(static_cast<char>(ch));
    } else {
      flush_word();
    }
  }
  flush_word();

  if (token_ids.empty()) {
    err_msg_ = "[WordTokenizer::TokenizeFlatWithPolicy] No words found";
    return INVALID_DATA;
  }
  return SUCCESS;
}

WordTokenizer::RC WordTokenizer::Decode(const std::vector<int> &token_ids,
                                        std::string &text) {
  if (!is_init_) {
    err_msg_ = "[WordTokenizer::Decode] WordTokenizer not init";
    return NOT_INIT;
  }
  if (token_ids.empty()) {
    err_msg_ = "[WordTokenizer::Decode] Empty token input";
    return INVALID_DATA;
  }

  text.clear();
  for (int i = 0; i < static_cast<int>(token_ids.size()); i++) {
    const int token_id = token_ids[i];
    if (token_id < 0 || token_id >= static_cast<int>(vocabulary_.size())) {
      err_msg_ = "[WordTokenizer::Decode] Invalid token input";
      return INVALID_DATA;
    }
    if (i > 0) {
      text.push_back(' ');
    }
    text += vocabulary_[token_id];
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

WordTokenizer::UnknownPolicy WordTokenizer::unknown_policy() const {
  return unknown_policy_;
}

const std::vector<std::string> &WordTokenizer::vocabulary() const {
  return vocabulary_;
}

} // namespace deeplearning

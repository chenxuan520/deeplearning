#include "embedding/word_tokenizer.h"
#include "lr_scheduler/warmup_cosine_lr.h"
#include "transformer/character_dataset.h"
#include "transformer/character_tokenizer.h"
#include "transformer/mini_transformer_lm.h"
#include "transformer/mini_transformer_lm_loader.h"
#include "transformer/utf8_char_tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace deeplearning;

namespace fs = std::filesystem;

namespace {

volatile std::sig_atomic_t g_stop_requested = 0;

void HandleStopSignal(int) { g_stop_requested = 1; }

bool StopRequested() { return g_stop_requested != 0; }

constexpr int kRecentLossWindowSize = 100;

// A base-model style CLI around MiniTransformerLM with four sub-commands:
//   init      define structure + build vocab from a corpus + save a fresh model
//   train     load a model and train/continue on a text file or directory
//   generate  load a model and generate text from a prompt
//   info      print a model's structure and vocabulary
//
// A model is stored as two sidecar files sharing the same base path:
//   <model>        binary weights + structure (MiniTransformerLMLoader format)
//   <model>.vocab  tokenizer metadata + vocabulary (this demo's own file)
// The vocabulary is required to map characters to token ids, so it is persisted
// alongside the weights instead of being rebuilt from the training corpus.

enum TokenizerKind {
  TOKENIZER_CHAR,
  TOKENIZER_UTF8_CHAR,
  TOKENIZER_WORD,
};

struct Option {
  string model = "mini_lm.param";
  string corpus = "abcabcabcabcabcabc";
  string corpus_file;
  string corpus_dir;
  string prompt = "ab";
  int generate_num = 20;
  int epoch_num = 800;
  int rand_seed = 0;
  int model_dim = 6;
  int head_num = 1;
  int feed_forward_dim = 12;
  int block_num = 2;
  int context_size = 2;
  int max_vocab_size = 0;
  int log_every = 1;
  int checkpoint_every = 1;
  int progress_every_sec = 5;
  int batch_size = 1;
  int thread_num = 1;
  double learning_rate = 0.01;
  double early_stop_loss = 0.02;
  double temperature = 1.0;
  int top_k = 0;
  double top_p = 1.0;
  string backbone = "decoder";
  string tokenizer = "char";
  string unknown_policy = "map";
  string checkpoint;
  bool tokenizer_specified = false;
  bool max_vocab_size_specified = false;
  bool unknown_policy_specified = false;
  bool train_control_specified = false;
  bool resume_checkpoint = false;
  bool no_checkpoint = false;
  bool skip_final_eval = false;
  bool sampling_specified = false;
};

struct TokenizerBuildOption {
  TokenizerKind kind = TOKENIZER_CHAR;
  WordTokenizer::Config word_config;
};

void PrintTopUsage(const char *prog) {
  cout << "Usage: " << prog << " <command> [options]\n"
       << "Commands:\n"
       << "  init      Create a base model: define structure and build the "
          "vocab from a corpus\n"
       << "  train     Load a model and train/continue on a text file or "
          "directory\n"
       << "  generate  Load a model and generate text from a prompt\n"
       << "  info      Print a model's structure and vocabulary\n"
       << "Run '" << prog << " <command> --help' for command-specific options.\n";
}

void PrintCommandUsage(const char *prog, const string &verb) {
  cout << "Usage: " << prog << " " << verb << " [options]\n";
  if (verb == "init") {
    cout << "  --model <path>              output model path (default "
            "mini_lm.param)\n"
         << "  --tokenizer <char|utf8-char|word>\n"
         << "                              tokenizer granularity (default char)\n"
         << "  --max-vocab-size <int>      for word tokenizer, keep top-N words "
            "plus <unk> (0 = all)\n"
         << "  --unknown-policy <map|drop> for word tokenizer, map OOV to "
            "<unk> or drop OOV (default map)\n"
         << "  --corpus <text>             inline corpus used to build the "
            "vocab\n"
         << "  --corpus-file <path>        read the vocab corpus from a file\n"
         << "  --corpus-dir <dir>          read the vocab corpus from all files "
            "in a directory\n"
         << "  --backbone <encoder|decoder>\n"
         << "  --model-dim <int>\n"
         << "  --head-num <int>            must divide model-dim\n"
         << "  --feed-forward-dim <int>\n"
         << "  --block-num <int>           number of stacked Transformer "
            "blocks\n"
         << "  --context-size <int>        training window / max context\n"
         << "  --rand-seed <int>\n";
  } else if (verb == "train") {
    cout << "  --model <path>              model to load and update (default "
            "mini_lm.param)\n"
         << "  --corpus <text>             inline training corpus\n"
         << "  --corpus-file <path>        read training corpus from a file\n"
         << "  --corpus-dir <dir>          read training corpus from all files "
            "in a directory\n"
         << "  --epochs <int>\n"
         << "  --batch-size <int>          train with a minibatch size "
            "(default 1)\n"
         << "  --thread-num <int>          worker threads for minibatch "
            "gradient calculation (default 1)\n"
         << "  --learning-rate <double>\n";
    cout << "  --log-every <int>           print epoch summary every N epochs "
            "(default 1)\n"
         << "  --progress-every-sec <int>  print in-epoch progress every N "
            "seconds (0 = off, default 5)\n"
         << "  --early-stop-loss <double>  stop after an epoch below this loss "
            "(<=0 = off, default 0.02)\n"
         << "  --checkpoint <path>         checkpoint path (default "
            "<model>.ckpt)\n"
         << "  --checkpoint-every <int>    save checkpoint every N completed "
            "epochs (default 1)\n"
         << "  --resume-checkpoint         load the checkpoint and continue "
            "toward --epochs\n"
         << "  --no-checkpoint             disable periodic checkpoint writes\n"
         << "  --skip-final-eval           save immediately after training "
            "without the final full-corpus loss/perplexity pass\n";
  } else if (verb == "generate") {
    cout << "  --model <path>              model to load (default "
            "mini_lm.param)\n"
         << "  --prompt <text>\n"
         << "  --generate-num <int>\n"
         << "  --temperature <double>      enable sampling (default greedy)\n"
         << "  --top-k <int>               enable sampling\n"
         << "  --top-p <double>            enable sampling\n";
  } else if (verb == "info") {
    cout << "  --model <path>              model to inspect (default "
            "mini_lm.param)\n";
  }
}

// Parses flags starting at argv[start]. Returns false when --help is seen so the
// caller can print usage and exit cleanly; throws on missing values / unknown
// flags, mirroring the other demos in this repo.
bool ParseArgs(int argc, char **argv, int start, Option &option) {
  for (int i = start; i < argc; i++) {
    string arg = argv[i];
    auto need_value = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        throw std::runtime_error(string("Missing value for ") + name);
      }
      i += 1;
      return argv[i];
    };

    if (arg == "--model") {
      option.model = need_value("--model");
    } else if (arg == "--corpus") {
      option.corpus = need_value("--corpus");
    } else if (arg == "--corpus-file") {
      option.corpus_file = need_value("--corpus-file");
    } else if (arg == "--corpus-dir") {
      option.corpus_dir = need_value("--corpus-dir");
    } else if (arg == "--prompt") {
      option.prompt = need_value("--prompt");
    } else if (arg == "--generate-num") {
      option.generate_num = std::stoi(need_value("--generate-num"));
    } else if (arg == "--epochs") {
      option.epoch_num = std::stoi(need_value("--epochs"));
    } else if (arg == "--batch-size") {
      option.batch_size = std::stoi(need_value("--batch-size"));
    } else if (arg == "--thread-num") {
      option.thread_num = std::stoi(need_value("--thread-num"));
    } else if (arg == "--learning-rate") {
      option.learning_rate = std::stod(need_value("--learning-rate"));
    } else if (arg == "--log-every") {
      option.log_every = std::stoi(need_value("--log-every"));
      option.train_control_specified = true;
    } else if (arg == "--progress-every-sec") {
      option.progress_every_sec =
          std::stoi(need_value("--progress-every-sec"));
      option.train_control_specified = true;
    } else if (arg == "--early-stop-loss") {
      option.early_stop_loss = std::stod(need_value("--early-stop-loss"));
      option.train_control_specified = true;
    } else if (arg == "--checkpoint") {
      option.checkpoint = need_value("--checkpoint");
      option.train_control_specified = true;
    } else if (arg == "--checkpoint-every") {
      option.checkpoint_every = std::stoi(need_value("--checkpoint-every"));
      option.train_control_specified = true;
    } else if (arg == "--resume-checkpoint") {
      option.resume_checkpoint = true;
      option.train_control_specified = true;
    } else if (arg == "--no-checkpoint") {
      option.no_checkpoint = true;
      option.train_control_specified = true;
    } else if (arg == "--skip-final-eval") {
      option.skip_final_eval = true;
      option.train_control_specified = true;
    } else if (arg == "--rand-seed") {
      option.rand_seed = std::stoi(need_value("--rand-seed"));
    } else if (arg == "--model-dim") {
      option.model_dim = std::stoi(need_value("--model-dim"));
    } else if (arg == "--head-num") {
      option.head_num = std::stoi(need_value("--head-num"));
    } else if (arg == "--feed-forward-dim") {
      option.feed_forward_dim = std::stoi(need_value("--feed-forward-dim"));
    } else if (arg == "--block-num") {
      option.block_num = std::stoi(need_value("--block-num"));
    } else if (arg == "--context-size") {
      option.context_size = std::stoi(need_value("--context-size"));
    } else if (arg == "--max-vocab-size") {
      option.max_vocab_size = std::stoi(need_value("--max-vocab-size"));
      option.max_vocab_size_specified = true;
    } else if (arg == "--unknown-policy") {
      option.unknown_policy = need_value("--unknown-policy");
      option.unknown_policy_specified = true;
    } else if (arg == "--temperature") {
      option.temperature = std::stod(need_value("--temperature"));
      option.sampling_specified = true;
    } else if (arg == "--top-k") {
      option.top_k = std::stoi(need_value("--top-k"));
      option.sampling_specified = true;
    } else if (arg == "--top-p") {
      option.top_p = std::stod(need_value("--top-p"));
      option.sampling_specified = true;
    } else if (arg == "--backbone") {
      option.backbone = need_value("--backbone");
    } else if (arg == "--tokenizer") {
      option.tokenizer = need_value("--tokenizer");
      option.tokenizer_specified = true;
    } else if (arg == "--help") {
      return false;
    } else {
      throw std::runtime_error(string("Unknown option: ") + arg);
    }
  }
  return true;
}

string VocabPath(const string &model_path) { return model_path + ".vocab"; }

string DefaultCheckpointPath(const string &model_path) {
  return model_path + ".ckpt";
}

string CheckpointMetaPath(const string &checkpoint_path) {
  return checkpoint_path + ".train";
}

bool ReadFileRaw(const string &filename, string &content) {
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs.is_open()) {
    return false;
  }
  content.assign((std::istreambuf_iterator<char>(ifs)),
                 std::istreambuf_iterator<char>());
  return true;
}

void StripTrailingNewlines(string &text) {
  while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) {
    text.pop_back();
  }
}

string TrimLineEnd(string line) {
  while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
    line.pop_back();
  }
  return line;
}

// Reads every regular file under dir (recursively) sorted by path, so the
// assembled corpus is reproducible, and joins them with newlines.
bool ReadCorpusDir(const string &dir, string &corpus, string &err) {
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) {
    err = "Corpus dir is not a directory: " + dir;
    return false;
  }
  vector<fs::path> files;
  for (fs::recursive_directory_iterator it(dir, ec), end; it != end;
       it.increment(ec)) {
    if (ec) {
      err = "Walk corpus dir failed: " + dir;
      return false;
    }
    if (it->is_regular_file(ec)) {
      files.push_back(it->path());
    }
  }
  std::sort(files.begin(), files.end());
  if (files.empty()) {
    err = "No files found under corpus dir: " + dir;
    return false;
  }

  corpus.clear();
  for (const auto &file : files) {
    string content;
    if (!ReadFileRaw(file.string(), content)) {
      err = "Read corpus file failed: " + file.string();
      return false;
    }
    StripTrailingNewlines(content);
    if (content.empty()) {
      continue;
    }
    if (!corpus.empty()) {
      corpus.push_back('\n');
    }
    corpus += content;
  }
  return true;
}

// Resolves the corpus text from --corpus-dir / --corpus-file / --corpus in that
// order and reports which source was used.
bool CollectCorpus(const Option &option, string &corpus, string &source,
                   string &err) {
  if (!option.corpus_dir.empty()) {
    source = "dir:" + option.corpus_dir;
    return ReadCorpusDir(option.corpus_dir, corpus, err);
  }
  if (!option.corpus_file.empty()) {
    if (!ReadFileRaw(option.corpus_file, corpus)) {
      err = "Read corpus file failed: " + option.corpus_file;
      return false;
    }
    StripTrailingNewlines(corpus);
    source = "file:" + option.corpus_file;
    return true;
  }
  corpus = option.corpus;
  source = "inline";
  return true;
}

// Drops characters that are not in the vocabulary so text from a fresh corpus
// can still be encoded by a model whose vocab was fixed at init time.
string FilterToVocab(const string &vocabulary, const string &text,
                     long long &dropped) {
  vector<bool> present(256, false);
  for (char ch : vocabulary) {
    present[static_cast<unsigned char>(ch)] = true;
  }
  string filtered;
  filtered.reserve(text.size());
  dropped = 0;
  for (char ch : text) {
    if (present[static_cast<unsigned char>(ch)]) {
      filtered.push_back(ch);
    } else {
      dropped++;
    }
  }
  return filtered;
}

TokenizerKind ParseTokenizerKind(const string &tokenizer) {
  if (tokenizer == "char") {
    return TOKENIZER_CHAR;
  }
  if (tokenizer == "utf8-char") {
    return TOKENIZER_UTF8_CHAR;
  }
  if (tokenizer == "word") {
    return TOKENIZER_WORD;
  }
  throw std::runtime_error("Invalid tokenizer, expected char, utf8-char or word");
}

const char *TokenizerName(TokenizerKind tokenizer_kind) {
  if (tokenizer_kind == TOKENIZER_WORD) {
    return "word";
  }
  if (tokenizer_kind == TOKENIZER_UTF8_CHAR) {
    return "utf8-char";
  }
  return "char";
}

WordTokenizer::UnknownPolicy ParseUnknownPolicy(const string &policy) {
  if (policy == "map") {
    return WordTokenizer::UNKNOWN_MAP_TO_UNK;
  }
  if (policy == "drop") {
    return WordTokenizer::UNKNOWN_DROP;
  }
  throw std::runtime_error("Invalid unknown-policy, expected map or drop");
}

const char *UnknownPolicyName(WordTokenizer::UnknownPolicy policy) {
  return policy == WordTokenizer::UNKNOWN_DROP ? "drop" : "map";
}

struct TokenizerBundle {
  TokenizerKind kind = TOKENIZER_CHAR;
  CharacterTokenizer char_tokenizer;
  Utf8CharTokenizer utf8_char_tokenizer;
  WordTokenizer word_tokenizer;
};

struct CheckpointMeta {
  int completed_epoch = 0;
  int target_epoch = 0;
  double learning_rate = 0.0;
  double last_loss = 0.0;
  double last_perplexity = 0.0;
  string source;
};

class RecentLossTracker {
public:
  explicit RecentLossTracker(int window_size) : window_size_(window_size) {}

  void Reset() {
    losses_.clear();
    loss_sum_ = 0.0;
    last_finished_sample_num_ = 0;
    last_cumulative_loss_sum_ = 0.0;
  }

  double AddCumulativeAverage(int finished_sample_num, double average_loss) {
    if (window_size_ <= 0) {
      return average_loss;
    }
    if (finished_sample_num <= last_finished_sample_num_) {
      Reset();
    }

    const int sample_delta = finished_sample_num - last_finished_sample_num_;
    const double cumulative_loss_sum =
        average_loss * static_cast<double>(finished_sample_num);
    if (sample_delta <= 0) {
      return Average();
    }

    const double sample_loss =
        (cumulative_loss_sum - last_cumulative_loss_sum_) / sample_delta;
    for (int i = 0; i < sample_delta; i++) {
      Push(sample_loss);
    }

    last_finished_sample_num_ = finished_sample_num;
    last_cumulative_loss_sum_ = cumulative_loss_sum;
    return Average();
  }

  double Average() const {
    if (losses_.empty()) {
      return 0.0;
    }
    return loss_sum_ / losses_.size();
  }

private:
  void Push(double loss) {
    losses_.push_back(loss);
    loss_sum_ += loss;
    while (static_cast<int>(losses_.size()) > window_size_) {
      loss_sum_ -= losses_.front();
      losses_.pop_front();
    }
  }

private:
  int window_size_ = 0;
  std::deque<double> losses_;
  double loss_sum_ = 0.0;
  int last_finished_sample_num_ = 0;
  double last_cumulative_loss_sum_ = 0.0;
};

double SecondsSince(std::chrono::steady_clock::time_point start,
                    std::chrono::steady_clock::time_point now) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(now - start)
      .count();
}

string FormatSeconds(double seconds) {
  if (seconds < 0) {
    seconds = 0;
  }
  long long total_seconds = static_cast<long long>(seconds + 0.5);
  long long hours = total_seconds / 3600;
  long long minutes = (total_seconds % 3600) / 60;
  long long secs = total_seconds % 60;
  std::ostringstream oss;
  if (hours > 0) {
    oss << hours << "h";
  }
  if (hours > 0 || minutes > 0) {
    oss << minutes << "m";
  }
  oss << secs << "s";
  return oss.str();
}

bool WriteCheckpointMeta(const string &checkpoint_path,
                         const CheckpointMeta &meta, string &err) {
  const string path = CheckpointMetaPath(checkpoint_path);
  std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
  if (!ofs.is_open()) {
    err = "Write checkpoint metadata failed: " + path;
    return false;
  }
  ofs << "completed_epoch=" << meta.completed_epoch << '\n'
      << "target_epoch=" << meta.target_epoch << '\n'
      << "learning_rate=" << meta.learning_rate << '\n'
      << "last_loss=" << meta.last_loss << '\n'
      << "last_perplexity=" << meta.last_perplexity << '\n'
      << "source=" << meta.source << '\n';
  if (!ofs.good()) {
    err = "Write checkpoint metadata failed: " + path;
    return false;
  }
  return true;
}

bool ReadCheckpointMeta(const string &checkpoint_path, CheckpointMeta &meta,
                        string &err) {
  const string path = CheckpointMetaPath(checkpoint_path);
  string content;
  if (!ReadFileRaw(path, content)) {
    err = "Checkpoint metadata not found: " + path;
    return false;
  }
  std::istringstream iss(content);
  string line;
  while (std::getline(iss, line)) {
    line = TrimLineEnd(line);
    if (line.empty()) {
      continue;
    }
    const auto pos = line.find('=');
    if (pos == string::npos) {
      err = "Invalid checkpoint metadata line: " + line;
      return false;
    }
    const string key = line.substr(0, pos);
    const string value = line.substr(pos + 1);
    try {
      if (key == "completed_epoch") {
        meta.completed_epoch = std::stoi(value);
      } else if (key == "target_epoch") {
        meta.target_epoch = std::stoi(value);
      } else if (key == "learning_rate") {
        meta.learning_rate = std::stod(value);
      } else if (key == "last_loss") {
        meta.last_loss = std::stod(value);
      } else if (key == "last_perplexity") {
        meta.last_perplexity = std::stod(value);
      } else if (key == "source") {
        meta.source = value;
      }
    } catch (const std::exception &ex) {
      err = "Invalid checkpoint metadata value for " + key + ": " + value;
      return false;
    }
  }
  if (meta.completed_epoch < 0 || meta.target_epoch < 0) {
    err = "Invalid checkpoint epoch metadata";
    return false;
  }
  return true;
}

string DisplayCharVocab(const string &vocabulary) {
  string out;
  for (char ch : vocabulary) {
    out.push_back('[');
    if (ch == '\n') {
      out += "\\n";
    } else if (ch == '\r') {
      out += "\\r";
    } else if (ch == '\t') {
      out += "\\t";
    } else {
      out.push_back(ch);
    }
    out.push_back(']');
  }
  return out;
}

string DisplayWordVocab(const vector<string> &vocabulary) {
  string out;
  for (const auto &word : vocabulary) {
    if (!out.empty()) {
      out += " ";
    }
    out.push_back('[');
    out += word;
    out.push_back(']');
  }
  return out;
}

string DisplayUtf8Vocab(const vector<string> &vocabulary) {
  string out;
  for (const auto &token : vocabulary) {
    if (!out.empty()) {
      out += " ";
    }
    out.push_back('[');
    if (token == "\n") {
      out += "\\n";
    } else if (token == "\r") {
      out += "\\r";
    } else if (token == "\t") {
      out += "\\t";
    } else {
      out += token;
    }
    out.push_back(']');
  }
  return out;
}

string HexEncode(const string &text) {
  std::ostringstream oss;
  oss << std::hex << std::setfill('0');
  for (unsigned char ch : text) {
    oss << std::setw(2) << static_cast<int>(ch);
  }
  return oss.str();
}

int HexValue(char ch) {
  if (ch >= '0' && ch <= '9') {
    return ch - '0';
  }
  if (ch >= 'a' && ch <= 'f') {
    return ch - 'a' + 10;
  }
  if (ch >= 'A' && ch <= 'F') {
    return ch - 'A' + 10;
  }
  return -1;
}

bool HexDecode(const string &hex_text, string &text) {
  if (hex_text.empty() || hex_text.size() % 2 != 0) {
    return false;
  }
  text.clear();
  text.reserve(hex_text.size() / 2);
  for (int i = 0; i < static_cast<int>(hex_text.size()); i += 2) {
    const int hi = HexValue(hex_text[i]);
    const int lo = HexValue(hex_text[i + 1]);
    if (hi < 0 || lo < 0) {
      return false;
    }
    text.push_back(static_cast<char>((hi << 4) | lo));
  }
  return true;
}

bool WriteVocabSidecar(const string &path, const TokenizerBundle &tokenizer,
                       string &err) {
  std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
  if (!ofs.is_open()) {
    err = "Write vocab sidecar failed: " + path;
    return false;
  }
  if (tokenizer.kind == TOKENIZER_CHAR) {
    ofs << tokenizer.char_tokenizer.vocabulary();
  } else if (tokenizer.kind == TOKENIZER_UTF8_CHAR) {
    ofs << "tokenizer=utf8-char\n";
    for (const auto &token : tokenizer.utf8_char_tokenizer.vocabulary()) {
      ofs << HexEncode(token) << '\n';
    }
  } else {
    ofs << "tokenizer=word\n";
    if (tokenizer.word_tokenizer.unknown_policy() ==
        WordTokenizer::UNKNOWN_DROP) {
      ofs << "unknown-policy=drop\n";
    }
    for (const auto &word : tokenizer.word_tokenizer.vocabulary()) {
      ofs << word << '\n';
    }
  }
  if (!ofs.good()) {
    err = "Write vocab sidecar failed: " + path;
    return false;
  }
  return true;
}

bool InitTokenizerFromCorpus(const TokenizerBuildOption &build_option,
                             const string &corpus, TokenizerBundle &tokenizer,
                             string &err) {
  if (build_option.word_config.max_vocab_size < 0) {
    err = "max-vocab-size must be >= 0";
    return false;
  }
  tokenizer.kind = build_option.kind;
  if (tokenizer.kind == TOKENIZER_CHAR || tokenizer.kind == TOKENIZER_UTF8_CHAR) {
    if (build_option.word_config.max_vocab_size != 0) {
      err = "--max-vocab-size is only supported with --tokenizer word";
      return false;
    }
    if (build_option.word_config.unknown_policy !=
        WordTokenizer::UNKNOWN_MAP_TO_UNK) {
      err = "--unknown-policy is only supported with --tokenizer word";
      return false;
    }
  }
  if (tokenizer.kind == TOKENIZER_CHAR) {
    const string vocabulary = CharacterTokenizer::BuildVocabularyFromText(corpus);
    if (tokenizer.char_tokenizer.Init(vocabulary) !=
        CharacterTokenizer::SUCCESS) {
      err = "Tokenizer init failed: " + tokenizer.char_tokenizer.err_msg();
      return false;
    }
    return true;
  }
  if (tokenizer.kind == TOKENIZER_UTF8_CHAR) {
    const auto vocabulary = Utf8CharTokenizer::BuildVocabularyFromText(corpus);
    if (tokenizer.utf8_char_tokenizer.Init(vocabulary) !=
        Utf8CharTokenizer::SUCCESS) {
      err = "Tokenizer init failed: " + tokenizer.utf8_char_tokenizer.err_msg();
      return false;
    }
    return true;
  }

  if (tokenizer.word_tokenizer.InitFromText(corpus,
                                            build_option.word_config) !=
      WordTokenizer::SUCCESS) {
    err = "Tokenizer init failed: " + tokenizer.word_tokenizer.err_msg();
    return false;
  }
  return true;
}

bool InitTokenizerFromSidecar(const string &content, TokenizerBundle &tokenizer,
                              string &err) {
  if (content.rfind("tokenizer=", 0) != 0) {
    tokenizer.kind = TOKENIZER_CHAR;
    if (tokenizer.char_tokenizer.Init(content) != CharacterTokenizer::SUCCESS) {
      err = "Tokenizer init failed: " + tokenizer.char_tokenizer.err_msg();
      return false;
    }
    return true;
  }

  std::istringstream iss(content);
  string header;
  std::getline(iss, header);
  header = TrimLineEnd(header);
  const string tokenizer_name = header.substr(string("tokenizer=").size());
  try {
    tokenizer.kind = ParseTokenizerKind(tokenizer_name);
  } catch (const std::exception &ex) {
    err = ex.what();
    return false;
  }

  string rest((std::istreambuf_iterator<char>(iss)),
              std::istreambuf_iterator<char>());
  if (tokenizer.kind == TOKENIZER_CHAR) {
    StripTrailingNewlines(rest);
    if (tokenizer.char_tokenizer.Init(rest) != CharacterTokenizer::SUCCESS) {
      err = "Tokenizer init failed: " + tokenizer.char_tokenizer.err_msg();
      return false;
    }
    return true;
  }
  if (tokenizer.kind == TOKENIZER_UTF8_CHAR) {
    vector<string> vocabulary;
    string line;
    std::istringstream vocab_stream(rest);
    while (std::getline(vocab_stream, line)) {
      line = TrimLineEnd(line);
      if (line.empty()) {
        continue;
      }
      string token;
      if (!HexDecode(line, token)) {
        err = "Invalid utf8-char vocab sidecar";
        return false;
      }
      vocabulary.push_back(token);
    }
    if (tokenizer.utf8_char_tokenizer.Init(vocabulary) !=
        Utf8CharTokenizer::SUCCESS) {
      err = "Tokenizer init failed: " + tokenizer.utf8_char_tokenizer.err_msg();
      return false;
    }
    return true;
  }

  vector<string> vocabulary;
  string line;
  std::istringstream vocab_stream(rest);
  WordTokenizer::Config word_config;
  word_config.unknown_policy = WordTokenizer::UNKNOWN_MAP_TO_UNK;
  bool unknown_policy_set = false;
  while (std::getline(vocab_stream, line)) {
    line = TrimLineEnd(line);
    if (line.rfind("unknown-policy=", 0) == 0) {
      const string policy_name = line.substr(string("unknown-policy=").size());
      try {
        word_config.unknown_policy = ParseUnknownPolicy(policy_name);
      } catch (const std::exception &ex) {
        err = ex.what();
        return false;
      }
      unknown_policy_set = true;
      continue;
    }
    if (!line.empty()) {
      vocabulary.push_back(line);
    }
  }
  if (!unknown_policy_set) {
    bool has_unknown = false;
    for (const auto &word : vocabulary) {
      if (word == "<unk>") {
        has_unknown = true;
        break;
      }
    }
    word_config.unknown_policy =
        has_unknown ? WordTokenizer::UNKNOWN_MAP_TO_UNK : WordTokenizer::UNKNOWN_DROP;
  }
  if (tokenizer.word_tokenizer.InitFromVocabulary(vocabulary, word_config) !=
      WordTokenizer::SUCCESS) {
    err = "Tokenizer init failed: " + tokenizer.word_tokenizer.err_msg();
    return false;
  }
  if (tokenizer.word_tokenizer.unknown_policy() ==
          WordTokenizer::UNKNOWN_MAP_TO_UNK &&
      tokenizer.word_tokenizer.Lookup("<unk>") < 0) {
    err = "Word vocab sidecar must contain <unk>";
    return false;
  }
  return true;
}

int TokenizerVocabSize(const TokenizerBundle &tokenizer) {
  if (tokenizer.kind == TOKENIZER_WORD) {
    return tokenizer.word_tokenizer.vocab_size();
  }
  if (tokenizer.kind == TOKENIZER_UTF8_CHAR) {
    return tokenizer.utf8_char_tokenizer.vocab_size();
  }
  return tokenizer.char_tokenizer.vocab_size();
}

bool EncodeForTraining(TokenizerBundle &tokenizer, const string &text,
                       vector<int> &token_ids, int &unknown_count,
                       string &err) {
  unknown_count = 0;
  if (tokenizer.kind == TOKENIZER_WORD) {
    if (tokenizer.word_tokenizer.TokenizeFlatWithPolicy(
            text, token_ids, unknown_count) != WordTokenizer::SUCCESS) {
      err = "Encode corpus failed: " + tokenizer.word_tokenizer.err_msg();
      return false;
    }
    return true;
  }
  if (tokenizer.kind == TOKENIZER_UTF8_CHAR) {
    vector<string> text_tokens;
    if (!Utf8CharTokenizer::SplitUtf8(text, text_tokens)) {
      err = "Encode corpus failed: invalid UTF-8 text";
      return false;
    }
    string filtered;
    for (const auto &token : text_tokens) {
      const int token_id =
          tokenizer.utf8_char_tokenizer.Lookup(token);
      if (token_id < 0) {
        unknown_count++;
        continue;
      }
      filtered += token;
    }
    if (filtered.empty()) {
      err = "No trainable text left after filtering to the model vocabulary";
      return false;
    }
    if (tokenizer.utf8_char_tokenizer.Encode(filtered, token_ids) !=
        Utf8CharTokenizer::SUCCESS) {
      err = "Encode corpus failed: " + tokenizer.utf8_char_tokenizer.err_msg();
      return false;
    }
    return true;
  }

  long long dropped = 0;
  const string filtered =
      FilterToVocab(tokenizer.char_tokenizer.vocabulary(), text, dropped);
  unknown_count = static_cast<int>(dropped);
  if (filtered.empty()) {
    err = "No trainable text left after filtering to the model vocabulary";
    return false;
  }
  if (tokenizer.char_tokenizer.Encode(filtered, token_ids) !=
      CharacterTokenizer::SUCCESS) {
    err = "Encode corpus failed: " + tokenizer.char_tokenizer.err_msg();
    return false;
  }
  return true;
}

bool EncodePrompt(TokenizerBundle &tokenizer, const string &text,
                  vector<int> &token_ids, int &unknown_count, string &prompt,
                  string &err) {
  unknown_count = 0;
  if (tokenizer.kind == TOKENIZER_WORD) {
    if (tokenizer.word_tokenizer.TokenizeFlatWithPolicy(
            text, token_ids, unknown_count) != WordTokenizer::SUCCESS) {
      err = "Encode prompt failed: " + tokenizer.word_tokenizer.err_msg();
      return false;
    }
    if (tokenizer.word_tokenizer.Decode(token_ids, prompt) !=
        WordTokenizer::SUCCESS) {
      err = "Decode prompt failed: " + tokenizer.word_tokenizer.err_msg();
      return false;
    }
    return true;
  }
  if (tokenizer.kind == TOKENIZER_UTF8_CHAR) {
    vector<string> text_tokens;
    if (!Utf8CharTokenizer::SplitUtf8(text, text_tokens)) {
      err = "Encode prompt failed: invalid UTF-8 text";
      return false;
    }
    prompt.clear();
    for (const auto &token : text_tokens) {
      const int token_id =
          tokenizer.utf8_char_tokenizer.Lookup(token);
      if (token_id < 0) {
        unknown_count++;
        continue;
      }
      prompt += token;
    }
    if (prompt.empty()) {
      err = "Prompt is empty after filtering to the model vocabulary";
      return false;
    }
    if (tokenizer.utf8_char_tokenizer.Encode(prompt, token_ids) !=
        Utf8CharTokenizer::SUCCESS) {
      err = "Encode prompt failed: " + tokenizer.utf8_char_tokenizer.err_msg();
      return false;
    }
    return true;
  }

  long long dropped = 0;
  prompt = FilterToVocab(tokenizer.char_tokenizer.vocabulary(), text, dropped);
  unknown_count = static_cast<int>(dropped);
  if (prompt.empty()) {
    err = "Prompt is empty after filtering to the model vocabulary";
    return false;
  }
  if (tokenizer.char_tokenizer.Encode(prompt, token_ids) !=
      CharacterTokenizer::SUCCESS) {
    err = "Encode prompt failed: " + tokenizer.char_tokenizer.err_msg();
    return false;
  }
  return true;
}

bool DecodeGenerated(TokenizerBundle &tokenizer, const vector<int> &token_ids,
                     string &text, string &err) {
  if (tokenizer.kind == TOKENIZER_WORD) {
    if (tokenizer.word_tokenizer.Decode(token_ids, text) !=
        WordTokenizer::SUCCESS) {
      err = "Decode generated text failed: " + tokenizer.word_tokenizer.err_msg();
      return false;
    }
    return true;
  }
  if (tokenizer.kind == TOKENIZER_UTF8_CHAR) {
    if (tokenizer.utf8_char_tokenizer.Decode(token_ids, text) !=
        Utf8CharTokenizer::SUCCESS) {
      err = "Decode generated text failed: " +
            tokenizer.utf8_char_tokenizer.err_msg();
      return false;
    }
    return true;
  }
  if (tokenizer.char_tokenizer.Decode(token_ids, text) !=
      CharacterTokenizer::SUCCESS) {
    err = "Decode generated text failed: " + tokenizer.char_tokenizer.err_msg();
    return false;
  }
  return true;
}

string DisplayTokenizerVocab(const TokenizerBundle &tokenizer) {
  if (tokenizer.kind == TOKENIZER_WORD) {
    return DisplayWordVocab(tokenizer.word_tokenizer.vocabulary());
  }
  if (tokenizer.kind == TOKENIZER_UTF8_CHAR) {
    return DisplayUtf8Vocab(tokenizer.utf8_char_tokenizer.vocabulary());
  }
  return DisplayCharVocab(tokenizer.char_tokenizer.vocabulary());
}

MiniTransformerLM::BackboneType ParseBackboneType(const string &backbone) {
  if (backbone == "encoder") {
    return MiniTransformerLM::BACKBONE_ENCODER;
  }
  if (backbone == "decoder") {
    return MiniTransformerLM::BACKBONE_DECODER;
  }
  throw std::runtime_error("Invalid backbone, expected encoder or decoder");
}

const char *BackboneName(MiniTransformerLM::BackboneType type) {
  return type == MiniTransformerLM::BACKBONE_DECODER ? "decoder" : "encoder";
}

// Loads the weights and the vocab sidecar produced by 'init'.
bool LoadModel(const string &model_path, MiniTransformerLM &model,
               TokenizerBundle &tokenizer, string &err) {
  std::ifstream probe(model_path, std::ios::binary);
  if (!probe.good()) {
    err = "Model file not found: " + model_path;
    return false;
  }
  probe.close();

  string vocab_content;
  if (!ReadFileRaw(VocabPath(model_path), vocab_content) ||
      vocab_content.empty()) {
    err = "Vocab sidecar not found or empty: " + VocabPath(model_path) +
          " (was this model created by 'mini_lm init'?)";
    return false;
  }
  if (!InitTokenizerFromSidecar(vocab_content, tokenizer, err)) {
    return false;
  }
  if (MiniTransformerLMLoader::ImportModelFromFile(model, model_path) !=
      MiniTransformerLMLoader::SUCCESS) {
    err = "Import model failed: " + model_path;
    return false;
  }
  if (model.vocab_size() != TokenizerVocabSize(tokenizer)) {
    err = "Model vocab size does not match the vocab sidecar";
    return false;
  }
  return true;
}

bool SaveModelWithVocab(const string &model_path, const MiniTransformerLM &model,
                        const TokenizerBundle &tokenizer, string &err) {
  if (MiniTransformerLMLoader::ExportModelToFile(model, model_path) !=
      MiniTransformerLMLoader::SUCCESS) {
    err = "Export model failed: " + model_path;
    return false;
  }
  if (!WriteVocabSidecar(VocabPath(model_path), tokenizer, err)) {
    return false;
  }
  return true;
}

bool SaveTrainingCheckpoint(const string &checkpoint_path,
                            const MiniTransformerLM &model,
                            const TokenizerBundle &tokenizer,
                            const CheckpointMeta &meta, string &err) {
  if (!SaveModelWithVocab(checkpoint_path, model, tokenizer, err)) {
    return false;
  }
  if (!WriteCheckpointMeta(checkpoint_path, meta, err)) {
    return false;
  }
  return true;
}

int RunInit(const Option &option) {
  if (option.model_dim <= 0 || option.head_num <= 0 ||
      option.feed_forward_dim <= 0 || option.block_num < 0 ||
      option.context_size <= 0) {
    cout << "Invalid structure option (need positive dims and block-num >= 0)"
         << endl;
    return -1;
  }
  if (option.model_dim % option.head_num != 0) {
    cout << "model-dim must be divisible by head-num" << endl;
    return -1;
  }

  string corpus;
  string source;
  string err;
  if (!CollectCorpus(option, corpus, source, err)) {
    cout << err << endl;
    return -1;
  }
  if (corpus.empty()) {
    cout << "Corpus is empty" << endl;
    return -1;
  }

  TokenizerBuildOption tokenizer_build_option;
  try {
    tokenizer_build_option.kind = ParseTokenizerKind(option.tokenizer);
  } catch (const std::exception &ex) {
    cout << ex.what() << endl;
    return -1;
  }
  tokenizer_build_option.word_config.max_vocab_size = option.max_vocab_size;
  try {
    tokenizer_build_option.word_config.unknown_policy =
        ParseUnknownPolicy(option.unknown_policy);
  } catch (const std::exception &ex) {
    cout << ex.what() << endl;
    return -1;
  }
  tokenizer_build_option.word_config.add_unknown_token =
      tokenizer_build_option.word_config.unknown_policy ==
      WordTokenizer::UNKNOWN_MAP_TO_UNK;
  TokenizerBundle tokenizer;
  if (!InitTokenizerFromCorpus(tokenizer_build_option, corpus, tokenizer, err)) {
    cout << err << endl;
    return -1;
  }

  MiniTransformerLM::Config config;
  config.vocab_size_ = TokenizerVocabSize(tokenizer);
  config.model_dim_ = option.model_dim;
  config.head_num_ = option.head_num;
  config.feed_forward_dim_ = option.feed_forward_dim;
  config.block_num_ = option.block_num;
  config.rand_seed_ = option.rand_seed;
  config.max_context_size_ = option.context_size;
  config.use_positional_encoding_ = true;
  config.scale_embedding_ = true;
  config.block_learning_rate_scale_ = 1.0;
  try {
    config.backbone_type_ = ParseBackboneType(option.backbone);
  } catch (const std::exception &ex) {
    cout << ex.what() << endl;
    return -1;
  }

  MiniTransformerLM model;
  if (model.Init(config) != MiniTransformerLM::SUCCESS) {
    cout << "Model init failed: " << model.err_msg() << endl;
    return -1;
  }
  if (!SaveModelWithVocab(option.model, model, tokenizer, err)) {
    cout << err << endl;
    return -1;
  }

  cout << "Initialized model: " << option.model << endl;
  cout << "Tokenizer: " << TokenizerName(tokenizer.kind) << endl;
  cout << "Vocab source: " << source
       << " vocab_size: " << TokenizerVocabSize(tokenizer) << endl;
  if (tokenizer.kind == TOKENIZER_WORD && option.max_vocab_size > 0) {
    cout << "Max vocab words: " << option.max_vocab_size
         << (tokenizer.word_tokenizer.unknown_policy() ==
                     WordTokenizer::UNKNOWN_MAP_TO_UNK
                 ? " (+ <unk>)"
                 : "")
         << endl;
  }
  if (tokenizer.kind == TOKENIZER_WORD) {
    cout << "Unknown policy: "
         << UnknownPolicyName(tokenizer.word_tokenizer.unknown_policy())
         << endl;
  }
  cout << "Backbone: " << BackboneName(config.backbone_type_)
       << " model_dim: " << config.model_dim_
       << " head_num: " << config.head_num_
       << " feed_forward_dim: " << config.feed_forward_dim_
       << " block_num: " << config.block_num_
       << " context_size: " << config.max_context_size_ << endl;
  cout << "Saved weights + " << VocabPath(option.model) << endl;
  return 0;
}

int RunTrain(const Option &option) {
  std::signal(SIGINT, HandleStopSignal);
  std::signal(SIGTERM, HandleStopSignal);

  string checkpoint_path =
      option.checkpoint.empty() ? DefaultCheckpointPath(option.model)
                                : option.checkpoint;
  if (option.no_checkpoint && option.resume_checkpoint) {
    cout << "--resume-checkpoint cannot be used with --no-checkpoint" << endl;
    return -1;
  }
  if (option.log_every <= 0 || option.progress_every_sec < 0 ||
      option.checkpoint_every <= 0) {
    cout << "Invalid logging/checkpoint option" << endl;
    return -1;
  }

  MiniTransformerLM model;
  TokenizerBundle tokenizer;
  string err;
  const string load_path =
      option.resume_checkpoint ? checkpoint_path : option.model;
  if (!LoadModel(load_path, model, tokenizer, err)) {
    cout << err << endl;
    return -1;
  }
  CheckpointMeta resume_meta;
  int start_epoch = 0;
  if (option.resume_checkpoint) {
    if (!ReadCheckpointMeta(checkpoint_path, resume_meta, err)) {
      cout << err << endl;
      return -1;
    }
    start_epoch = resume_meta.completed_epoch;
  }
  const int context_size = model.max_context_size();
  if (context_size <= 0) {
    cout << "Loaded model has no context size; re-create it with 'init'" << endl;
    return -1;
  }
  if (option.epoch_num <= 0 || option.batch_size <= 0 ||
      option.thread_num <= 0 || option.learning_rate <= 0.0) {
    cout << "Invalid training option (epochs, batch-size, thread-num and "
            "learning-rate must be > 0)"
         << endl;
    return -1;
  }
  if (start_epoch >= option.epoch_num) {
    cout << "Checkpoint already reached target epochs: " << start_epoch
         << "/" << option.epoch_num << endl;
    if (!SaveModelWithVocab(option.model, model, tokenizer, err)) {
      cout << err << endl;
      return -1;
    }
    cout << "Saved checkpoint weights to model: " << option.model << endl;
    return 0;
  }

  string corpus;
  string source;
  if (!CollectCorpus(option, corpus, source, err)) {
    cout << err << endl;
    return -1;
  }

  vector<int> corpus_token_ids;
  int unknown_or_dropped = 0;
  if (!EncodeForTraining(tokenizer, corpus, corpus_token_ids,
                         unknown_or_dropped, err)) {
    cout << err << endl;
    return -1;
  }
  if (unknown_or_dropped > 0) {
    if (tokenizer.kind == TOKENIZER_WORD) {
      if (tokenizer.word_tokenizer.unknown_policy() ==
          WordTokenizer::UNKNOWN_DROP) {
        cout << "Warning: dropped " << unknown_or_dropped
             << " words outside the model vocabulary" << endl;
      } else {
        cout << "Warning: mapped " << unknown_or_dropped
             << " words outside the model vocabulary to <unk>" << endl;
      }
    } else {
      cout << "Warning: skipped " << unknown_or_dropped
           << " characters outside the model vocabulary" << endl;
    }
  }

  CharacterDataset dataset;
  if (dataset.Init(corpus_token_ids, context_size) !=
      CharacterDataset::SUCCESS) {
    cout << "Dataset init failed: " << dataset.err_msg()
         << " (need more than " << context_size << " in-vocab tokens)"
         << endl;
    return -1;
  }
  vector<vector<int>> input_samples;
  vector<int> target_tokens;
  if (dataset.BuildNextTokenSamples(input_samples, target_tokens) !=
      CharacterDataset::SUCCESS) {
    cout << "BuildNextTokenSamples failed: " << dataset.err_msg() << endl;
    return -1;
  }

  int warmup_epochs = std::max(1, option.epoch_num / 20);
  WarmupCosineLR lr_scheduler(option.learning_rate, warmup_epochs,
                              option.epoch_num, option.learning_rate * 0.1);

  cout << "Training on " << source << " (" << input_samples.size()
       << " samples, context " << context_size << ")" << endl;
  cout << "Tokenizer: " << TokenizerName(tokenizer.kind)
       << " vocab_size: " << TokenizerVocabSize(tokenizer) << endl;
  cout << "Epochs: " << start_epoch << " -> " << option.epoch_num
       << " batch_size: " << option.batch_size
       << " thread_num: " << option.thread_num
       << " learning_rate: " << option.learning_rate
       << " warmup_epochs: " << warmup_epochs << endl;
  if (option.no_checkpoint) {
    cout << "Checkpoint: disabled" << endl;
  } else {
    cout << "Checkpoint: " << checkpoint_path << " every "
         << option.checkpoint_every << " epoch(s)" << endl;
  }
  cout << "Early stop loss: "
       << (option.early_stop_loss > 0 ? std::to_string(option.early_stop_loss)
                                      : string("off"))
       << endl;
  if (option.resume_checkpoint) {
    cout << "Resumed checkpoint: completed_epoch=" << start_epoch
         << " last_loss=" << resume_meta.last_loss
         << " last_perplexity=" << resume_meta.last_perplexity << endl;
  }

  const auto train_start = std::chrono::steady_clock::now();
  auto last_progress_time = train_start;
  double last_loss = 0.0;
  double recent_loss = 0.0;
  bool interrupted = false;
  bool stopped_by_loss = false;
  int completed_epoch = start_epoch;
  RecentLossTracker recent_loss_tracker(kRecentLossWindowSize);

  for (int global_epoch = start_epoch; global_epoch < option.epoch_num;
       global_epoch++) {
    const double current_lr = lr_scheduler.GetLR(global_epoch);
    auto epoch_start = std::chrono::steady_clock::now();
    recent_loss_tracker.Reset();
    bool stop_after_epoch = false;
    bool stop_during_epoch = false;

    auto sample_callback = [&](int, int finished_sample_num, int sample_num,
                               double average_loss, bool &early_stop) {
      last_loss = average_loss;
      recent_loss =
          recent_loss_tracker.AddCumulativeAverage(finished_sample_num,
                                                   average_loss);
      if (StopRequested()) {
        early_stop = true;
        stop_during_epoch = true;
      }
      if (option.progress_every_sec == 0 || finished_sample_num == sample_num) {
        return;
      }
      const auto now = std::chrono::steady_clock::now();
      if (SecondsSince(last_progress_time, now) <
          option.progress_every_sec) {
        return;
      }
      last_progress_time = now;
      const double epoch_elapsed = SecondsSince(epoch_start, now);
      const double samples_per_sec =
          epoch_elapsed > 0 ? finished_sample_num / epoch_elapsed : 0.0;
      const int total_samples = static_cast<int>(input_samples.size());
      const double epoch_eta =
          samples_per_sec > 0
              ? (total_samples - finished_sample_num) / samples_per_sec
              : 0.0;
      const long long done_steps =
          static_cast<long long>(global_epoch - start_epoch) * total_samples +
          finished_sample_num;
      const long long total_steps =
          static_cast<long long>(option.epoch_num - start_epoch) *
          total_samples;
      const double total_eta =
          done_steps > 0
              ? SecondsSince(train_start, now) *
                    (static_cast<double>(total_steps - done_steps) / done_steps)
              : 0.0;
      cout << "epoch " << (global_epoch + 1) << "/" << option.epoch_num
           << " sample " << finished_sample_num << "/" << sample_num
           << " loss=" << average_loss
           << " recent_loss=" << recent_loss
           << " samples/s=" << samples_per_sec
           << " epoch_eta=" << FormatSeconds(epoch_eta)
           << " total_eta=" << FormatSeconds(total_eta) << endl;
    };

    auto epoch_callback = [&](int, double average_loss, bool &early_stop) {
      last_loss = average_loss;
      if (StopRequested() || stop_during_epoch) {
        early_stop = true;
        stop_after_epoch = true;
        return;
      }
      if (option.early_stop_loss > 0 &&
          average_loss < option.early_stop_loss) {
        early_stop = true;
        stop_after_epoch = true;
        stopped_by_loss = true;
      }
    };

    MiniTransformerLM::RC train_rc = MiniTransformerLM::SUCCESS;
    if (option.batch_size > 1 && option.thread_num > 1) {
      train_rc = model.TrainNextTokenBatchParallel(
          input_samples, target_tokens, option.batch_size, option.thread_num,
          epoch_callback, 1, current_lr, nullptr, sample_callback);
    } else if (option.batch_size > 1) {
      train_rc = model.TrainNextTokenBatch(
          input_samples, target_tokens, option.batch_size, epoch_callback, 1,
          current_lr, nullptr, sample_callback);
    } else {
      train_rc = model.TrainNextToken(input_samples, target_tokens,
                                      epoch_callback, 1, current_lr, nullptr,
                                      sample_callback);
    }
    if (train_rc != MiniTransformerLM::SUCCESS) {
      cout << "Train failed: " << model.err_msg() << endl;
      return -1;
    }

    const bool finished_full_epoch = !stop_during_epoch;
    if (finished_full_epoch) {
      completed_epoch = global_epoch + 1;
    }
    const double perplexity = std::exp(last_loss);
    const auto now = std::chrono::steady_clock::now();
    const double elapsed = SecondsSince(train_start, now);
    const int finished_epochs = std::max(1, completed_epoch - start_epoch);
    const double epoch_per_sec = finished_epochs / std::max(elapsed, 1e-9);
    const double eta =
        epoch_per_sec > 0 ? (option.epoch_num - completed_epoch) / epoch_per_sec
                          : 0.0;
    if ((completed_epoch % option.log_every == 0) ||
        completed_epoch == option.epoch_num || stop_after_epoch ||
        stop_during_epoch) {
      cout << "epoch " << completed_epoch << "/" << option.epoch_num
           << " loss=" << last_loss << " perplexity=" << perplexity
           << " recent_loss=" << recent_loss
           << " lr=" << current_lr
           << " elapsed=" << FormatSeconds(elapsed)
           << " eta=" << FormatSeconds(eta) << endl;
    }

    if (!option.no_checkpoint &&
        (completed_epoch % option.checkpoint_every == 0 ||
         completed_epoch == option.epoch_num || stop_after_epoch ||
         stop_during_epoch)) {
      CheckpointMeta meta;
      meta.completed_epoch = completed_epoch;
      meta.target_epoch = option.epoch_num;
      meta.learning_rate = option.learning_rate;
      meta.last_loss = last_loss;
      meta.last_perplexity = perplexity;
      meta.source = source;
      if (!SaveTrainingCheckpoint(checkpoint_path, model, tokenizer, meta, err)) {
        cout << err << endl;
        return -1;
      }
      cout << "Saved checkpoint: " << checkpoint_path
           << " completed_epoch=" << completed_epoch << endl;
    }

    if (stop_during_epoch) {
      interrupted = true;
      break;
    }
    if (stop_after_epoch) {
      break;
    }
  }

  if (interrupted) {
    if (option.no_checkpoint) {
      cout << "Training interrupted; checkpoint disabled, weights were not "
              "saved"
           << endl;
    } else {
      cout << "Training interrupted; checkpoint saved at " << checkpoint_path
           << endl;
    }
    return 130;
  }

  double average_loss = last_loss;
  double perplexity = std::exp(average_loss);
  if (!option.skip_final_eval) {
    if (model.CalcNextTokenLoss(input_samples, target_tokens, average_loss) !=
        MiniTransformerLM::SUCCESS) {
      cout << "Evaluate failed: " << model.err_msg() << endl;
      return -1;
    }
    perplexity = std::exp(average_loss);
  }

  if (!SaveModelWithVocab(option.model, model, tokenizer, err)) {
    cout << err << endl;
    return -1;
  }
  if (option.skip_final_eval) {
    cout << "Skipped final evaluation; last training loss: " << average_loss
         << " perplexity: " << perplexity << endl;
  } else {
    cout << "Loss: " << average_loss << " Perplexity: " << perplexity << endl;
  }
  cout << "Saved updated weights: " << option.model << endl;
  if (stopped_by_loss) {
    cout << "Stopped early because loss dropped below "
         << option.early_stop_loss << endl;
  }
  return 0;
}

int RunGenerate(const Option &option) {
  MiniTransformerLM model;
  TokenizerBundle tokenizer;
  string err;
  if (!LoadModel(option.model, model, tokenizer, err)) {
    cout << err << endl;
    return -1;
  }
  if (option.generate_num < 0) {
    cout << "generate-num must be >= 0" << endl;
    return -1;
  }

  vector<int> prompt_token_ids;
  int unknown_or_dropped = 0;
  string prompt;
  if (!EncodePrompt(tokenizer, option.prompt, prompt_token_ids,
                    unknown_or_dropped, prompt, err)) {
    cout << err << endl;
    return -1;
  }
  if (unknown_or_dropped > 0) {
    if (tokenizer.kind == TOKENIZER_WORD) {
      if (tokenizer.word_tokenizer.unknown_policy() ==
          WordTokenizer::UNKNOWN_DROP) {
        cout << "Warning: dropped " << unknown_or_dropped
             << " prompt words outside the model vocabulary" << endl;
      } else {
        cout << "Warning: mapped " << unknown_or_dropped
             << " prompt words outside the model vocabulary to <unk>" << endl;
      }
    } else {
      cout << "Warning: dropped " << unknown_or_dropped
           << " prompt characters outside the model vocabulary" << endl;
    }
  }

  vector<int> generated_token_ids;
  if (option.sampling_specified) {
    MiniTransformerLM::SamplingOption sampling_option;
    sampling_option.temperature_ = option.temperature;
    sampling_option.top_k_ = option.top_k;
    sampling_option.top_p_ = option.top_p;
    if (model.GenerateSample(prompt_token_ids, option.generate_num,
                             generated_token_ids, sampling_option) !=
        MiniTransformerLM::SUCCESS) {
      cout << "GenerateSample failed: " << model.err_msg() << endl;
      return -1;
    }
  } else {
    if (model.Generate(prompt_token_ids, option.generate_num,
                       generated_token_ids) != MiniTransformerLM::SUCCESS) {
      cout << "Generate failed: " << model.err_msg() << endl;
      return -1;
    }
  }

  string generated_text;
  if (!DecodeGenerated(tokenizer, generated_token_ids, generated_text, err)) {
    cout << err << endl;
    return -1;
  }

  cout << "Prompt: " << prompt << endl;
  cout << "Mode: " << (option.sampling_specified ? "sampling" : "greedy")
       << endl;
  cout << "Generated: " << generated_text << endl;
  return 0;
}

int RunInfo(const Option &option) {
  MiniTransformerLM model;
  TokenizerBundle tokenizer;
  string err;
  if (!LoadModel(option.model, model, tokenizer, err)) {
    cout << err << endl;
    return -1;
  }

  cout << "Model: " << option.model << endl;
  cout << "Backbone: " << BackboneName(model.backbone_type())
       << " model_dim: " << model.model_dim()
       << " head_num: " << model.head_num()
       << " feed_forward_dim: " << model.feed_forward_dim()
       << " block_num: " << model.block_num()
       << " context_size: " << model.max_context_size()
       << " rand_seed: " << model.rand_seed() << endl;
  cout << "Tokenizer: " << TokenizerName(tokenizer.kind) << endl;
  if (tokenizer.kind == TOKENIZER_WORD) {
    cout << "Unknown policy: "
         << UnknownPolicyName(tokenizer.word_tokenizer.unknown_policy())
         << endl;
  }
  cout << "Vocab size: " << model.vocab_size() << endl;
  cout << "Vocabulary: " << DisplayTokenizerVocab(tokenizer) << endl;
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    PrintTopUsage(argv[0]);
    return 0;
  }
  string verb = argv[1];
  if (verb == "--help" || verb == "-h") {
    PrintTopUsage(argv[0]);
    return 0;
  }
  if (verb != "init" && verb != "train" && verb != "generate" &&
      verb != "info") {
    cout << "Unknown command: " << verb << endl;
    PrintTopUsage(argv[0]);
    return -1;
  }

  Option option;
  try {
    if (!ParseArgs(argc, argv, 2, option)) {
      PrintCommandUsage(argv[0], verb);
      return 0;
    }
  } catch (const std::exception &ex) {
    cout << ex.what() << endl;
    PrintCommandUsage(argv[0], verb);
    return -1;
  }
  if (option.tokenizer_specified && verb != "init") {
    cout << "--tokenizer is only used by init; existing models load tokenizer "
            "type from .vocab"
         << endl;
    return -1;
  }
  if (option.max_vocab_size_specified && verb != "init") {
    cout << "--max-vocab-size is only used by init; existing models load vocab "
            "from .vocab"
         << endl;
    return -1;
  }
  if (option.unknown_policy_specified && verb != "init") {
    cout << "--unknown-policy is only used by init; existing models load "
            "unknown policy from .vocab"
         << endl;
    return -1;
  }
  if (option.train_control_specified && verb != "train") {
    cout << "training log/checkpoint options are only used by train" << endl;
    return -1;
  }

  if (verb == "init") {
    return RunInit(option);
  }
  if (verb == "train") {
    return RunTrain(option);
  }
  if (verb == "generate") {
    return RunGenerate(option);
  }
  return RunInfo(option);
}

#include "lr_scheduler/warmup_cosine_lr.h"
#include "transformer/character_dataset.h"
#include "transformer/character_tokenizer.h"
#include "transformer/mini_transformer_lm.h"
#include "transformer/mini_transformer_lm_loader.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace deeplearning;

namespace fs = std::filesystem;

namespace {

// A base-model style CLI around MiniTransformerLM with four sub-commands:
//   init      define structure + build vocab from a corpus + save a fresh model
//   train     load a model and train/continue on a text file or directory
//   generate  load a model and generate text from a prompt
//   info      print a model's structure and vocabulary
//
// A model is stored as two sidecar files sharing the same base path:
//   <model>        binary weights + structure (MiniTransformerLMLoader format)
//   <model>.vocab  the character vocabulary, raw bytes (this demo's own file)
// The vocabulary is required to map characters to token ids, so it is persisted
// alongside the weights instead of being rebuilt from the training corpus.

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
  double learning_rate = 0.01;
  double temperature = 1.0;
  int top_k = 0;
  double top_p = 1.0;
  string backbone = "decoder";
  bool sampling_specified = false;
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
         << "  --learning-rate <double>\n";
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
    } else if (arg == "--learning-rate") {
      option.learning_rate = std::stod(need_value("--learning-rate"));
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
    } else if (arg == "--help") {
      return false;
    } else {
      throw std::runtime_error(string("Unknown option: ") + arg);
    }
  }
  return true;
}

string VocabPath(const string &model_path) { return model_path + ".vocab"; }

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

string DisplayVocab(const string &vocabulary) {
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
               CharacterTokenizer &tokenizer, string &err) {
  std::ifstream probe(model_path, std::ios::binary);
  if (!probe.good()) {
    err = "Model file not found: " + model_path;
    return false;
  }
  probe.close();

  string vocabulary;
  if (!ReadFileRaw(VocabPath(model_path), vocabulary) || vocabulary.empty()) {
    err = "Vocab sidecar not found or empty: " + VocabPath(model_path) +
          " (was this model created by 'mini_lm init'?)";
    return false;
  }
  if (tokenizer.Init(vocabulary) != CharacterTokenizer::SUCCESS) {
    err = "Tokenizer init failed: " + tokenizer.err_msg();
    return false;
  }
  if (MiniTransformerLMLoader::ImportModelFromFile(model, model_path) !=
      MiniTransformerLMLoader::SUCCESS) {
    err = "Import model failed: " + model_path;
    return false;
  }
  if (model.vocab_size() != tokenizer.vocab_size()) {
    err = "Model vocab size does not match the vocab sidecar";
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

  string vocabulary = CharacterTokenizer::BuildVocabularyFromText(corpus);
  CharacterTokenizer tokenizer;
  if (tokenizer.Init(vocabulary) != CharacterTokenizer::SUCCESS) {
    cout << "Tokenizer init failed: " << tokenizer.err_msg() << endl;
    return -1;
  }

  MiniTransformerLM::Config config;
  config.vocab_size_ = tokenizer.vocab_size();
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
  if (MiniTransformerLMLoader::ExportModelToFile(model, option.model) !=
      MiniTransformerLMLoader::SUCCESS) {
    cout << "Export model failed: " << option.model << endl;
    return -1;
  }
  {
    std::ofstream ofs(VocabPath(option.model), std::ios::binary | std::ios::trunc);
    if (!ofs.is_open() || !(ofs << vocabulary).good()) {
      cout << "Write vocab sidecar failed: " << VocabPath(option.model) << endl;
      return -1;
    }
  }

  cout << "Initialized model: " << option.model << endl;
  cout << "Vocab source: " << source << " vocab_size: " << tokenizer.vocab_size()
       << endl;
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
  MiniTransformerLM model;
  CharacterTokenizer tokenizer;
  string err;
  if (!LoadModel(option.model, model, tokenizer, err)) {
    cout << err << endl;
    return -1;
  }
  const int context_size = model.max_context_size();
  if (context_size <= 0) {
    cout << "Loaded model has no context size; re-create it with 'init'" << endl;
    return -1;
  }
  if (option.epoch_num <= 0 || option.learning_rate <= 0.0) {
    cout << "Invalid training option (epochs and learning-rate must be > 0)"
         << endl;
    return -1;
  }

  string corpus;
  string source;
  if (!CollectCorpus(option, corpus, source, err)) {
    cout << err << endl;
    return -1;
  }
  long long dropped = 0;
  string filtered = FilterToVocab(tokenizer.vocabulary(), corpus, dropped);
  if (dropped > 0) {
    cout << "Warning: skipped " << dropped
         << " characters outside the model vocabulary" << endl;
  }
  if (filtered.empty()) {
    cout << "No trainable text left after filtering to the model vocabulary"
         << endl;
    return -1;
  }

  vector<int> corpus_token_ids;
  if (tokenizer.Encode(filtered, corpus_token_ids) !=
      CharacterTokenizer::SUCCESS) {
    cout << "Encode corpus failed: " << tokenizer.err_msg() << endl;
    return -1;
  }

  CharacterDataset dataset;
  if (dataset.Init(corpus_token_ids, context_size) !=
      CharacterDataset::SUCCESS) {
    cout << "Dataset init failed: " << dataset.err_msg()
         << " (need more than " << context_size << " in-vocab characters)"
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
  int log_every = std::max(1, option.epoch_num / 10);
  auto callback = [&](int epoch, double average_loss, bool &early_stop) {
    if ((epoch + 1) % log_every == 0 || epoch + 1 == option.epoch_num) {
      cout << "epoch " << (epoch + 1) << "/" << option.epoch_num
           << " loss=" << average_loss << endl;
    }
    if (average_loss < 0.02) {
      early_stop = true;
    }
  };

  cout << "Training on " << source << " (" << input_samples.size()
       << " samples, context " << context_size << ")" << endl;
  if (model.TrainNextToken(input_samples, target_tokens, callback,
                           option.epoch_num, option.learning_rate,
                           &lr_scheduler) != MiniTransformerLM::SUCCESS) {
    cout << "TrainNextToken failed: " << model.err_msg() << endl;
    return -1;
  }

  double average_loss = 0.0;
  double perplexity = 0.0;
  if (model.CalcNextTokenLoss(input_samples, target_tokens, average_loss) !=
          MiniTransformerLM::SUCCESS ||
      model.CalcPerplexity(input_samples, target_tokens, perplexity) !=
          MiniTransformerLM::SUCCESS) {
    cout << "Evaluate failed: " << model.err_msg() << endl;
    return -1;
  }

  if (MiniTransformerLMLoader::ExportModelToFile(model, option.model) !=
      MiniTransformerLMLoader::SUCCESS) {
    cout << "Export model failed: " << option.model << endl;
    return -1;
  }
  cout << "Loss: " << average_loss << " Perplexity: " << perplexity << endl;
  cout << "Saved updated weights: " << option.model << endl;
  return 0;
}

int RunGenerate(const Option &option) {
  MiniTransformerLM model;
  CharacterTokenizer tokenizer;
  string err;
  if (!LoadModel(option.model, model, tokenizer, err)) {
    cout << err << endl;
    return -1;
  }
  if (option.generate_num < 0) {
    cout << "generate-num must be >= 0" << endl;
    return -1;
  }

  long long dropped = 0;
  string prompt = FilterToVocab(tokenizer.vocabulary(), option.prompt, dropped);
  if (dropped > 0) {
    cout << "Warning: dropped " << dropped
         << " prompt characters outside the model vocabulary" << endl;
  }
  if (prompt.empty()) {
    cout << "Prompt is empty after filtering to the model vocabulary" << endl;
    return -1;
  }

  vector<int> prompt_token_ids;
  if (tokenizer.Encode(prompt, prompt_token_ids) !=
      CharacterTokenizer::SUCCESS) {
    cout << "Encode prompt failed: " << tokenizer.err_msg() << endl;
    return -1;
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
  if (tokenizer.Decode(generated_token_ids, generated_text) !=
      CharacterTokenizer::SUCCESS) {
    cout << "Decode generated text failed: " << tokenizer.err_msg() << endl;
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
  CharacterTokenizer tokenizer;
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
  cout << "Vocab size: " << model.vocab_size() << endl;
  cout << "Vocabulary: " << DisplayVocab(tokenizer.vocabulary()) << endl;
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

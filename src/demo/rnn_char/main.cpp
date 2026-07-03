#include "lr_scheduler/warmup_cosine_lr.h"
#include "rnn/mini_rnn_lm.h"
#include "transformer/character_dataset.h"
#include "transformer/character_tokenizer.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace deeplearning;

namespace {

struct DemoOption {
  string corpus = "abcabcabcabcabcabc";
  string corpus_file;
  string prompt = "abc";
  int generate_num = 9;
  int epoch_num = 300;
  int rand_seed = 0;
  int hidden_dim = 16;
  int context_size = 3;
  double learning_rate = 0.03;
  double gradient_clip_norm = 1.0;
};

void PrintUsage(const char *prog) {
  cout << "Usage: " << prog << " [options]\n"
       << "  --prompt <text>\n"
       << "  --generate-num <int>\n"
       << "  --epochs <int>\n"
       << "  --learning-rate <double>\n"
       << "  --rand-seed <int>\n"
       << "  --hidden-dim <int>\n"
       << "  --context-size <int>\n"
       << "  --gradient-clip-norm <double>\n"
       << "  --corpus <text>\n"
       << "  --corpus-file <path>\n"
       << "  --help\n";
}

bool ParseArgs(int argc, char **argv, DemoOption &option) {
  for (int i = 1; i < argc; i++) {
    string arg = argv[i];
    auto need_value = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        throw std::runtime_error(string("Missing value for ") + name);
      }
      i += 1;
      return argv[i];
    };

    if (arg == "--prompt") {
      option.prompt = need_value("--prompt");
    } else if (arg == "--generate-num") {
      option.generate_num = std::stoi(need_value("--generate-num"));
    } else if (arg == "--epochs") {
      option.epoch_num = std::stoi(need_value("--epochs"));
    } else if (arg == "--learning-rate") {
      option.learning_rate = std::stod(need_value("--learning-rate"));
    } else if (arg == "--rand-seed") {
      option.rand_seed = std::stoi(need_value("--rand-seed"));
    } else if (arg == "--hidden-dim") {
      option.hidden_dim = std::stoi(need_value("--hidden-dim"));
    } else if (arg == "--context-size") {
      option.context_size = std::stoi(need_value("--context-size"));
    } else if (arg == "--gradient-clip-norm") {
      option.gradient_clip_norm = std::stod(need_value("--gradient-clip-norm"));
    } else if (arg == "--corpus") {
      option.corpus = need_value("--corpus");
    } else if (arg == "--corpus-file") {
      option.corpus_file = need_value("--corpus-file");
    } else if (arg == "--help") {
      return false;
    } else {
      throw std::runtime_error(string("Unknown option: ") + arg);
    }
  }
  return true;
}

bool ReadTextFile(const string &filename, string &content) {
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    return false;
  }
  content.assign((std::istreambuf_iterator<char>(ifs)),
                 std::istreambuf_iterator<char>());
  while (!content.empty() &&
         (content.back() == '\n' || content.back() == '\r')) {
    content.pop_back();
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  DemoOption option;
  try {
    if (!ParseArgs(argc, argv, option)) {
      PrintUsage(argv[0]);
      return 0;
    }
  } catch (const std::exception &ex) {
    cout << ex.what() << endl;
    PrintUsage(argv[0]);
    return -1;
  }

  if (!option.corpus_file.empty() && !ReadTextFile(option.corpus_file, option.corpus)) {
    cout << "Read corpus file failed: " << option.corpus_file << endl;
    return -1;
  }
  if (option.epoch_num <= 0 || option.learning_rate <= 0.0 ||
      option.hidden_dim <= 0 || option.context_size <= 0 ||
      option.generate_num < 0) {
    cout << "Invalid option" << endl;
    return -1;
  }

  string vocabulary = CharacterTokenizer::BuildVocabularyFromText(option.corpus);
  CharacterTokenizer tokenizer;
  auto tokenizer_rc = tokenizer.Init(vocabulary);
  if (tokenizer_rc != CharacterTokenizer::SUCCESS) {
    cout << "Tokenizer init failed: " << tokenizer.err_msg() << endl;
    return -1;
  }

  vector<int> corpus_token_ids;
  tokenizer_rc = tokenizer.Encode(option.corpus, corpus_token_ids);
  if (tokenizer_rc != CharacterTokenizer::SUCCESS) {
    cout << "Encode corpus failed: " << tokenizer.err_msg() << endl;
    return -1;
  }

  CharacterDataset dataset;
  auto dataset_rc = dataset.Init(corpus_token_ids, option.context_size);
  if (dataset_rc != CharacterDataset::SUCCESS) {
    cout << "Dataset init failed: " << dataset.err_msg() << endl;
    return -1;
  }
  vector<vector<int>> input_samples;
  vector<int> target_tokens;
  dataset_rc = dataset.BuildNextTokenSamples(input_samples, target_tokens);
  if (dataset_rc != CharacterDataset::SUCCESS) {
    cout << "BuildNextTokenSamples failed: " << dataset.err_msg() << endl;
    return -1;
  }

  vector<int> prompt_token_ids;
  tokenizer_rc = tokenizer.Encode(option.prompt, prompt_token_ids);
  if (tokenizer_rc != CharacterTokenizer::SUCCESS) {
    cout << "Encode prompt failed: " << tokenizer.err_msg() << endl;
    return -1;
  }

  MiniRNNLM model;
  MiniRNNLM::Config config;
  config.vocab_size_ = tokenizer.vocab_size();
  config.hidden_dim_ = option.hidden_dim;
  config.rand_seed_ = option.rand_seed;
  config.gradient_clip_norm_ = option.gradient_clip_norm;
  auto init_rc = model.Init(config);
  if (init_rc != MiniRNNLM::SUCCESS) {
    cout << "Init failed: " << model.err_msg() << endl;
    return -1;
  }

  int warmup_epochs = std::max(1, option.epoch_num / 10);
  WarmupCosineLR lr_scheduler(option.learning_rate, warmup_epochs,
                              option.epoch_num, option.learning_rate * 0.1);
  int log_every = std::max(1, option.epoch_num / 10);
  auto callback = [&](int epoch, double average_loss, bool &) {
    if ((epoch + 1) % log_every != 0 && epoch + 1 != option.epoch_num) {
      return;
    }
    cout << "epoch " << (epoch + 1) << "/" << option.epoch_num
         << " loss=" << average_loss << endl;
  };

  auto train_rc = model.TrainNextToken(input_samples, target_tokens, callback,
                                       option.epoch_num, option.learning_rate,
                                       &lr_scheduler);
  if (train_rc != MiniRNNLM::SUCCESS) {
    cout << "TrainNextToken failed: " << model.err_msg() << endl;
    return -1;
  }

  double average_loss = 0.0;
  auto loss_rc = model.CalcNextTokenLoss(input_samples, target_tokens, average_loss);
  if (loss_rc != MiniRNNLM::SUCCESS) {
    cout << "CalcNextTokenLoss failed: " << model.err_msg() << endl;
    return -1;
  }
  double perplexity = 0.0;
  auto ppl_rc = model.CalcPerplexity(input_samples, target_tokens, perplexity);
  if (ppl_rc != MiniRNNLM::SUCCESS) {
    cout << "CalcPerplexity failed: " << model.err_msg() << endl;
    return -1;
  }

  vector<int> generated_token_ids;
  auto gen_rc = model.Generate(prompt_token_ids, option.generate_num,
                               generated_token_ids);
  if (gen_rc != MiniRNNLM::SUCCESS) {
    cout << "Generate failed: " << model.err_msg() << endl;
    return -1;
  }
  string generated_text;
  tokenizer_rc = tokenizer.Decode(generated_token_ids, generated_text);
  if (tokenizer_rc != CharacterTokenizer::SUCCESS) {
    cout << "Decode generated text failed: " << tokenizer.err_msg() << endl;
    return -1;
  }

  cout << "Corpus length: " << option.corpus.size()
       << " vocab_size: " << tokenizer.vocab_size()
       << " sample_size: " << input_samples.size() << endl;
  cout << "Prompt: " << option.prompt << endl;
  cout << "Hidden dim: " << option.hidden_dim
       << " context_size: " << option.context_size
       << " gradient_clip_norm: " << option.gradient_clip_norm << endl;
  cout << "Loss: " << average_loss << " Perplexity: " << perplexity << endl;
  cout << "Generated: " << generated_text << endl;
  cout << "This demo trains a minimal tanh RNN with BPTT on a character corpus."
       << endl;
  return 0;
}

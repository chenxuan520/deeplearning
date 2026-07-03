#include "embedding/word2vec.h"
#include "embedding/word_tokenizer.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace deeplearning;

namespace {

const char *kDefaultCorpus = R"(the cat sat on the mat
the dog sat on the log
cats and dogs are pets
kittens play with yarn
puppies play in the park
the cat chased the mouse
the dog chased the ball
birds fly in the sky
fish swim in the water
apples grow on trees
cars drive on roads
the king ruled the land
the queen ruled the land
)";

struct DemoOption {
  string corpus_file;
  string query_word = "cat";
  string compare_word = "dog";
  string mode = "skip-gram";
  int embed_dim = 16;
  int window_size = 2;
  int negative_num = 5;
  int epochs = 30;
  double learning_rate = 0.08;
  int rand_seed = 7;
  int top_k = 5;
  bool show_pairs = false;
};

void PrintUsage(const char *prog) {
  cout << "Usage: " << prog << " [options]\n"
       << "  --corpus-file <path>       text file, one sentence per line\n"
       << "  --query-word <word>        word for nearest-neighbor lookup\n"
       << "  --compare-word <word>      second word for cosine similarity\n"
       << "  --mode skip-gram|cbow      training objective (default skip-gram)\n"
       << "  --embed-dim <int>\n"
       << "  --window-size <int>\n"
       << "  --negative-num <int>\n"
       << "  --epochs <int>\n"
       << "  --learning-rate <double>\n"
       << "  --rand-seed <int>\n"
       << "  --top-k <int>\n"
       << "  --show-pairs               print a few skip-gram training pairs\n"
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

    if (arg == "--corpus-file") {
      option.corpus_file = need_value("--corpus-file");
    } else if (arg == "--query-word") {
      option.query_word = need_value("--query-word");
    } else if (arg == "--compare-word") {
      option.compare_word = need_value("--compare-word");
    } else if (arg == "--mode") {
      option.mode = need_value("--mode");
    } else if (arg == "--embed-dim") {
      option.embed_dim = std::stoi(need_value("--embed-dim"));
    } else if (arg == "--window-size") {
      option.window_size = std::stoi(need_value("--window-size"));
    } else if (arg == "--negative-num") {
      option.negative_num = std::stoi(need_value("--negative-num"));
    } else if (arg == "--epochs") {
      option.epochs = std::stoi(need_value("--epochs"));
    } else if (arg == "--learning-rate") {
      option.learning_rate = std::stod(need_value("--learning-rate"));
    } else if (arg == "--rand-seed") {
      option.rand_seed = std::stoi(need_value("--rand-seed"));
    } else if (arg == "--top-k") {
      option.top_k = std::stoi(need_value("--top-k"));
    } else if (arg == "--show-pairs") {
      option.show_pairs = true;
    } else if (arg == "--help") {
      return false;
    } else {
      throw std::runtime_error(string("Unknown option: ") + arg);
    }
  }
  return true;
}

string ReadFile(const string &path) {
  ifstream ifs(path);
  if (!ifs) {
    throw std::runtime_error("Failed to open corpus file: " + path);
  }
  stringstream buffer;
  buffer << ifs.rdbuf();
  return buffer.str();
}

void PrintPairs(const WordTokenizer &tokenizer, const vector<int> &tokens,
                int window_size) {
  cout << "Sample skip-gram pairs (center -> context):\n";
  int printed = 0;
  const int seq_len = static_cast<int>(tokens.size());
  for (int center_pos = 0; center_pos < seq_len && printed < 8; center_pos++) {
    const int left = std::max(0, center_pos - window_size);
    const int right = std::min(seq_len - 1, center_pos + window_size);
    for (int context_pos = left; context_pos <= right; context_pos++) {
      if (context_pos == center_pos) {
        continue;
      }
      cout << "  " << tokenizer.Word(tokens[center_pos]) << " -> "
           << tokenizer.Word(tokens[context_pos]) << "\n";
      printed++;
      if (printed >= 8) {
        break;
      }
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    DemoOption option;
    if (!ParseArgs(argc, argv, option)) {
      PrintUsage(argv[0]);
      return 0;
    }

    const string corpus =
        option.corpus_file.empty() ? kDefaultCorpus : ReadFile(option.corpus_file);

    WordTokenizer tokenizer;
    if (tokenizer.InitFromText(corpus) != WordTokenizer::SUCCESS) {
      cerr << tokenizer.err_msg() << "\n";
      return 1;
    }

    vector<vector<int>> sentences;
    if (tokenizer.TokenizeSentences(corpus, sentences) != WordTokenizer::SUCCESS) {
      cerr << tokenizer.err_msg() << "\n";
      return 1;
    }

    Word2Vec model;
    Word2Vec::Config config;
    config.embed_dim = option.embed_dim;
    config.window_size = option.window_size;
    config.negative_num = option.negative_num;
    config.epochs = option.epochs;
    config.learning_rate = option.learning_rate;
    config.rand_seed = option.rand_seed;
    config.mode =
        option.mode == "cbow" ? Word2Vec::Mode::CBOW : Word2Vec::Mode::SKIP_GRAM;

    if (model.Init(tokenizer.vocab_size(), config) != Word2Vec::SUCCESS) {
      cerr << model.err_msg() << "\n";
      return 1;
    }

    cout << "vocab_size=" << tokenizer.vocab_size()
         << " embed_dim=" << config.embed_dim
         << " mode=" << option.mode << "\n";

    if (option.show_pairs && !sentences.empty()) {
      PrintPairs(tokenizer, sentences.front(), option.window_size);
    }

    Word2Vec::TrainStats stats;
    if (model.Train(sentences, &stats) != Word2Vec::SUCCESS) {
      cerr << model.err_msg() << "\n";
      return 1;
    }

    cout << "trained pairs=" << stats.pair_count
         << " average_loss=" << stats.average_loss << "\n";

    const int query_id = tokenizer.Lookup(option.query_word);
    const int compare_id = tokenizer.Lookup(option.compare_word);
    if (query_id < 0) {
      cerr << "Unknown query word: " << option.query_word << "\n";
      return 1;
    }

    const auto neighbors = model.MostSimilar(query_id, option.top_k);
    cout << "Most similar to \"" << option.query_word << "\":\n";
    for (const auto &item : neighbors) {
      cout << "  " << tokenizer.Word(item.first) << "  cos="
           << item.second << "\n";
    }

    if (compare_id >= 0) {
      cout << "cos(\"" << option.query_word << "\", \"" << option.compare_word
           << "\") = " << model.CosineSimilarity(query_id, compare_id) << "\n";
    }

    const int apple_id = tokenizer.Lookup("apples");
    if (apple_id >= 0) {
      cout << "cos(\"" << option.query_word << "\", \"apples\") = "
           << model.CosineSimilarity(query_id, apple_id) << "\n";
    }

    return 0;
  } catch (const std::exception &ex) {
    cerr << ex.what() << "\n";
    return 1;
  }
}

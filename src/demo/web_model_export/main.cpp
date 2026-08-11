#include "neural_network_loader.h"
#include "transformer/mini_transformer_lm_loader.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace deeplearning;

namespace {

struct Option {
  string type;
  string model;
  string vocab;
  string out;
};

void PrintUsage(const char *prog) {
  cout << "Usage: " << prog << " --type <mlp|mini-lm> --model <path> "
       << "--out <path> [--vocab <path>]\n"
       << "  mlp     export NeuralNetworkLoader .param to JSON\n"
       << "  mini-lm export MiniTransformerLM .param + .vocab to JSON\n";
}

Option ParseArgs(int argc, char **argv) {
  Option option;
  for (int i = 1; i < argc; i++) {
    string arg = argv[i];
    auto need_value = [&](const char *name) -> string {
      if (i + 1 >= argc) {
        throw std::runtime_error(string("Missing value for ") + name);
      }
      i += 1;
      return argv[i];
    };

    if (arg == "--type") {
      option.type = need_value("--type");
    } else if (arg == "--model") {
      option.model = need_value("--model");
    } else if (arg == "--vocab") {
      option.vocab = need_value("--vocab");
    } else if (arg == "--out") {
      option.out = need_value("--out");
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown option: " + arg);
    }
  }
  if (option.type.empty() || option.model.empty() || option.out.empty()) {
    throw std::runtime_error("Missing required option");
  }
  if (option.type != "mlp" && option.type != "mini-lm") {
    throw std::runtime_error("Invalid type, expected mlp or mini-lm");
  }
  return option;
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

string JsonEscape(const string &text) {
  std::ostringstream oss;
  for (unsigned char ch : text) {
    switch (ch) {
    case '"':
      oss << "\\\"";
      break;
    case '\\':
      oss << "\\\\";
      break;
    case '\b':
      oss << "\\b";
      break;
    case '\f':
      oss << "\\f";
      break;
    case '\n':
      oss << "\\n";
      break;
    case '\r':
      oss << "\\r";
      break;
    case '\t':
      oss << "\\t";
      break;
    default:
      if (ch < 0x20) {
        oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
            << static_cast<int>(ch) << std::dec;
      } else {
        oss << static_cast<char>(ch);
      }
    }
  }
  return oss.str();
}

const char *LossName(LossType type) {
  switch (type) {
  case LOSS_MSE:
    return "mse";
  case LOSS_CROSS_ENTROPY:
    return "cross_entropy";
  }
  return "unknown";
}

const char *ActivateName(ActivateType type) {
  switch (type) {
  case ACTIVATE_SIGMOID:
    return "sigmoid";
  case ACTIVATE_RELU:
    return "relu";
  case ACTIVATE_TANH:
    return "tanh";
  case ACTIVATE_LEAKY_RELU:
    return "leaky_relu";
  case ACTIVATE_GELU:
    return "gelu";
  }
  return "unknown";
}

const char *SoftmaxName(SoftmaxType type) {
  switch (type) {
  case SOFTMAX_NONE:
    return "none";
  case SOFTMAX_STD:
    return "std";
  }
  return "unknown";
}

const char *OptimizerName(OptimizerType type) {
  switch (type) {
  case OPTIMIZER_SGD:
    return "sgd";
  case OPTIMIZER_MOMENTUM:
    return "momentum";
  case OPTIMIZER_ADAM:
    return "adam";
  case OPTIMIZER_RMSPROP:
    return "rmsprop";
  case OPTIMIZER_ADAMW:
    return "adamw";
  }
  return "unknown";
}

const char *BackboneName(MiniTransformerLM::BackboneType type) {
  return type == MiniTransformerLM::BACKBONE_DECODER ? "decoder" : "encoder";
}

void WriteIndent(ostream &os, int indent) {
  for (int i = 0; i < indent; i++) {
    os << ' ';
  }
}

void WriteIntArray(ostream &os, const vector<int> &values) {
  os << "[";
  for (int i = 0; i < static_cast<int>(values.size()); i++) {
    if (i != 0) {
      os << ",";
    }
    os << values[i];
  }
  os << "]";
}

void WriteDoubleArray(ostream &os, const vector<double> &values) {
  os << "[";
  for (int i = 0; i < static_cast<int>(values.size()); i++) {
    if (i != 0) {
      os << ",";
    }
    os << std::setprecision(17) << values[i];
  }
  os << "]";
}

void WriteStringArray(ostream &os, const vector<string> &values) {
  os << "[";
  for (int i = 0; i < static_cast<int>(values.size()); i++) {
    if (i != 0) {
      os << ",";
    }
    os << "\"" << JsonEscape(values[i]) << "\"";
  }
  os << "]";
}

void WriteMatrix(ostream &os, const vector<vector<double>> &matrix,
                 int indent) {
  os << "[";
  if (!matrix.empty()) {
    os << "\n";
  }
  for (int i = 0; i < static_cast<int>(matrix.size()); i++) {
    if (i != 0) {
      os << ",\n";
    }
    WriteIndent(os, indent + 2);
    WriteDoubleArray(os, matrix[i]);
  }
  if (!matrix.empty()) {
    os << "\n";
    WriteIndent(os, indent);
  }
  os << "]";
}

void WriteTensor3(ostream &os, const vector<vector<vector<double>>> &tensor,
                  int indent) {
  os << "[";
  if (!tensor.empty()) {
    os << "\n";
  }
  for (int i = 0; i < static_cast<int>(tensor.size()); i++) {
    if (i != 0) {
      os << ",\n";
    }
    WriteIndent(os, indent + 2);
    WriteMatrix(os, tensor[i], indent + 2);
  }
  if (!tensor.empty()) {
    os << "\n";
    WriteIndent(os, indent);
  }
  os << "]";
}

vector<string> SplitLines(const string &content) {
  vector<string> lines;
  std::istringstream iss(content);
  string line;
  while (std::getline(iss, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
      line.pop_back();
    }
    lines.push_back(line);
  }
  return lines;
}

bool HexDecode(const string &hex_text, string &text) {
  if (hex_text.empty() || hex_text.size() % 2 != 0) {
    return false;
  }
  auto hex_value = [](char ch) -> int {
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
  };
  text.clear();
  text.reserve(hex_text.size() / 2);
  for (int i = 0; i < static_cast<int>(hex_text.size()); i += 2) {
    const int hi = hex_value(hex_text[i]);
    const int lo = hex_value(hex_text[i + 1]);
    if (hi < 0 || lo < 0) {
      return false;
    }
    text.push_back(static_cast<char>((hi << 4) | lo));
  }
  return true;
}

bool ParseMiniLmVocab(const string &content, string &tokenizer,
                      vector<string> &vocabulary, string &err) {
  vocabulary.clear();
  if (content.empty()) {
    err = "empty vocab file";
    return false;
  }
  if (content.rfind("tokenizer=", 0) != 0) {
    tokenizer = "char";
    for (char ch : content) {
      vocabulary.push_back(string(1, ch));
    }
    return true;
  }

  const auto lines = SplitLines(content);
  if (lines.empty()) {
    err = "invalid vocab file";
    return false;
  }
  tokenizer = lines[0].substr(string("tokenizer=").size());
  if (tokenizer == "word") {
    for (int i = 1; i < static_cast<int>(lines.size()); i++) {
      if (lines[i].empty() || lines[i].rfind("unknown-policy=", 0) == 0) {
        continue;
      }
      vocabulary.push_back(lines[i]);
    }
    return true;
  }
  if (tokenizer == "utf8-char") {
    for (int i = 1; i < static_cast<int>(lines.size()); i++) {
      if (lines[i].empty()) {
        continue;
      }
      string token;
      if (!HexDecode(lines[i], token)) {
        err = "invalid utf8-char vocab line";
        return false;
      }
      vocabulary.push_back(token);
    }
    return true;
  }
  if (tokenizer == "char") {
    for (int i = 1; i < static_cast<int>(lines.size()); i++) {
      for (char ch : lines[i]) {
        vocabulary.push_back(string(1, ch));
      }
      if (i + 1 < static_cast<int>(lines.size())) {
        vocabulary.push_back("\n");
      }
    }
    return true;
  }
  err = "unknown tokenizer: " + tokenizer;
  return false;
}

bool ExportMlp(const Option &option, string &err) {
  NeuralNetwork::NetworkParam param;
  NeuralNetwork::NetworkOption network_option;
  if (NeuralNetworkLoader::ImportParamFromFile(param, network_option,
                                               option.model) !=
      NeuralNetworkLoader::SUCCESS) {
    err = "import mlp model failed: " + option.model;
    return false;
  }

  std::ofstream ofs(option.out, std::ios::binary | std::ios::trunc);
  if (!ofs.is_open()) {
    err = "open output failed: " + option.out;
    return false;
  }

  ofs << "{\n";
  ofs << "  \"format\": \"deeplearning.web_model\",\n";
  ofs << "  \"version\": 1,\n";
  ofs << "  \"type\": \"mlp\",\n";
  ofs << "  \"config\": {\n";
  ofs << "    \"layers\": ";
  WriteIntArray(ofs, param.layer_);
  ofs << ",\n";
  ofs << "    \"learningRate\": " << std::setprecision(17)
      << network_option.learning_rate_ << ",\n";
  ofs << "    \"randSeed\": " << network_option.rand_seed_ << ",\n";
  ofs << "    \"activation\": \"" << ActivateName(network_option.activate_type_)
      << "\",\n";
  ofs << "    \"softmax\": \"" << SoftmaxName(network_option.softmax_type_)
      << "\",\n";
  ofs << "    \"loss\": \"" << LossName(network_option.loss_type_) << "\",\n";
  ofs << "    \"optimizer\": \"" << OptimizerName(network_option.optimizer_type_)
      << "\"\n";
  ofs << "  },\n";
  ofs << "  \"biases\": ";
  WriteMatrix(ofs, param.neuron_bias_, 2);
  ofs << ",\n";
  ofs << "  \"weights\": ";
  WriteTensor3(ofs, param.neuron_weight_, 2);
  ofs << "\n";
  ofs << "}\n";
  return ofs.good();
}

void WriteBlockJson(ostream &ofs, const TransformerBlock &block, int indent) {
  WriteIndent(ofs, indent);
  ofs << "{\n";
  WriteIndent(ofs, indent + 2);
  ofs << "\"depthResidualScale\": " << std::setprecision(17)
      << block.depth_residual_scale() << ",\n";
  WriteIndent(ofs, indent + 2);
  ofs << "\"attention\": {\n";
  WriteIndent(ofs, indent + 4);
  ofs << "\"queryWeight\": ";
  WriteMatrix(ofs, block.self_attention().query_weight(), indent + 4);
  ofs << ",\n";
  WriteIndent(ofs, indent + 4);
  ofs << "\"keyWeight\": ";
  WriteMatrix(ofs, block.self_attention().key_weight(), indent + 4);
  ofs << ",\n";
  WriteIndent(ofs, indent + 4);
  ofs << "\"valueWeight\": ";
  WriteMatrix(ofs, block.self_attention().value_weight(), indent + 4);
  ofs << ",\n";
  WriteIndent(ofs, indent + 4);
  ofs << "\"outputWeight\": ";
  WriteMatrix(ofs, block.self_attention().output_weight(), indent + 4);
  ofs << "\n";
  WriteIndent(ofs, indent + 2);
  ofs << "},\n";

  WriteIndent(ofs, indent + 2);
  ofs << "\"attentionNorm\": {\"scale\": ";
  WriteDoubleArray(ofs, block.attention_norm().scale());
  ofs << ", \"bias\": ";
  WriteDoubleArray(ofs, block.attention_norm().bias());
  ofs << "},\n";

  WriteIndent(ofs, indent + 2);
  ofs << "\"feedForward\": {\n";
  WriteIndent(ofs, indent + 4);
  ofs << "\"weight1\": ";
  WriteMatrix(ofs, block.feed_forward_weight_1(), indent + 4);
  ofs << ",\n";
  WriteIndent(ofs, indent + 4);
  ofs << "\"bias1\": ";
  WriteDoubleArray(ofs, block.feed_forward_bias_1());
  ofs << ",\n";
  WriteIndent(ofs, indent + 4);
  ofs << "\"weight2\": ";
  WriteMatrix(ofs, block.feed_forward_weight_2(), indent + 4);
  ofs << ",\n";
  WriteIndent(ofs, indent + 4);
  ofs << "\"bias2\": ";
  WriteDoubleArray(ofs, block.feed_forward_bias_2());
  ofs << "\n";
  WriteIndent(ofs, indent + 2);
  ofs << "},\n";

  WriteIndent(ofs, indent + 2);
  ofs << "\"feedForwardNorm\": {\"scale\": ";
  WriteDoubleArray(ofs, block.feed_forward_norm().scale());
  ofs << ", \"bias\": ";
  WriteDoubleArray(ofs, block.feed_forward_norm().bias());
  ofs << "}\n";
  WriteIndent(ofs, indent);
  ofs << "}";
}

bool ExportMiniLm(const Option &option, string &err) {
  MiniTransformerLM model;
  if (MiniTransformerLMLoader::ImportModelFromFile(model, option.model) !=
      MiniTransformerLMLoader::SUCCESS) {
    err = "import mini-lm model failed: " + option.model;
    return false;
  }

  string vocab_path = option.vocab.empty() ? option.model + ".vocab" : option.vocab;
  string vocab_content;
  if (!ReadFileRaw(vocab_path, vocab_content)) {
    err = "read vocab failed: " + vocab_path;
    return false;
  }
  string tokenizer;
  vector<string> vocabulary;
  if (!ParseMiniLmVocab(vocab_content, tokenizer, vocabulary, err)) {
    return false;
  }
  if (static_cast<int>(vocabulary.size()) != model.vocab_size()) {
    err = "vocab size does not match model";
    return false;
  }

  std::ofstream ofs(option.out, std::ios::binary | std::ios::trunc);
  if (!ofs.is_open()) {
    err = "open output failed: " + option.out;
    return false;
  }

  ofs << "{\n";
  ofs << "  \"format\": \"deeplearning.web_model\",\n";
  ofs << "  \"version\": 1,\n";
  ofs << "  \"type\": \"mini-lm\",\n";
  ofs << "  \"config\": {\n";
  ofs << "    \"backbone\": \"" << BackboneName(model.backbone_type()) << "\",\n";
  ofs << "    \"vocabSize\": " << model.vocab_size() << ",\n";
  ofs << "    \"modelDim\": " << model.model_dim() << ",\n";
  ofs << "    \"headNum\": " << model.head_num() << ",\n";
  ofs << "    \"feedForwardDim\": " << model.feed_forward_dim() << ",\n";
  ofs << "    \"blockNum\": " << model.block_num() << ",\n";
  ofs << "    \"maxContextSize\": " << model.max_context_size() << ",\n";
  ofs << "    \"usePositionalEncoding\": "
      << (model.use_positional_encoding() ? "true" : "false") << ",\n";
  ofs << "    \"scaleEmbedding\": "
      << (model.scale_embedding() ? "true" : "false") << ",\n";
  ofs << "    \"blockLearningRateScale\": " << std::setprecision(17)
      << model.block_learning_rate_scale() << "\n";
  ofs << "  },\n";
  ofs << "  \"tokenizer\": \"" << JsonEscape(tokenizer) << "\",\n";
  ofs << "  \"vocabulary\": ";
  WriteStringArray(ofs, vocabulary);
  ofs << ",\n";
  ofs << "  \"embedding\": ";
  WriteMatrix(ofs, model.token_embedding().embedding_table(), 2);
  ofs << ",\n";
  ofs << "  \"outputWeight\": ";
  WriteMatrix(ofs, model.output_weight(), 2);
  ofs << ",\n";
  ofs << "  \"outputBias\": ";
  WriteDoubleArray(ofs, model.output_bias());
  ofs << ",\n";
  ofs << "  \"blocks\": [";
  const auto &blocks = model.backbone_type() == MiniTransformerLM::BACKBONE_DECODER
                           ? model.decoder().blocks()
                           : model.encoder().blocks();
  if (!blocks.empty()) {
    ofs << "\n";
  }
  for (int i = 0; i < static_cast<int>(blocks.size()); i++) {
    if (i != 0) {
      ofs << ",\n";
    }
    WriteBlockJson(ofs, blocks[i], 4);
  }
  if (!blocks.empty()) {
    ofs << "\n  ";
  }
  ofs << "]\n";
  ofs << "}\n";
  return ofs.good();
}

} // namespace

int main(int argc, char **argv) {
  Option option;
  try {
    option = ParseArgs(argc, argv);
  } catch (const std::exception &ex) {
    cout << ex.what() << endl;
    PrintUsage(argv[0]);
    return -1;
  }

  string err;
  bool ok = false;
  if (option.type == "mlp") {
    ok = ExportMlp(option, err);
  } else {
    ok = ExportMiniLm(option, err);
  }
  if (!ok) {
    cout << err << endl;
    return -1;
  }
  cout << "Exported web model: " << option.out << endl;
  return 0;
}

#include "lr_scheduler/warmup_cosine_lr.h"
#include "transformer/character_dataset.h"
#include "transformer/character_tokenizer.h"
#include "transformer/mini_transformer_lm.h"
#include "transformer/mini_transformer_lm_loader.h"

#include <cstdio>
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
  string prompt = "ab";
  string model_file = "transformer_char_demo.param";
  string config_file;
  string attention_export_file;
  int generate_num = 6;
  int epoch_num = 1500;
  int rand_seed = 0;
  int model_dim = 6;
  int head_num = 1;
  int feed_forward_dim = 12;
  int block_num = 2;
  int context_size = 2;
  double learning_rate = 0.01;
  double block_learning_rate_scale = 1.0;
  double temperature = 0.8;
  int top_k = 2;
  double top_p = 0.9;
  string backbone = "decoder";
  bool force_train = false;
  bool save_model = true;
  bool eval_only = false;
};

void PrintUsage(const char *prog) {
  cout << "Usage: " << prog << " [options]\n"
       << "  --prompt <text>\n"
       << "  --generate-num <int>\n"
       << "  --temperature <double>\n"
       << "  --top-k <int>\n"
       << "  --top-p <double>\n"
       << "  --epochs <int>\n"
       << "  --learning-rate <double>\n"
       << "  --rand-seed <int>\n"
       << "  --backbone <encoder|decoder>\n"
       << "  --model-dim <int>\n"
       << "  --head-num <int>\n"
       << "  --feed-forward-dim <int>\n"
       << "  --block-num <int>\n"
       << "  --context-size <int>\n"
       << "  --block-learning-rate-scale <double>\n"
       << "  --model-file <path>\n"
       << "  --config-file <path>\n"
       << "  --attention-export-file <path>\n"
       << "  --corpus <text>\n"
       << "  --corpus-file <path>\n"
       << "  --save-model\n"
       << "  --no-save-model\n"
       << "  --eval-only\n"
       << "  --force-train\n"
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
    } else if (arg == "--temperature") {
      option.temperature = std::stod(need_value("--temperature"));
    } else if (arg == "--top-k") {
      option.top_k = std::stoi(need_value("--top-k"));
    } else if (arg == "--top-p") {
      option.top_p = std::stod(need_value("--top-p"));
    } else if (arg == "--epochs") {
      option.epoch_num = std::stoi(need_value("--epochs"));
    } else if (arg == "--learning-rate") {
      option.learning_rate = std::stod(need_value("--learning-rate"));
    } else if (arg == "--rand-seed") {
      option.rand_seed = std::stoi(need_value("--rand-seed"));
    } else if (arg == "--backbone") {
      option.backbone = need_value("--backbone");
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
    } else if (arg == "--block-learning-rate-scale") {
      option.block_learning_rate_scale =
          std::stod(need_value("--block-learning-rate-scale"));
    } else if (arg == "--model-file") {
      option.model_file = need_value("--model-file");
    } else if (arg == "--config-file") {
      option.config_file = need_value("--config-file");
    } else if (arg == "--attention-export-file") {
      option.attention_export_file = need_value("--attention-export-file");
    } else if (arg == "--corpus") {
      option.corpus = need_value("--corpus");
    } else if (arg == "--corpus-file") {
      option.corpus_file = need_value("--corpus-file");
    } else if (arg == "--save-model") {
      option.save_model = true;
    } else if (arg == "--no-save-model") {
      option.save_model = false;
    } else if (arg == "--eval-only") {
      option.eval_only = true;
    } else if (arg == "--force-train") {
      option.force_train = true;
    } else if (arg == "--help") {
      return false;
    } else {
      throw std::runtime_error(string("Unknown option: ") + arg);
    }
  }
  return true;
}

bool FileExists(const string &filename) {
  std::ifstream ifs(filename);
  return ifs.good();
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

MiniTransformerLM::BackboneType ParseBackboneType(const string &backbone) {
  if (backbone == "encoder") {
    return MiniTransformerLM::BACKBONE_ENCODER;
  }
  if (backbone == "decoder") {
    return MiniTransformerLM::BACKBONE_DECODER;
  }
  throw std::runtime_error("Invalid backbone, expected encoder or decoder");
}

string BuildConfigText(const DemoOption &option, const MiniTransformerLM &model,
                       const CharacterTokenizer &tokenizer, double average_loss,
                       double perplexity) {
  auto model_config = model.config();
  string backbone = model_config.backbone_type_ == MiniTransformerLM::BACKBONE_DECODER
                        ? "decoder"
                        : "encoder";
  return string("corpus_length=") + std::to_string(option.corpus.size()) + "\n" +
         "vocab_size=" + std::to_string(tokenizer.vocab_size()) + "\n" +
         "prompt=" + option.prompt + "\n" +
         "model_file=" + option.model_file + "\n" +
         "config_file=" + option.config_file + "\n" +
         "backbone=" + backbone + "\n" +
         "rand_seed=" + std::to_string(model_config.rand_seed_) + "\n" +
         "model_dim=" + std::to_string(model_config.model_dim_) + "\n" +
         "head_num=" + std::to_string(model_config.head_num_) + "\n" +
         "feed_forward_dim=" + std::to_string(model_config.feed_forward_dim_) + "\n" +
         "block_num=" + std::to_string(model_config.block_num_) + "\n" +
         "context_size=" + std::to_string(option.context_size) + "\n" +
         "block_learning_rate_scale=" +
         std::to_string(model_config.block_learning_rate_scale_) + "\n" +
         "generate_num=" + std::to_string(option.generate_num) + "\n" +
         "temperature=" + std::to_string(option.temperature) + "\n" +
         "top_k=" + std::to_string(option.top_k) + "\n" +
         "top_p=" + std::to_string(option.top_p) + "\n" +
         "save_model=" + std::string(option.save_model ? "true" : "false") +
         "\n" +
         "eval_only=" + std::string(option.eval_only ? "true" : "false") +
         "\n" +
         "average_loss=" + std::to_string(average_loss) + "\n" +
         "perplexity=" + std::to_string(perplexity) + "\n";
}

bool WriteTextFile(const string &filename, const string &content) {
  std::ofstream ofs(filename, std::ios::trunc);
  if (!ofs.is_open()) {
    return false;
  }
  ofs << content;
  return ofs.good();
}

string JsonEscape(const string &text) {
  string result;
  result.reserve(text.size());
  for (char ch : text) {
    if (ch == '\\') {
      result += "\\\\";
    } else if (ch == '"') {
      result += "\\\"";
    } else if (ch == '\n') {
      result += "\\n";
    } else if (ch == '\r') {
      result += "\\r";
    } else {
      result.push_back(ch);
    }
  }
  return result;
}

void WriteJsonTokenArray(std::ofstream &ofs, const CharacterTokenizer &tokenizer,
                         const vector<int> &token_ids) {
  ofs << "[";
  for (int i = 0; i < token_ids.size(); i++) {
    if (i != 0) {
      ofs << ", ";
    }
    string token(1, tokenizer.vocabulary()[token_ids[i]]);
    ofs << '"' << JsonEscape(token) << '"';
  }
  ofs << "]";
}

void WriteJsonLayers(std::ofstream &ofs, const MiniTransformerLM &model) {
  const auto &blocks = model.backbone_type() == MiniTransformerLM::BACKBONE_DECODER
                           ? model.decoder().blocks()
                           : model.encoder().blocks();
  ofs << "[\n";
  for (int layer = 0; layer < blocks.size(); layer++) {
    if (layer != 0) {
      ofs << ",\n";
    }
    ofs << "    {\n";
    ofs << "      \"layer_index\": " << layer << ",\n";
    ofs << "      \"heads\": [\n";
    const auto &heads = blocks[layer].self_attention().last_attention_weight();
    for (int head = 0; head < heads.size(); head++) {
      if (head != 0) {
        ofs << ",\n";
      }
      ofs << "        [\n";
      for (int row = 0; row < heads[head].size(); row++) {
        if (row != 0) {
          ofs << ",\n";
        }
        ofs << "          [";
        for (int col = 0; col < heads[head][row].size(); col++) {
          if (col != 0) {
            ofs << ", ";
          }
          ofs << heads[head][row][col];
        }
        ofs << "]";
      }
      ofs << "\n        ]";
    }
    ofs << "\n      ]\n";
    ofs << "    }";
  }
  ofs << "\n  ]";
}

bool WriteAttentionJson(const string &filename, MiniTransformerLM &model,
                        const CharacterTokenizer &tokenizer,
                        const vector<int> &prompt_token_ids,
                        const vector<int> &generated_token_ids) {
  std::ofstream ofs(filename, std::ios::trunc);
  if (!ofs.is_open()) {
    return false;
  }

  ofs << "{\n";
  ofs << "  \"backbone\": \""
      << (model.backbone_type() == MiniTransformerLM::BACKBONE_DECODER ? "decoder"
                                                                       : "encoder")
      << "\",\n";
  ofs << "  \"tokens\": ";
  WriteJsonTokenArray(ofs, tokenizer, generated_token_ids);
  ofs << ",\n";

  vector<int> step_tokens = prompt_token_ids;
  ofs << "  \"steps\": [\n";
  int step_count = generated_token_ids.size() - prompt_token_ids.size();
  for (int step = 0; step < step_count; step++) {
    int next_token_id = 0;
    if (model.PredictNextToken(step_tokens, next_token_id) != MiniTransformerLM::SUCCESS) {
      return false;
    }
    if (step != 0) {
      ofs << ",\n";
    }
    ofs << "    {\n";
    ofs << "      \"step_index\": " << step << ",\n";
    ofs << "      \"context_tokens\": ";
    WriteJsonTokenArray(ofs, tokenizer, step_tokens);
    ofs << ",\n";
    ofs << "      \"predicted_token\": \""
        << JsonEscape(string(1, tokenizer.vocabulary()[next_token_id])) << "\",\n";
    ofs << "      \"layers\": ";
    WriteJsonLayers(ofs, model);
    ofs << "\n    }";
    step_tokens.push_back(next_token_id);
  }
  ofs << "\n  ],\n";

  MiniTransformerLM::Matrix final_logits;
  if (model.Forward(generated_token_ids, final_logits,
                    model.backbone_type() != MiniTransformerLM::BACKBONE_ENCODER) !=
      MiniTransformerLM::SUCCESS) {
    return false;
  }
  ofs << "  \"layers\": ";
  WriteJsonLayers(ofs, model);
  ofs << "\n";
  ofs << "}\n";
  return ofs.good();
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

  if (!option.corpus_file.empty()) {
    if (!ReadTextFile(option.corpus_file, option.corpus)) {
      cout << "Read corpus file failed: " << option.corpus_file << endl;
      return -1;
    }
  }
  if (option.corpus.empty()) {
    cout << "Corpus is empty" << endl;
    return -1;
  }
  if (option.config_file.empty()) {
    option.config_file = option.model_file + ".config.txt";
  }

  CharacterTokenizer tokenizer;
  auto rc_tokenizer =
      tokenizer.Init(CharacterTokenizer::BuildVocabularyFromText(option.corpus));
  if (rc_tokenizer != CharacterTokenizer::SUCCESS) {
    cout << "Tokenizer init failed: " << tokenizer.err_msg() << endl;
    return -1;
  }

  vector<int> corpus_token_ids;
  rc_tokenizer = tokenizer.Encode(option.corpus, corpus_token_ids);
  if (rc_tokenizer != CharacterTokenizer::SUCCESS) {
    cout << "Encode corpus failed: " << tokenizer.err_msg() << endl;
    return -1;
  }

  CharacterDataset dataset;
  auto rc_dataset = dataset.Init(corpus_token_ids, option.context_size);
  if (rc_dataset != CharacterDataset::SUCCESS) {
    cout << "Dataset init failed: " << dataset.err_msg() << endl;
    return -1;
  }
  vector<vector<int>> input_samples;
  vector<int> target_tokens;
  rc_dataset = dataset.BuildNextTokenSamples(input_samples, target_tokens);
  if (rc_dataset != CharacterDataset::SUCCESS) {
    cout << "BuildNextTokenSamples failed: " << dataset.err_msg() << endl;
    return -1;
  }

  MiniTransformerLM model;
  MiniTransformerLM loaded_model;
  MiniTransformerLM *active_model = &model;
  MiniTransformerLM::Config model_config;
  model_config.vocab_size_ = tokenizer.vocab_size();
  model_config.model_dim_ = option.model_dim;
  model_config.head_num_ = option.head_num;
  model_config.feed_forward_dim_ = option.feed_forward_dim;
  model_config.block_num_ = option.block_num;
  model_config.rand_seed_ = option.rand_seed;
  model_config.max_context_size_ = option.context_size;
  model_config.use_positional_encoding_ = true;
  model_config.scale_embedding_ = true;
  model_config.block_learning_rate_scale_ = option.block_learning_rate_scale;
  try {
    model_config.backbone_type_ = ParseBackboneType(option.backbone);
  } catch (const std::exception &ex) {
    cout << ex.what() << endl;
    return -1;
  }

  auto rc = model.Init(model_config);
  if (rc != MiniTransformerLM::SUCCESS) {
    cout << "Model init failed: " << model.err_msg() << endl;
    return -1;
  }

  auto train_callback = [](int epoch_num, double average_loss, bool &early_stop) {
    if (epoch_num % 50 == 0) {
      cout << "epoch: " << epoch_num << " loss: " << average_loss << endl;
    }
    if (average_loss < 0.02) {
      early_stop = true;
    }
  };
  const string &model_file = option.model_file;
  if (option.eval_only) {
    if (!FileExists(model_file)) {
      cout << "Eval-only requires an existing model file: " << model_file << endl;
      return -1;
    }
    auto rc_loader =
        MiniTransformerLMLoader::ImportModelFromFile(loaded_model, model_file);
    if (rc_loader != MiniTransformerLMLoader::SUCCESS) {
      cout << "ImportModelFromFile failed" << endl;
      return -1;
    }
    if (loaded_model.vocab_size() != tokenizer.vocab_size()) {
      cout << "Loaded model vocab size does not match current corpus" << endl;
      return -1;
    }
    active_model = &loaded_model;
  } else if (!option.force_train && FileExists(model_file)) {
    auto rc_loader =
        MiniTransformerLMLoader::ImportModelFromFile(loaded_model, model_file);
    if (rc_loader != MiniTransformerLMLoader::SUCCESS) {
      cout << "ImportModelFromFile failed" << endl;
      return -1;
    }
    if (loaded_model.vocab_size() != tokenizer.vocab_size()) {
      cout << "Loaded model vocab size does not match current corpus" << endl;
      return -1;
    }
    active_model = &loaded_model;
  } else {
    // Linear warmup + cosine decay, a common recipe for Transformer training.
    int warmup_epochs = std::max(1, option.epoch_num / 20);
    WarmupCosineLR lr_scheduler(option.learning_rate, warmup_epochs,
                                option.epoch_num, option.learning_rate * 0.1);
    rc = model.TrainNextToken(input_samples, target_tokens, train_callback,
                              option.epoch_num, option.learning_rate,
                              &lr_scheduler);
    if (rc != MiniTransformerLM::SUCCESS) {
      cout << "TrainNextToken failed: " << model.err_msg() << endl;
      return -1;
    }

    if (option.save_model) {
      auto rc_loader = MiniTransformerLMLoader::ExportModelToFile(model, model_file);
      if (rc_loader != MiniTransformerLMLoader::SUCCESS) {
        cout << "ExportModelToFile failed" << endl;
        return -1;
      }
    }
  }

  double average_loss = 0;
  rc = active_model->CalcNextTokenLoss(input_samples, target_tokens, average_loss);
  if (rc != MiniTransformerLM::SUCCESS) {
    cout << "CalcNextTokenLoss failed: " << active_model->err_msg() << endl;
    return -1;
  }
  double perplexity = 0;
  rc = active_model->CalcPerplexity(input_samples, target_tokens, perplexity);
  if (rc != MiniTransformerLM::SUCCESS) {
    cout << "CalcPerplexity failed: " << active_model->err_msg() << endl;
    return -1;
  }

  string config_text =
      BuildConfigText(option, *active_model, tokenizer, average_loss, perplexity);
  if (!WriteTextFile(option.config_file, config_text)) {
    cout << "Write config file failed: " << option.config_file << endl;
    return -1;
  }

  string prompt = option.prompt;
  vector<int> token_ids;
  rc_tokenizer = tokenizer.Encode(prompt, token_ids);
  if (rc_tokenizer != CharacterTokenizer::SUCCESS) {
    cout << "Encode prompt failed: " << tokenizer.err_msg() << endl;
    return -1;
  }

  vector<int> generated_token_ids;
  rc = active_model->Generate(token_ids, option.generate_num, generated_token_ids);
  if (rc != MiniTransformerLM::SUCCESS) {
    cout << "Generate failed: " << active_model->err_msg() << endl;
    return -1;
  }

  MiniTransformerLM::SamplingOption sampling_option;
  sampling_option.temperature_ = option.temperature;
  sampling_option.top_k_ = option.top_k;
  sampling_option.top_p_ = option.top_p;
  vector<int> sampled_generated_token_ids;
  rc = active_model->GenerateSample(token_ids, option.generate_num,
                                    sampled_generated_token_ids, sampling_option);
  if (rc != MiniTransformerLM::SUCCESS) {
    cout << "GenerateSample failed: " << active_model->err_msg() << endl;
    return -1;
  }

  string generated_text;
  rc_tokenizer = tokenizer.Decode(generated_token_ids, generated_text);
  if (rc_tokenizer != CharacterTokenizer::SUCCESS) {
    cout << "Decode result failed: " << tokenizer.err_msg() << endl;
    return -1;
  }

  string sampled_generated_text;
  rc_tokenizer = tokenizer.Decode(sampled_generated_token_ids, sampled_generated_text);
  if (rc_tokenizer != CharacterTokenizer::SUCCESS) {
    cout << "Decode sampled result failed: " << tokenizer.err_msg() << endl;
    return -1;
  }

  if (!option.attention_export_file.empty()) {
    MiniTransformerLM::Matrix attention_logits;
    rc = active_model->Forward(generated_token_ids, attention_logits,
                               active_model->backbone_type() != MiniTransformerLM::BACKBONE_ENCODER);
    if (rc != MiniTransformerLM::SUCCESS) {
      cout << "Forward for attention export failed: " << active_model->err_msg() << endl;
      return -1;
    }
    if (!WriteAttentionJson(option.attention_export_file, *active_model, tokenizer,
                            token_ids, generated_token_ids)) {
      cout << "Write attention export failed: " << option.attention_export_file << endl;
      return -1;
    }
  }

  string first_sample_text;
  rc_tokenizer = tokenizer.Decode(input_samples[0], first_sample_text);
  if (rc_tokenizer != CharacterTokenizer::SUCCESS) {
    cout << "Decode sample failed: " << tokenizer.err_msg() << endl;
    return -1;
  }

  cout << "Prompt: " << prompt << endl;
  cout << "Dataset sample[0]: " << first_sample_text << " -> "
       << tokenizer.vocabulary()[target_tokens[0]] << endl;
  cout << "Backbone: "
       << (active_model->backbone_type() == MiniTransformerLM::BACKBONE_DECODER
               ? "decoder"
               : "encoder")
       << " blocks: " << active_model->block_num()
       << " model_dim: " << active_model->model_dim() << endl;
  cout << "Loss: " << average_loss << " Perplexity: " << perplexity << endl;
  cout << "Greedy: " << generated_text << endl;
  cout << "Sampled: " << sampled_generated_text << endl;
  cout << "Config saved: " << option.config_file << endl;
  if (!option.attention_export_file.empty()) {
    cout << "Attention export: " << option.attention_export_file << endl;
  }
  cout << "Model save: " << (option.save_model ? "on" : "off")
       << " eval_only: " << (option.eval_only ? "true" : "false") << endl;
  cout << "This demo trains, saves, reloads, and generates with a tiny "
       << (active_model->backbone_type() == MiniTransformerLM::BACKBONE_DECODER
               ? "decoder-only"
               : "encoder-style")
       << " character language model." << endl;
  return 0;
}

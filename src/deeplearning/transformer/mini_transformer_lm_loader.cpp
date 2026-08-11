#include "mini_transformer_lm_loader.h"

#include <cstring>
#include <fstream>
#include <utility>

namespace deeplearning {
namespace {

constexpr char kExtensionMagic[] = "MTLMEX1";
constexpr int kExtensionVersion = 1;

bool WriteInt(std::ofstream &ofs, int value) {
  return ofs.write((const char *)&value, sizeof(value)).good();
}

bool ReadInt(std::ifstream &ifs, int &value) {
  return ifs.read((char *)&value, sizeof(value)).good();
}

bool WriteDouble(std::ofstream &ofs, double value) {
  return ofs.write((const char *)&value, sizeof(value)).good();
}

bool ReadDouble(std::ifstream &ifs, double &value) {
  return ifs.read((char *)&value, sizeof(value)).good();
}

bool WriteMatrix(std::ofstream &ofs,
                 const std::vector<std::vector<double>> &matrix) {
  if (!WriteInt(ofs, matrix.size())) {
    return false;
  }
  for (const auto &row : matrix) {
    if (!WriteInt(ofs, row.size())) {
      return false;
    }
    for (double value : row) {
      if (!WriteDouble(ofs, value)) {
        return false;
      }
    }
  }
  return true;
}

bool ReadMatrix(std::ifstream &ifs, std::vector<std::vector<double>> &matrix) {
  int row_size = 0;
  if (!ReadInt(ifs, row_size) || row_size < 0) {
    return false;
  }
  matrix.resize(row_size);
  for (auto &row : matrix) {
    int col_size = 0;
    if (!ReadInt(ifs, col_size) || col_size < 0) {
      return false;
    }
    row.resize(col_size);
    for (double &value : row) {
      if (!ReadDouble(ifs, value)) {
        return false;
      }
    }
  }
  return true;
}

bool WriteVector(std::ofstream &ofs, const std::vector<double> &arr) {
  if (!WriteInt(ofs, arr.size())) {
    return false;
  }
  for (double value : arr) {
    if (!WriteDouble(ofs, value)) {
      return false;
    }
  }
  return true;
}

bool ReadVector(std::ifstream &ifs, std::vector<double> &arr) {
  int size = 0;
  if (!ReadInt(ifs, size) || size < 0) {
    return false;
  }
  arr.resize(size);
  for (double &value : arr) {
    if (!ReadDouble(ifs, value)) {
      return false;
    }
  }
  return true;
}

bool WriteBlock(std::ofstream &ofs, const TransformerBlock &block) {
  return WriteMatrix(ofs, block.self_attention().query_weight()) &&
         WriteMatrix(ofs, block.self_attention().key_weight()) &&
         WriteMatrix(ofs, block.self_attention().value_weight()) &&
         WriteMatrix(ofs, block.self_attention().output_weight()) &&
         WriteVector(ofs, block.attention_norm().scale()) &&
         WriteVector(ofs, block.attention_norm().bias()) &&
         WriteMatrix(ofs, block.feed_forward_weight_1()) &&
         WriteVector(ofs, block.feed_forward_bias_1()) &&
         WriteMatrix(ofs, block.feed_forward_weight_2()) &&
         WriteVector(ofs, block.feed_forward_bias_2()) &&
         WriteVector(ofs, block.feed_forward_norm().scale()) &&
         WriteVector(ofs, block.feed_forward_norm().bias());
}

bool ReadBlock(std::ifstream &ifs, TransformerBlock &block) {
  std::vector<std::vector<double>> query_weight;
  std::vector<std::vector<double>> key_weight;
  std::vector<std::vector<double>> value_weight;
  std::vector<std::vector<double>> output_weight;
  std::vector<std::vector<double>> ff_weight_1;
  std::vector<std::vector<double>> ff_weight_2;
  std::vector<double> attention_scale;
  std::vector<double> attention_bias;
  std::vector<double> ff_bias_1;
  std::vector<double> ff_bias_2;
  std::vector<double> ff_scale;
  std::vector<double> ff_norm_bias;
  if (!ReadMatrix(ifs, query_weight) || !ReadMatrix(ifs, key_weight) ||
      !ReadMatrix(ifs, value_weight) || !ReadMatrix(ifs, output_weight) ||
      !ReadVector(ifs, attention_scale) || !ReadVector(ifs, attention_bias) ||
      !ReadMatrix(ifs, ff_weight_1) || !ReadVector(ifs, ff_bias_1) ||
      !ReadMatrix(ifs, ff_weight_2) || !ReadVector(ifs, ff_bias_2) ||
      !ReadVector(ifs, ff_scale) || !ReadVector(ifs, ff_norm_bias)) {
    return false;
  }

  return block.self_attention().set_query_weight(query_weight) ==
             SelfAttention::SUCCESS &&
         block.self_attention().set_key_weight(key_weight) ==
             SelfAttention::SUCCESS &&
         block.self_attention().set_value_weight(value_weight) ==
             SelfAttention::SUCCESS &&
         block.self_attention().set_output_weight(output_weight) ==
             SelfAttention::SUCCESS &&
         block.attention_norm().set_scale(attention_scale) == LayerNorm::SUCCESS &&
         block.attention_norm().set_bias(attention_bias) == LayerNorm::SUCCESS &&
         block.set_feed_forward_weight_1(ff_weight_1) == TransformerBlock::SUCCESS &&
         block.set_feed_forward_bias_1(ff_bias_1) == TransformerBlock::SUCCESS &&
         block.set_feed_forward_weight_2(ff_weight_2) == TransformerBlock::SUCCESS &&
         block.set_feed_forward_bias_2(ff_bias_2) == TransformerBlock::SUCCESS &&
         block.feed_forward_norm().set_scale(ff_scale) == LayerNorm::SUCCESS &&
         block.feed_forward_norm().set_bias(ff_norm_bias) == LayerNorm::SUCCESS;
}

} // namespace

MiniTransformerLMLoader::RC
MiniTransformerLMLoader::ExportModelToFile(const MiniTransformerLM &model,
                                           const std::string &filename) {
  std::ofstream ofs(filename, std::ios::binary | std::ios::trunc);
  if (!ofs.is_open()) {
    return EXPORT_ERROR;
  }

  ModelConfig config = {model.rand_seed(),
                        model.vocab_size(),
                        model.model_dim(),
                        model.head_num(),
                        model.feed_forward_dim(),
                        model.block_num(),
                        model.max_context_size(),
                        model.backbone_type(),
                        model.use_positional_encoding() ? 1 : 0,
                        model.scale_embedding() ? 1 : 0,
                        model.block_learning_rate_scale()};
  if (!ofs.write((const char *)&config, sizeof(config)).good() ||
      !WriteMatrix(ofs, model.token_embedding().embedding_table()) ||
      !WriteMatrix(ofs, model.output_weight()) ||
      !WriteVector(ofs, model.output_bias())) {
    ofs.close();
    return EXPORT_ERROR;
  }

  const auto &blocks = model.backbone_type() == MiniTransformerLM::BACKBONE_DECODER
                            ? model.decoder().blocks()
                            : model.encoder().blocks();
  for (const auto &block : blocks) {
    if (!WriteBlock(ofs, block)) {
      ofs.close();
      return EXPORT_ERROR;
    }
  }
  if (!ofs.write(kExtensionMagic, sizeof(kExtensionMagic) - 1).good() ||
      !WriteInt(ofs, kExtensionVersion) ||
      !WriteInt(ofs, model.block_num())) {
    ofs.close();
    return EXPORT_ERROR;
  }
  const std::vector<const std::vector<TransformerBlock> *> gate_stacks = {
      &model.encoder().blocks(), &model.decoder().blocks()};
  for (const auto *stack : gate_stacks) {
    for (const auto &block : *stack) {
      if (!WriteDouble(ofs, block.depth_residual_scale()) ||
          !WriteInt(ofs, block.depth_residual_trainable() ? 1 : 0)) {
        ofs.close();
        return EXPORT_ERROR;
      }
    }
  }

  ofs.close();
  return SUCCESS;
}

MiniTransformerLMLoader::RC
MiniTransformerLMLoader::ImportModelFromFile(MiniTransformerLM &model,
                                             const std::string &filename) {
  if (model.vocab_size() != 0) {
    return INPORT_ERROR;
  }
  std::ifstream ifs(filename, std::ios::binary);
  if (!ifs.is_open()) {
    return INPORT_ERROR;
  }

  ModelConfig config;
  if (!ifs.read((char *)&config, sizeof(config)).good()) {
    ifs.close();
    return INPORT_ERROR;
  }

  MiniTransformerLM candidate;
  candidate.set_random_seed(config.rand_seed_);
  candidate.set_backbone_type(
      static_cast<MiniTransformerLM::BackboneType>(config.backbone_type_));
  candidate.set_use_positional_encoding(config.use_positional_encoding_ != 0);
  candidate.set_scale_embedding(config.scale_embedding_ != 0);
  candidate.set_max_context_size(config.max_context_size_);
  candidate.set_block_learning_rate_scale(config.block_learning_rate_scale_);
  if (candidate.Init(config.vocab_size_, config.model_dim_, config.head_num_,
                     config.feed_forward_dim_, config.block_num_) !=
      MiniTransformerLM::SUCCESS) {
    ifs.close();
    return INPORT_ERROR;
  }

  std::vector<std::vector<double>> embedding_table;
  std::vector<std::vector<double>> output_weight;
  std::vector<double> output_bias;
  if (!ReadMatrix(ifs, embedding_table) || !ReadMatrix(ifs, output_weight) ||
      !ReadVector(ifs, output_bias) ||
      candidate.token_embedding().set_embedding_table(embedding_table) !=
          TokenEmbedding::SUCCESS ||
      candidate.set_output_weight(output_weight) != MiniTransformerLM::SUCCESS ||
      candidate.set_output_bias(output_bias) != MiniTransformerLM::SUCCESS) {
    ifs.close();
    return INPORT_ERROR;
  }

  for (int i = 0; i < config.block_num_; i++) {
    auto *block = config.backbone_type_ == MiniTransformerLM::BACKBONE_DECODER
                      ? candidate.decoder().mutable_block(i)
                      : candidate.encoder().mutable_block(i);
    if (block == nullptr || !ReadBlock(ifs, *block)) {
      ifs.close();
      return INPORT_ERROR;
    }
  }

  if (ifs.peek() != std::char_traits<char>::eof()) {
    char magic[sizeof(kExtensionMagic) - 1] = {};
    int version = 0;
    int block_count = 0;
    if (!ifs.read(magic, sizeof(magic)).good() ||
        std::memcmp(magic, kExtensionMagic, sizeof(magic)) != 0 ||
        !ReadInt(ifs, version) || version != kExtensionVersion ||
        !ReadInt(ifs, block_count) || block_count != config.block_num_) {
      ifs.close();
      return INPORT_ERROR;
    }
    for (int stack = 0; stack < 2; stack++) {
      for (int i = 0; i < block_count; i++) {
        double scale = 1.0;
        int trainable = 0;
        auto *block = stack == 0 ? candidate.encoder().mutable_block(i)
                                 : candidate.decoder().mutable_block(i);
        if (block == nullptr || !ReadDouble(ifs, scale) ||
            !ReadInt(ifs, trainable) || (trainable != 0 && trainable != 1) ||
            block->set_depth_residual(scale, trainable != 0) !=
                TransformerBlock::SUCCESS) {
          ifs.close();
          return INPORT_ERROR;
        }
      }
    }
    if (ifs.peek() != std::char_traits<char>::eof()) {
      ifs.close();
      return INPORT_ERROR;
    }
  }

  ifs.close();
  model = std::move(candidate);
  return SUCCESS;
}

} // namespace deeplearning

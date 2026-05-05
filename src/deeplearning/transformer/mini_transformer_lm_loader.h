#pragma once

#include "mini_transformer_lm.h"

#include <string>

namespace deeplearning {

class MiniTransformerLMLoader {
public:
  enum RC {
    SUCCESS,
    EXPORT_ERROR,
    INPORT_ERROR,
  };

public:
  static RC ExportModelToFile(const MiniTransformerLM &model,
                              const std::string &filename);
  static RC ImportModelFromFile(MiniTransformerLM &model,
                                const std::string &filename);

private:
  struct ModelConfig {
    int rand_seed_;
    int vocab_size_;
    int model_dim_;
    int head_num_;
    int feed_forward_dim_;
    int block_num_;
    int context_size_;
    int backbone_type_;
    int use_positional_encoding_;
    int scale_embedding_;
    double block_learning_rate_scale_;
  };
};

} // namespace deeplearning

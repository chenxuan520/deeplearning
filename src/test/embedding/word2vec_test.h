#pragma once

#include "embedding/word2vec.h"
#include "embedding/word_tokenizer.h"
#include "test.h"

TEST(WordTokenizer, BuildsVocabularyAndTokenizes) {
  WordTokenizer tokenizer;
  const std::string text =
      "The cat sat on the mat.\nThe DOG sat on the log.";
  MUST_TRUE(tokenizer.InitFromText(text) == WordTokenizer::SUCCESS,
            tokenizer.err_msg());
  MUST_TRUE(tokenizer.vocab_size() >= 6, "vocabulary should contain core words");

  std::vector<std::vector<int>> sentences;
  MUST_TRUE(tokenizer.TokenizeSentences(text, sentences) ==
                WordTokenizer::SUCCESS,
            tokenizer.err_msg());
  MUST_TRUE(sentences.size() == 2, "should split into two sentences");
  MUST_TRUE(tokenizer.Lookup("cat") >= 0, "cat should exist");
  MUST_TRUE(tokenizer.Lookup("dog") >= 0, "dog should be lowercased");
}

TEST(Word2Vec, SkipGramPullsRelatedWordsTogether) {
  const std::string corpus = R"(the cat sat on the mat
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

  WordTokenizer tokenizer;
  MUST_TRUE(tokenizer.InitFromText(corpus) == WordTokenizer::SUCCESS,
            tokenizer.err_msg());

  std::vector<std::vector<int>> sentences;
  MUST_TRUE(tokenizer.TokenizeSentences(corpus, sentences) ==
                WordTokenizer::SUCCESS,
            tokenizer.err_msg());

  Word2Vec model;
  Word2Vec::Config config;
  config.embed_dim = 16;
  config.window_size = 2;
  config.negative_num = 5;
  config.epochs = 40;
  config.learning_rate = 0.08;
  config.rand_seed = 7;
  config.mode = Word2Vec::Mode::SKIP_GRAM;
  MUST_TRUE(model.Init(tokenizer.vocab_size(), config) == Word2Vec::SUCCESS,
            model.err_msg());

  Word2Vec::TrainStats stats;
  MUST_TRUE(model.Train(sentences, &stats) == Word2Vec::SUCCESS,
            model.err_msg());
  MUST_TRUE(stats.pair_count > 0, "training should process pairs");

  const int cat_id = tokenizer.Lookup("cat");
  const int dog_id = tokenizer.Lookup("dog");
  const int apple_id = tokenizer.Lookup("apples");
  MUST_TRUE(cat_id >= 0 && dog_id >= 0 && apple_id >= 0,
            "test words should exist");

  const double cat_dog = model.CosineSimilarity(cat_id, dog_id);
  const double cat_apple = model.CosineSimilarity(cat_id, apple_id);
  MUST_TRUE(cat_dog > cat_apple,
            "cat should be closer to dog than to apples after training");

  const auto neighbors = model.MostSimilar(cat_id, 3);
  MUST_TRUE(!neighbors.empty(), "nearest neighbors should not be empty");
}

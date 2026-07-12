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

TEST(WordTokenizer, UnknownAndDecodeRoundTrip) {
  WordTokenizer tokenizer;
  const std::string text = "Alice was beginning to get very tired";
  MUST_EQUAL(tokenizer.InitFromText(text, true), WordTokenizer::SUCCESS);
  MUST_EQUAL(tokenizer.Lookup("<unk>"), 0);
  MUST_TRUE(tokenizer.Lookup("alice") > 0, "alice should be in vocabulary");

  std::vector<int> token_ids;
  int unknown_count = 0;
  MUST_EQUAL(tokenizer.TokenizeFlatWithUnknown("Alice met Dinah", token_ids,
                                               unknown_count),
             WordTokenizer::SUCCESS);
  MUST_EQUAL(unknown_count, 2);
  MUST_EQUAL(token_ids[0], tokenizer.Lookup("alice"));
  MUST_EQUAL(token_ids[1], tokenizer.Lookup("<unk>"));
  MUST_EQUAL(token_ids[2], tokenizer.Lookup("<unk>"));

  std::string decoded;
  MUST_EQUAL(tokenizer.Decode(token_ids, decoded), WordTokenizer::SUCCESS);
  MUST_EQUAL(decoded, "alice <unk> <unk>");
}

TEST(WordTokenizer, MaxVocabKeepsMostFrequentWords) {
  WordTokenizer tokenizer;
  const std::string text =
      "the cat sat the cat ran the dog barked the dog ran owl";
  WordTokenizer::Config config;
  config.add_unknown_token = true;
  config.max_vocab_size = 3;
  MUST_EQUAL(tokenizer.InitFromText(text, config), WordTokenizer::SUCCESS);
  MUST_EQUAL(tokenizer.Lookup("<unk>"), 0);
  MUST_EQUAL(tokenizer.vocab_size(), 4);
  MUST_TRUE(tokenizer.Lookup("the") > 0, "the should be kept");
  MUST_TRUE(tokenizer.Lookup("cat") > 0, "cat should be kept");
  MUST_TRUE(tokenizer.Lookup("dog") > 0, "dog should be kept");
  MUST_EQUAL(tokenizer.Lookup("owl"), -1);

  std::vector<int> token_ids;
  int unknown_count = 0;
  MUST_EQUAL(tokenizer.TokenizeFlatWithUnknown("the owl dog", token_ids,
                                               unknown_count),
             WordTokenizer::SUCCESS);
  MUST_EQUAL(unknown_count, 1);

  std::string decoded;
  MUST_EQUAL(tokenizer.Decode(token_ids, decoded), WordTokenizer::SUCCESS);
  MUST_EQUAL(decoded, "the <unk> dog");
}

TEST(WordTokenizer, DropUnknownPolicySkipsOutOfVocabWords) {
  WordTokenizer tokenizer;
  const std::string text =
      "the cat sat the cat ran the dog barked the dog ran owl";
  WordTokenizer::Config config;
  config.unknown_policy = WordTokenizer::UNKNOWN_DROP;
  config.max_vocab_size = 3;
  MUST_EQUAL(tokenizer.InitFromText(text, config), WordTokenizer::SUCCESS);
  MUST_EQUAL(tokenizer.Lookup("<unk>"), -1);
  MUST_EQUAL(tokenizer.vocab_size(), 3);
  MUST_TRUE(tokenizer.Lookup("the") >= 0, "the should be kept");
  MUST_TRUE(tokenizer.Lookup("cat") >= 0, "cat should be kept");
  MUST_TRUE(tokenizer.Lookup("dog") >= 0, "dog should be kept");

  std::vector<int> token_ids;
  int unknown_count = 0;
  MUST_EQUAL(tokenizer.TokenizeFlatWithPolicy("the owl dog", token_ids,
                                              unknown_count),
             WordTokenizer::SUCCESS);
  MUST_EQUAL(unknown_count, 1);

  std::string decoded;
  MUST_EQUAL(tokenizer.Decode(token_ids, decoded), WordTokenizer::SUCCESS);
  MUST_EQUAL(decoded, "the dog");
}

TEST(WordTokenizer, InitFromVocabularyRejectsDuplicates) {
  WordTokenizer tokenizer;
  const std::vector<std::string> vocabulary = {"<unk>", "the", "alice", "The"};
  MUST_EQUAL(tokenizer.InitFromVocabulary(vocabulary),
             WordTokenizer::INVALID_DATA);
}

TEST(WordTokenizer, InitFromVocabularyPreservesIds) {
  WordTokenizer tokenizer;
  const std::vector<std::string> vocabulary = {"<unk>", "the", "alice",
                                               "said"};
  MUST_EQUAL(tokenizer.InitFromVocabulary(vocabulary), WordTokenizer::SUCCESS);
  MUST_EQUAL(tokenizer.Lookup("<unk>"), 0);
  MUST_EQUAL(tokenizer.Lookup("the"), 1);
  MUST_EQUAL(tokenizer.Lookup("alice"), 2);
  MUST_EQUAL(tokenizer.Lookup("said"), 3);

  std::vector<int> token_ids;
  int unknown_count = 0;
  MUST_EQUAL(tokenizer.TokenizeFlatWithUnknown("The Alice smiled", token_ids,
                                               unknown_count),
             WordTokenizer::SUCCESS);
  MUST_EQUAL(unknown_count, 1);

  std::string decoded;
  MUST_EQUAL(tokenizer.Decode(token_ids, decoded), WordTokenizer::SUCCESS);
  MUST_EQUAL(decoded, "the alice <unk>");
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

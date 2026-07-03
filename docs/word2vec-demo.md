# word2vec 演示

配套第 15 章的最小 **skip-gram + 负采样** 训练程序，代码在 `src/deeplearning/embedding/`。

## 构建

```bash
cd src
./build.sh
```

## 运行

在 `src/` 目录下：

```bash
./bin/word2vec
./bin/word2vec --show-pairs
./bin/word2vec --query-word cat --compare-word dog --epochs 40
./bin/word2vec --mode cbow --query-word king --compare-word queen
```

常用参数：

| 参数 | 含义 | 默认 |
|------|------|------|
| `--corpus-file` | 语料文本（一行一句） | 内置小动物语料 |
| `--mode` | `skip-gram` / `cbow` | skip-gram |
| `--embed-dim` | 词向量维度 | 16 |
| `--window-size` | 上下文窗口半径 | 2 |
| `--negative-num` | 负采样个数 | 5 |
| `--epochs` | 训练轮数 | 30 |
| `--query-word` | 查最近邻的词 | cat |
| `--compare-word` | 算余弦相似度的第二个词 | dog |
| `--show-pairs` | 打印几条滑窗训练对 | 关 |

## 代码对应关系

| 书中概念 | 源码 |
|----------|------|
| 分词 / 词表 | `WordTokenizer` · `word_tokenizer.cpp` |
| W<sub>in</sub> / W<sub>out</sub> 两张表 | `Word2Vec::input_embeddings_` / `output_embeddings_` |
| 滑窗造 (中心, 上下文) 样本 | `BuildSkipGramPairs` |
| 正样本 sigmoid 拉近 | `UpdatePositivePair` |
| 负采样推开 | `UpdateNegativePair` + `SampleNegative` |
| 训完查相似词 | `CosineSimilarity` / `MostSimilar` |

## 预期现象

默认语料训练后，`cat` 与 `dog` 的余弦相似度应明显高于 `cat` 与 `apples`（无关词）。单元测试 `Word2Vec SkipGramPullsRelatedWordsTogether` 会检查这一点。

## 测试

```bash
./bin/test_bin "Word2Vec|WordTokenizer"
```

## 书中链接

- [第 15 章 · 词嵌入与 word2vec](https://chenxuan520.github.io/deeplearning/chapter-15.html)

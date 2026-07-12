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

## 它是怎么训练的（一个"只训练词向量"的模型）

和语言模型（比如本仓库的 `mini_lm`）最大的不同：在语言模型里，词向量只是**输入层的中间产物**，真正要的是"能预测下一个词"的整个网络；而在 word2vec 里，**整个"模型"就是词向量表本身，训练的唯一目的就是把这张表打磨好**，训完预测部分直接扔掉。

### 整个模型就两张表，没有别的

`InitEmbeddings`（`word2vec.cpp:40`）只随机初始化两张 `vocab × embed-dim` 的表，**没有 attention、没有 Transformer block、没有隐藏层**：

- `input_embeddings_`：一个词**当中心词时**的向量；
- `output_embeddings_`：同一个词**当上下文时**的向量。

训练完只保留 `input_embeddings_` 当作最终词向量（`InputVector`），`output_embeddings_` 是训练时的"陪练"，用完丢弃。

### 训练目标：负采样把"全词表 softmax"换成一堆二分类

语言模型每预测一步要对整个词表做 softmax（词表一大就很贵）。word2vec 用**负采样（negative sampling）**绕开这件事——每次只做几个二分类：

- **正样本**：真实共现的（中心词, 上下文词）对，把两者向量的点积经 sigmoid **往 1 推**（`UpdatePositivePair`，`word2vec.cpp:264`）；
- **负样本**：随机抽 `--negative-num` 个词，把点积经 sigmoid **往 0 压**（`UpdateNegativePair`）；
- 单对的 loss = 正样本 `-log σ(点积)` + 各负样本 `-log(1 − σ(点积))`（`TrainSkipGramPair`，`word2vec.cpp:292`）。

负样本不是均匀抽，而是按**词频的 0.75 次方**分布采样（既照顾高频词、又不让它们淹没一切），见 `BuildNoiseDistribution`（`word2vec.cpp:148`，`pow(freq, 0.75)`）+ `SampleNegative`（前缀和 + 二分查找）。

直觉：**共现的词互相拉近，随机词互相推开**，反复做下去，语义相近的词向量就聚到一起了。

### 两种造样本的方向：skip-gram 与 CBOW

同一个"共现"目标，两种对偶的做法：

- **skip-gram**（默认）：用**中心词**去预测窗口里的**每个上下文词**（`BuildSkipGramPairs`，`word2vec.cpp:174`）；
- **CBOW**：反过来，用**上下文词的平均向量**去预测**中心词**（`AverageInputVectors` + `TrainCBOWSample`，`word2vec.cpp:332`）。

两者都不看词序——窗口内是"词袋"，这也是 word2vec 和有位置编码的 Transformer 的根本区别。

### 学习率线性衰减

每个 epoch 的学习率从 `--learning-rate` 线性降到 `min_learning_rate`（`Train` 里按 `progress` 插值，`word2vec.cpp:98`），越训到后面步子越小，收敛更稳。

### 训完怎么用

词向量的好坏用**余弦相似度**衡量：语义近的词，向量夹角小、相似度高。`CosineSimilarity`（`word2vec.cpp:372`）+ `MostSimilar` 就是"找最像的词"，也就是经典的 `cat` 最像 `dog` 那套（本 demo 语料很小，别期待 `king − man + woman ≈ queen` 那种大语料才有的效果）。

## 代码对应关系

| 书中概念 | 源码 |
|----------|------|
| 分词 / 词表 | `WordTokenizer` · `word_tokenizer.cpp` |
| W<sub>in</sub> / W<sub>out</sub> 两张表 | `Word2Vec::input_embeddings_` / `output_embeddings_` |
| 滑窗造 (中心, 上下文) 样本 | `BuildSkipGramPairs` |
| CBOW：上下文平均预测中心词 | `AverageInputVectors` + `TrainCBOWSample` |
| 正样本 sigmoid 拉近 | `UpdatePositivePair` |
| 负采样推开 | `UpdateNegativePair` + `SampleNegative` |
| 负样本按词频<sup>0.75</sup> 采样 | `BuildNoiseDistribution` |
| 训完查相似词 | `CosineSimilarity` / `MostSimilar` |

## 预期现象

默认语料训练后，`cat` 与 `dog` 的余弦相似度应明显高于 `cat` 与 `apples`（无关词）。单元测试 `Word2Vec SkipGramPullsRelatedWordsTogether` 会检查这一点。

## 测试

```bash
./bin/test_bin "Word2Vec|WordTokenizer"
```

## 书中链接

- [第 15 章 · 词嵌入与 word2vec](https://chenxuan520.github.io/deeplearning/chapter-15.html)

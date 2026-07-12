# mini_lm 基座模型 Demo

`src/bin/mini_lm` 是一个围绕 `deeplearning::MiniTransformerLM` 的**小型语言模型命令行工具**，把「定义结构 → 喂语料训练 → 加载推理」拆成独立子命令，方便当作一个小小的**基座模型**来用：先初始化一个空模型，再用任意文本把它喂大，最后加载参数做文本续写。它默认使用字符级 tokenizer，也可以通过 `--tokenizer word` 切到词级 tokenizer。

它复用的是仓库已有的 Transformer 模块（`src/deeplearning/transformer/`）；demo 层负责命令调度、数据 IO、tokenizer sidecar 读写，词级模式复用并扩展了 `src/deeplearning/embedding/word_tokenizer.*`。

## 能力边界（先说清楚，别误解）

- 默认是**字符级续写模型**：逐个字符地学「看到这些字符，下一个字符最可能是什么」。训练充分后它能续写出**像英文的字符流**（真实单词、空格、标点），而不是背 `abcabc`。
- 也支持**词级续写模型**：`--tokenizer word` 会按英文单词切分，小写化后预测下一个词，生成结果用空格拼回文本，例如 `alice was beginning to get very tired`。可用 `--max-vocab-size` 只保留高频词，词表外单词默认映射到 `<unk>`，也可用 `--unknown-policy drop` 直接丢弃，控制输出层大小和生成可读性。
- 它**不是对话模型**。你对它说 `hello` 它不会"理解并回答 hello"——它只会沿着 `hello` 这个前缀，按训练语料的统计继续往下写字符。想要"问答/指令遵循"需要指令微调 + 大得多的模型，超出本库范围。
- 底层是**朴素实现**（纯 `std::vector`、逐样本、无 batch、无 BLAS/SIMD、Debug 构建），算力有限。请把它当**教学/实验**用途，模型规模和语料都要按「你的机器能训得动」来选（见 [训练成本与选参](#10-训练成本与选参)）。

## 1. 构建

```bash
cd src
./build.sh                 # 产出 ./bin/mini_lm
```

## 2. 快速上手：四个子命令跑一遍

工作目录建议在 `src/` 下。下面用一份很小的语料把完整链路跑通：

```bash
cd src

# ① init：定义模型结构，并从语料扫出词表，随机初始化后保存
./bin/mini_lm init --model /tmp/toy.param --corpus "hello mini language model" \
    --model-dim 16 --head-num 2 --feed-forward-dim 32 --block-num 1 --context-size 8

# ② info：确认它初始化了哪些参数、词表多大
./bin/mini_lm info --model /tmp/toy.param

# ③ train：加载模型，喂语料训练，权重写回原文件
./bin/mini_lm train --model /tmp/toy.param --corpus "hello mini language model" \
    --epochs 400 --learning-rate 0.01

# ④ generate：加载模型，从 prompt 续写
./bin/mini_lm generate --model /tmp/toy.param --prompt "hel" --generate-num 20
```

`init` 无参或加 `--help` 会打印用法；每个子命令都支持 `<command> --help`。

## 3. 模型文件长什么样（两个 sidecar 文件）

一个模型 = **两个文件**，共享同一个基础路径：

| 文件 | 内容 | 谁写的 |
|------|------|--------|
| `<model>`（如 `toy.param`） | 二进制权重 + 结构超参 | `MiniTransformerLMLoader`（库自带格式） |
| `<model>.vocab`（如 `toy.param.vocab`） | tokenizer 类型 + 词表 | 本 demo |

**为什么要单独存词表？** 库的模型文件只存了权重和结构，没存"token↔编号"的对应关系。而语言模型必须靠这张表才能把文本编码成 token、把输出解码回文本。所以本 demo 在 `init` 时把词表落到 `.vocab` sidecar，`train`/`generate`/`info` 都会连它一起加载。**两个文件要一起拷贝、一起备份**，丢了 `.vocab` 就没法用了。

> 词表在 `init` 时就**固定死**了（它的大小 = embedding 行数 = 输出层维度，属于模型结构）。之后 `train`/`generate` 遇到字符级词表外字符会**自动跳过并告警**；词级模型遇到词表外单词时，默认映射到 `<unk>`，也可以在 `init` 时选择 `--unknown-policy drop` 直接丢弃。`init` 用的语料要尽量覆盖你后续会用到的字符集或常用词；如果使用 `--max-vocab-size`，低频词会被有意留在词表外。

### `.vocab` sidecar 格式

字符级模型为了兼容旧文件，仍然把字符表按原始字节直接写进 `.vocab`。词级模型会写成可区分的行格式：

```text
tokenizer=word
<unk>
the
alice
said
...
```

第一行说明 tokenizer 类型；后面每行一个词，行号就是 token id。默认 `map` 策略下，`<unk>` 固定在 id 0，用来承接推理或续训时没见过的词；`drop` 策略下不会写 `<unk>`，词表外词会在编码时被跳过。

## 4. 词表是怎么来的、怎么训练的

先破除一个常见混淆：**词表本身不训练，被训练的是「词向量」。** 这是两个不同的东西：

| | 词表（vocabulary） | 词向量（embedding table） |
|---|---|---|
| 是什么 | 一张 `token ↔ 编号` 的静态查找表 | 每个编号对应一行 `model-dim` 维的可学向量 |
| 怎么来的 | `init` 时扫语料生成，之后**冻结** | `init` 时随机初始化，`train` 时不断被更新 |
| 存在哪 | `<model>.vocab`（本 demo 写的纯文本） | `<model>`（`.param` 二进制权重的一部分） |
| 训练时 | 一动不动，只用来编码/解码 | 参与前向、反向传播、Adam 更新 |

### 词表怎么生成（`init` 阶段，只做一次）

`init` 会根据 `--tokenizer` 选择不同词表生成方式。

字符级默认走 `CharacterTokenizer::BuildVocabularyFromText`（`character_tokenizer.cpp:5`）扫一遍语料，逻辑非常朴素——见过的字符跳过，没见过的就分配下一个编号：

- **字节级**：粒度就是单个 `char`，不做 BPE、不切词。`hello` = `h/e/l/o` 四个 token（`l` 去重）。
- **按首次出现顺序编号**：语料里第一个出现的字符拿 id 0，第二个新字符拿 id 1…… 所以 `info` 打出来的词表顺序，就是语料开头的字符顺序。
- **去重、无特殊 token**：没有 `<pad>`/`<unk>`/`<bos>`/`<eos>`，词表里只有语料真实出现过的字符。
- **大小 = distinct 字符数**：它同时等于 embedding 的行数和输出 softmax 的维度，属于模型结构，所以 `init` 后**冻结**（词表外的字符在 `train`/`generate` 时被 `FilterToVocab` 丢弃并告警）。

一个副作用：编号跟“语料开头长什么样”绑定，**相同字符集、不同语料开头 → 不同的 id 分配**。所以两个模型的 `.vocab` 不能互换，`LoadModel` 里 `vocab_size` 那道校验就是为此。

词级模式走 `WordTokenizer::InitFromText(text, config)`，其中 `config` 统一描述 `<unk>`、词表上限等词级 tokenizer 构建策略：

- **按英文单词切分**：连续的字母、数字、下划线算一个词，逗号、句号、引号、换行等都只是分隔符。
- **统一小写**：`Alice` 和 `alice` 是同一个 token，生成时也会输出小写词。
- **可限制高频词表**：`--max-vocab-size N` 只保留出现次数最高的 N 个普通词，并用首次出现顺序打破同频词排序；默认 `0` 表示保留全部词。
- **编号规则**：默认 `map` 策略下 `<unk>` 固定为 id 0；不限制词表时，普通词按语料里第一次出现的顺序编号；限制词表时，普通词按频率降序编号。
- **词表外单词处理**：`--unknown-policy map`（默认）把没见过的词映射到 `<unk>` 并告警，兼容旧词级模型；`--unknown-policy drop` 直接跳过词表外词，生成时不会吐 `<unk>`，但会让少量长尾词从上下文中消失。
- **解码用空格拼词**：词级生成不会恢复原始标点和大小写，输出形如 `alice was beginning to get very tired`。

### 词向量怎么训练（`train` 阶段，每步都在变）

真正被“训练”的是每个 token 对应的那行向量：

1. **随机初始化**：`TokenEmbedding::Init`（`token_embedding.cpp:8`）用 Xavier-uniform 在 `±sqrt(6/(vocab+dim))` 内随机填出 `vocab × model-dim` 的表。此刻向量毫无意义。
2. **前向查表**：`Encode` 拿 token id 当行号查表，取出那一行当作这个字符的输入表示。
3. **反向更新**：`TrainNextToken`（`mini_transformer_lm.cpp:539`）把该位置回传的梯度 `grad_hidden` 按 token id **散射累加**回对应的 embedding 行（同一字符在窗口里出现多次就累加多次），再对整张表走一步 Adam（`embedding_optimizer_.Apply`）。
4. 训练久了，**经常出现在相似上下文里的字符，向量会越来越接近**——这就是“学到”的字符表示。

所以整条链是：**词表把字符/单词变成编号（不训练）→ 词向量把编号变成向量（这才是训练对象）→ Transformer 在向量上做预测。** 词表存在 `.vocab` 里、词向量存在 `.param` 里，也正对应这个分工。

## 5. 命令与参数

### `init` — 创建基座模型

从语料扫词表 + 按结构超参随机初始化 + 保存。

| 参数 | 含义 | 默认 |
|------|------|------|
| `--model <path>` | 输出模型路径 | `mini_lm.param` |
| `--tokenizer <char\|word>` | tokenizer 粒度：字符级或词级 | `char` |
| `--max-vocab-size <int>` | 词级模式保留 top-N 高频普通词；`0` 表示全量词表 | `0` |
| `--unknown-policy <map\|drop>` | 词级 OOV 策略：映射到 `<unk>` 或直接丢弃 | `map` |
| `--corpus <text>` | 内联文本，用来建词表 | 一段占位串 |
| `--corpus-file <path>` | 从文件读建词表的语料 | — |
| `--corpus-dir <dir>` | 从目录下所有文件读语料 | — |
| `--backbone <encoder\|decoder>` | 主干类型 | `decoder` |
| `--model-dim <int>` | 隐藏维度（成本按平方增长，最敏感） | 6 |
| `--head-num <int>` | 注意力头数，**必须整除 model-dim** | 1 |
| `--feed-forward-dim <int>` | 前馈网络中间维度 | 12 |
| `--block-num <int>` | **堆叠的 Transformer 层数** | 2 |
| `--context-size <int>` | 训练窗口 / 最大上下文长度 | 2 |
| `--rand-seed <int>` | 随机种子（可复现） | 0 |

`--corpus-dir` / `--corpus-file` / `--corpus` 三者按此优先级取第一个提供的。

### `train` — 训练 / 续训

加载已有模型，喂语料训练，训完把权重写回原 `--model` 文件（词表不变）。

| 参数 | 含义 | 默认 |
|------|------|------|
| `--model <path>` | 要加载并更新的模型 | `mini_lm.param` |
| `--corpus <text>` / `--corpus-file <path>` / `--corpus-dir <dir>` | 训练语料来源 | 内联占位串 |
| `--epochs <int>` | 训练轮数 | 800 |
| `--learning-rate <double>` | 基础学习率（内部套 `WarmupCosineLR`） | 0.01 |
| `--log-every <int>` | 每 N 个 epoch 打印一次汇总日志 | 1 |
| `--progress-every-sec <int>` | 单个 epoch 内每 N 秒打印进度（0=关闭） | 5 |
| `--early-stop-loss <double>` | epoch loss 低于该值后提前停止（<=0=关闭） | 0.02 |
| `--checkpoint <path>` | checkpoint 文件路径 | `<model>.ckpt` |
| `--checkpoint-every <int>` | 每 N 个完整 epoch 保存 checkpoint | 1 |
| `--resume-checkpoint` | 从 checkpoint 恢复并继续训到 `--epochs` 指定的总轮数 | 关闭 |
| `--no-checkpoint` | 关闭周期性 checkpoint | 关闭 |

训练用逐位置 next-token 交叉熵 + Adam，学习率走线性 warmup + cosine 退火（与仓库其他 demo 同款配方）。默认每个 epoch 都会打印 loss / perplexity / lr / elapsed / eta，并保存一份 checkpoint，避免长时间训练完全黑盒。

checkpoint 由三份 sidecar 组成：

```text
<model>.ckpt        # 权重
<model>.ckpt.vocab  # 词表 / tokenizer 元数据
<model>.ckpt.train  # 训练进度元数据(completed_epoch、last_loss 等)
```

中断后继续训练：

```bash
./bin/mini_lm train --model /tmp/alice.param \
  --checkpoint /tmp/alice.param.ckpt \
  --resume-checkpoint \
  --corpus-file /tmp/alice.txt \
  --epochs 200 --learning-rate 0.002
```

`--epochs` 在恢复时表示“目标总 epoch 数”，不是“再额外训多少轮”。例如 `.train` 里 `completed_epoch=73`，命令传 `--epochs 200`，就会从 checkpoint 权重继续训到第 200 轮。Ctrl+C / SIGTERM 会在当前样本更新后尽量保存 checkpoint 并退出；如果刚好中断在 epoch 中途，元数据仍只记录已完整完成的 epoch，恢复时会从这个 epoch 重新开始，但权重已经包含中途学到的参数。

注意：当前 checkpoint 保存的是模型权重、词表和训练进度，**不保存 Adam 的动量 / 二阶矩状态**。恢复后不会从零开始学，已训练出的权重都在；但优化器状态会冷启动，这对教学 demo 可接受，追求严格训练复现时需要扩展模型序列化格式。

### `generate` — 推理生成

| 参数 | 含义 | 默认 |
|------|------|------|
| `--model <path>` | 要加载的模型 | `mini_lm.param` |
| `--prompt <text>` | 起始文本 | `ab` |
| `--generate-num <int>` | 续写多少个 token（字符级=字符数，词级=词数） | 20 |
| `--temperature <double>` | 采样温度（**给了就切采样模式**） | — |
| `--top-k <int>` | top-k 采样 | — |
| `--top-p <double>` | top-p (nucleus) 采样 | — |

**不给采样参数 = 贪心**（每步选概率最高的字符，确定性输出，但容易重复）。给了 `--temperature`/`--top-k`/`--top-p` 中任意一个就切到**采样模式**（更多样，能跳出重复）。

### `info` — 查看模型

打印结构超参、词表大小与词表内容。`--model <path>`（默认 `mini_lm.param`）。

## 6. 准备训练数据（清洗 → 喂给模型）

字符级模型对"字符集大小"很敏感：语料里每多一个稀有字符（花引号、emoji、非拉丁字母……）词表就大一圈，小模型更难学。词级模型对"词表大小"敏感：低频词越多，embedding 和输出层越大，小模型越难训。所以**喂给模型前先清洗成干净的 ASCII 文本**，必要时用 `--max-vocab-size` 只保留高频词。

`src/demo/mini_lm/tools/` 下提供两个只依赖标准库的脚本（都已纳入 git，可直接改）：

### 6.1 一键获取公版数据：`fetch_data.sh`

从 [Project Gutenberg](https://www.gutenberg.org/)（公有领域）下载整本书并自动清洗：

```bash
cd src/demo/mini_lm
./tools/fetch_data.sh                 # 默认下载《爱丽丝梦游仙境》到 ./tools/data/
./tools/fetch_data.sh --book sherlock # 换一本《福尔摩斯》
./tools/fetch_data.sh --book all      # 全部下载
```

产物（**不入库**，见 [§11](#11-训练数据与模型产物不入库)）：

- `tools/data/<name>.raw.txt`：下载的原始文本
- `tools/data/<name>.txt`：清洗后的训练语料（喂这个）

### 6.2 清洗任意文本：`clean_text.py`

`fetch_data.sh` 内部调用它，你也可以拿它清洗**自己的**文本：

```bash
# 清洗一个文件
python3 tools/clean_text.py -i my_raw.txt -o my_corpus.txt

# 从管道清洗
cat a.txt b.txt | python3 tools/clean_text.py > my_corpus.txt
```

它做四件事：① 去掉 Project Gutenberg 的页眉页脚标记；② 把常见 Unicode 标点（花引号 `“”‘’`、破折号 `——`、省略号 `…`）归一化成 ASCII；③ 默认丢弃剩余的非 ASCII 字节；④ 合并 3 行以上空行、去首尾空白。

常用开关：`--keep-gutenberg-markers`（保留页眉页脚）、`--keep-non-ascii`（保留非 ASCII，比如你要训中文时）。

### 6.3 用你自己的数据 / 一个目录

- **单个文件**：`--corpus-file path/to/corpus.txt`
- **整个目录**：`--corpus-dir path/to/dir/`。会**递归**遍历目录下所有常规文件、按路径排序后拼接（排序保证可复现），适合把很多小文本喂进去。
- **内联小文本**：`--corpus "..."`，适合快速试。

## 7. 预训练与后训练

这个 CLI 的三段式正好对应大模型的工作流，只是规模缩到能在你机器上跑：

### 预训练（pretrain）

在**大而杂**的通用语料上，从随机初始化开始训，让模型学会语言的基本统计规律（哪些字符常连在一起、单词长什么样）。对应命令：

```bash
cd src

# 1) 用大语料 init，词表覆盖面要广
./bin/mini_lm init --model /tmp/base.param \
    --corpus-file demo/mini_lm/tools/data/alice.txt \
    --model-dim 32 --head-num 2 --feed-forward-dim 128 --block-num 2 --context-size 32

# 2) 在同一份大语料上训足够多轮（这一步最耗时，建议放云上）
./bin/mini_lm train --model /tmp/base.param \
    --corpus-file demo/mini_lm/tools/data/alice.txt \
    --epochs 60 --learning-rate 0.002
```

训完的 `base.param` + `base.param.vocab` 就是你的**基座**。

### 后训练 / 微调（post-train / fine-tune）

**从预训练好的基座继续训**，喂一份更小、更专的语料（某种写作风格、某个领域的文本），让它在保留通用能力的基础上偏向新数据。因为 `train` 本来就是"加载已有模型 → 继续训 → 写回"，所以后训练就是**对着基座再跑一次 `train`，换成新语料、用更小的学习率**：

```bash
# 复制一份基座，避免覆盖（连 .vocab 一起复制！）
cp /tmp/base.param /tmp/tuned.param
cp /tmp/base.param.vocab /tmp/tuned.param.vocab

# 在新语料上小学习率微调
./bin/mini_lm train --model /tmp/tuned.param \
    --corpus-file my_style_corpus.txt \
    --epochs 15 --learning-rate 0.0005
```

要点：
- 字符级微调语料的字符必须落在基座 `init` 时定的词表内，否则会被跳过（有告警）；词级微调语料里的新词会按模型初始化时保存的 unknown policy 映射到 `<unk>` 或被丢弃。想真正支持新字符/新词，只能重新 `init` + 重新预训练。
- 微调学习率通常**比预训练小一个量级**（例：预训练 `0.002` → 微调 `0.0005`），避免把基座学到的东西冲掉。
- 这是**风格/领域适配**，不是"教它对话"。无论字符级还是词级，只靠普通续写语料都做不到"你问它答"。

## 8. 真实训练示例（诚实的结果）

下面这组数是在本机（Apple Silicon、单线程 Debug 构建）真实跑出来的，供你对参照：

```bash
cd src
head -c 4000 demo/mini_lm/tools/data/alice.txt > /tmp/alice4k.txt   # 取《爱丽丝》前 4000 字符
./bin/mini_lm init  --model /tmp/alice.param --corpus-file /tmp/alice4k.txt \
    --model-dim 32 --head-num 2 --feed-forward-dim 128 --block-num 1 --context-size 16 --rand-seed 42
./bin/mini_lm train --model /tmp/alice.param --corpus-file /tmp/alice4k.txt \
    --epochs 12 --learning-rate 0.002
```

loss 从 2.59 稳步降到 2.16（最终评估 loss 2.29 / perplexity 9.87），约 5 分钟。生成效果：

```text
# 贪心（确定性，但会退化成重复）
$ ./bin/mini_lm generate --model /tmp/alice.param --prompt "Alice " --generate-num 80
Alice the the the the the the ...

# 采样（temperature 0.6 + top-k 5，更像语言）
$ ./bin/mini_lm generate --model /tmp/alice.param --prompt "The " --generate-num 80 --temperature 0.6 --top-k 5
The wat tout ald s, tiledofoudeve ththerasang s s t tude o thed oud t t se tuthe wan
```

只训了 12 轮 4000 字符，模型就已经学会了**空格分词、常见词 `the`、以英文字母簇为主的片段**——这正是"字符级语言规律"的雏形，和 `abcabc` 完全不同。想让它续写出更多**成词、成句**的文本，就往这三个方向加：

- **更多轮**（`--epochs` 提到 60~200）
- **更多数据**（用整本 `alice.txt`，甚至 `--book all`）
- **更大模型**（`--model-dim 64`、`--block-num 2`、`--context-size 32~64`）

这三项都会显著拉长训练时间，适合放云上跑。

## 9. 典型输出

```text
$ ./bin/mini_lm info --model /tmp/alice.param
Model: /tmp/alice.param
Backbone: decoder model_dim: 32 head_num: 2 feed_forward_dim: 128 block_num: 1 context_size: 16 rand_seed: 42
Tokenizer: char
Vocab size: 65
Vocabulary: [I][l][u][s][t][r][a][i][o][n][ ][A][c][e]['][d][v][W]...
```

词级模型的 `info` 会显示：

```text
Tokenizer: word
Vocab size: 8
Vocabulary: [<unk>] [alice] [was] [beginning] [to] [get] [very] [tired]
```

## 10. 训练成本与选参

这是**单线程、无 batch** 的朴素实现，训练时间 ≈ `样本数 × epoch数 × 每样本耗时`，其中每样本耗时对 `model-dim` 和 `context-size` 很敏感（本机实测）：

| 配置 | 每样本耗时 |
|------|-----------|
| `model-dim 32 / ffn 128 / block 1 / context 16` | ~6 ms |
| `model-dim 64 / ffn 128 / block 2 / context 32` | ~74 ms |

`model-dim` 是平方级成本，`context-size`、`block-num` 也都直接拉高耗时。选参建议：

- **先小后大**：先用小配置（`d=32`）验证语料和流程跑通，再逐步加大。
- **样本数 ≈ 语料字符数**，`--corpus-dir` 喂大目录时留意总量。
- 想估算总时长：先 `--epochs 1` 跑一轮看 wall-clock，再乘目标轮数。
- 训练很久时保留默认 checkpoint；如果只是跑极小 smoke test，才考虑 `--no-checkpoint`。

## 11. 训练数据与模型产物不入库

`src/demo/mini_lm/.gitignore` 约定：**只有源码和 `tools/` 脚本进 git，训练数据和模型产物一律不进**。被忽略的有：

- `tools/data/`（`fetch_data.sh` 下载 + 清洗出来的语料）
- `*.param` / `*.vocab`（`init`/`train` 产出的模型）
- `*.ckpt` / `*.train`（训练 checkpoint 和进度元数据）

这样仓库保持干净，别人拉下来用 `tools/fetch_data.sh` 就能自己重新拿数据、重新训。

## 12. 代码对应

| 概念 | 源码 |
|------|------|
| 命令调度 + IO（本 demo） | `src/demo/mini_lm/main.cpp` |
| 数据获取 / 清洗脚本 | `src/demo/mini_lm/tools/fetch_data.sh` · `clean_text.py` |
| 模型主体（Transformer LM） | `src/deeplearning/transformer/mini_transformer_lm.*` |
| 权重存取 | `src/deeplearning/transformer/mini_transformer_lm_loader.*` |
| 字符/词级分词 / 数据集 | `character_tokenizer.*` · `embedding/word_tokenizer.*` · `character_dataset.*` |
| 学习率调度 | `src/deeplearning/lr_scheduler/warmup_cosine_lr.h` |

## 13. 常见问题

- **`Vocab sidecar not found`**：`.vocab` 文件丢了或没跟 `.param` 放一起。两个文件必须成对；模型要重新 `init`。
- **`model-dim must be divisible by head-num`**：`--model-dim` 必须能被 `--head-num` 整除（多头要均分维度）。
- **训练时 `skipped N characters outside the model vocabulary`**：字符级训练语料含 `init` 词表外的字符，已自动跳过。要支持这些字符得用覆盖它们的语料重新 `init`。
- **训练/生成时 `mapped N words outside the model vocabulary to <unk>`**：词级模型使用默认 `map` 策略，遇到没见过的词会映射到 `<unk>`。如果用了 `--max-vocab-size`，这通常是低频词被主动裁掉；想让模型真正学会这些词，需要增大词表或用覆盖它们的语料重新 `init`。
- **训练/生成时 `dropped N words outside the model vocabulary`**：词级模型使用 `drop` 策略，遇到没见过的词会直接跳过。这通常让生成更可读，但会损失少量长尾词上下文。
- **训练中断后怎么继续**：默认会写 `<model>.ckpt`，重新运行 `train` 时加 `--resume-checkpoint`，并把 `--epochs` 设为目标总轮数。
- **为什么恢复后 loss 有一点波动**：checkpoint 不保存 Adam 状态，恢复后优化器动量会冷启动；权重不会丢，但短期 loss 可能有小波动。
- **生成一直重复**：贪心的通病，改用采样（`--temperature 0.7 --top-k 5`），或训得更久/更大。
- **`Model vocab size does not match the vocab sidecar`**：`.param` 和 `.vocab` 来自不同模型，别混用。

## 相关章节

本 demo 的模型主体与电子书的 Transformer 部分同源，可配合阅读：

- [第 19 章 · Transformer 与自注意力](https://chenxuan520.github.io/deeplearning/chapter-19.html)（概念）
- [第 24 章 · 字符级 Transformer 实战](https://chenxuan520.github.io/deeplearning/chapter-24.html)（`transformer_char` 逐行精读；`mini_lm` 是在同一套模块上做的基座式 CLI 封装）

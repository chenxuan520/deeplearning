# AGENTS.md

## 仓库规则补充

- Git commit message 一律使用英文。
- 写 commit message 之前,必须先查看当前仓库最近的提交历史(至少 `git log --oneline -n 10`),严格沿用本仓库已有的格式与措辞风格(如 `docs:` / `docs(scope):` 等),不要自己发明不符合仓库历史的 subject。

本仓库是一个小型的 C++ 深度学习 / 神经网络学习项目。
当前仓库已经不是纯头文件库：`src/deeplearning/` 下同时包含头文件和 `.cpp` 实现，并通过 CMake 构建静态库。
除原有的 MLP / MNIST 路线外，仓库现在还包含最小 CNN / RNN / Transformer 实验链路。

在此仓库中未找到代理/编辑器规则文件（没有 `.cursor/rules/`，`.trae/rules/` 或 `.github/copilot-instructions.md`）。

## 1) 项目概述

### 项目简介
- 一个最小化的 C++ 神经网络实现，核心库位于 `src/deeplearning/` 下。
- 示例程序位于 `src/demo/` 下，目前包含：
  - `mnist`：MNIST 手写数字 10 类分类示例。默认 `784→128→64→10` / ReLU + He +
    Softmax + Cross-Entropy + Adam + `WarmupCosineLR`，batch=64，5 epochs，
    测试准确率约 97.8%。详见 `docs/mnist-demo.md`。
  - `cnn_mnist`：最小 CNN (`conv + ReLU + max-pool + linear`) 在 MNIST 子集上的训练示例
  - `rnn_char`：最小字符级 RNN 语言模型示例（BPTT 训练 / 生成 / perplexity）
  - `rl_tictactoe`：井字棋 Q-learning 示例（表格式 TD 学习 / ε-greedy / 对战随机或最优对手）
  - `word2vec`：skip-gram / CBOW + 负采样词向量训练，见 `docs/word2vec-demo.md`
  - `transformer_char`：最小字符级语言模型示例
  - `optimizer_bench`：不同 optimizer 在 MNIST 上的对比 benchmark
    （SGD / Momentum / Adam / AdamW / RMSProp / Adam+CosineLR）
- 测试位于 `src/test/` 下。

### 核心架构

**NeuralNetwork (神经网络)**
- 核心类型：`src/deeplearning/neural_network.h` 中的 `deeplearning::NeuralNetwork`。
- 表示一个全连接的前馈网络，由层大小向量定义（`std::vector<int> layer_`）。
- 参数存储为嵌套向量：
  - 偏置：`std::vector<std::vector<double>> neuron_bias_`
  - 权重：`std::vector<std::vector<std::vector<double>>> neuron_weight_`
  - 前向输出：`neuron_output_`
  - 反向传播 Delta：`neuron_delta_`

**通过工厂模式的可插拔策略**
网络组合了几个策略接口，每个接口都通过枚举 + 工厂进行选择：
- 激活函数：`ActivateType` / `ActivateFactory` (`src/deeplearning/activate/`)
  - `ACTIVATE_SIGMOID` / `ACTIVATE_RELU` / `ACTIVATE_TANH`
  - `ACTIVATE_LEAKY_RELU` (默认 slope=0.01)
  - `ACTIVATE_GELU` (erf 精确版本; Transformer/BERT 类常用)
- 损失函数：`LossType` / `LossFactory` (`src/deeplearning/loss/`)
- Softmax：`SoftmaxType` / `SoftmaxFactory` (`src/deeplearning/softmax/`)
- 参数初始化：`ParamInitType` / `ParamInitFactory` (`src/deeplearning/param_init/`)
  - 全部 `*_RANDOM` / `*_XAVIER` / `*_HE` 支持 `set_seed(int)`, NN `set_random_seed`
    会自动透传 seed, 保证可复现.
- 优化器：`OptimizerType` / `OptimizerFactory` (`src/deeplearning/optimizer/`)
  - `OPTIMIZER_SGD` / `OPTIMIZER_MOMENTUM`
  - `OPTIMIZER_RMSPROP` (Hinton)
  - `OPTIMIZER_ADAM` (Kingma & Ba, 含 bias correction)
  - `OPTIMIZER_ADAMW` (Loshchilov & Hutter, decoupled weight decay)
  - 所有优化器都支持 `set_weight_decay(double)`; Adam/AdamW 区别在于 wd 与
    adaptive step 的耦合方式.
- 学习率调度: `LRScheduler` (`src/deeplearning/lr_scheduler/`)
  - `StepDecayLR` / `ExponentialDecayLR` / `CosineAnnealingLR` / `WarmupCosineLR`
  - 通过 `network.set_lr_scheduler(...)`, Train 会在每个 minibatch step 之前
    调 `scheduler->GetLR(step)` 并更新 `learning_rate_`.

`NeuralNetwork::Init()` 设置默认值（例如 `SOFTMAX_NONE`, `LOSS_MSE`, `ACTIVATE_SIGMOID`, `PARAM_INIT_ZERO`, `OPTIMIZER_SGD`）并初始化参数。

**训练流程**
- `Train(...)` 打乱数据索引，按 `batch_num` 大小切 minibatch:
  1. (如果有 scheduler) 用 `scheduler->GetLR(step)` 更新 lr
  2. `ForwardPropagationBatch(batch)`: 同时记录 pre-activation (供 GELU 等用)
     和 post-activation
  3. `BackPropagationBatch(batch)`: 累加梯度到 `grad_bias_` / `grad_weight_`
  4. (可选) `ClipGradients(B)`: 按 norm 或按 value 裁剪平均梯度
  5. `ApplyGradient(B)`: 先调 `optimizer->BeforeStep()` (Adam 推进 t),
     再对每个参数 `optimizer->CalcChangeValue(avg_grad, lr, pos, weight_pos, param_value)`
     并把结果从参数里扣除.
- 可选的每轮回调：`each_epoch_call(NeuralNetwork&, int epoch_num, bool& early_stop)`。

**梯度裁剪**
- `set_gradient_clip_norm(double)`: 全局 L2 范数裁剪 (按比例缩放)
- `set_gradient_clip_value(double)`: 逐分量绝对值裁剪
- 默认关闭 (设 ≤ 0). 两者可同时开, 先 by-value, 再 by-norm.

**模型序列化**
- `src/deeplearning/neural_network_loader.h` 中的 `deeplearning::NeuralNetworkLoader` 提供模型参数的二进制导出/导入。
- MNIST 演示使用它来缓存/加载 `demo.v2.param` (`src/demo/mnist/main.cpp`)。
  历史 v1 模型存为 `demo.param`（已不再被本 demo 读写，仅作历史参考）。

**Transformer / 字符级语言模型**
- `src/deeplearning/transformer/` 下提供最小 Transformer 相关模块：
  - `TokenEmbedding`
  - `PositionalEncoding`
  - `LayerNorm`
  - `SelfAttention`
  - `TransformerBlock`
  - `TransformerEncoder`
  - `TransformerDecoder`
  - `MiniTransformerLM`
  - `MiniTransformerLMLoader`
  - `CharacterTokenizer`
  - `CharacterDataset`
- `MiniTransformerLM` 目前支持：
  - encoder / decoder 两种主干
  - `block_num == 0/1/2` 的训练和生成测试路径
  - greedy 生成与 `temperature/top-k/top-p` 采样生成
  - next-token loss / perplexity 评估
  - 模型保存/加载
  - `MiniTransformerLM::Config` 统一描述模型结构与主干配置
  - demo 可导出真实 attention 权重 JSON 供电子书第 19 章「真实权重回放」观察

**CNN**
- `src/deeplearning/cnn/` 下提供最小 CNN 相关模块：
  - `Conv2D`
  - `MaxPool2D`
  - `MiniCNNClassifier`
- `MiniCNNClassifier` 当前支持：
  - 单层卷积 + ReLU + 最大池化 + 线性分类头
  - 单样本 SGD 训练与 softmax 预测
  - 适合 MNIST / 小型二维图像教学示例

**RNN**
- `src/deeplearning/rnn/` 下提供最小 RNN 相关模块：
  - `SimpleRNN`
  - `MiniRNNLM`
- `MiniRNNLM` 当前支持：
  - tanh RNN 按时间展开前向
  - BPTT 训练
  - next-token loss / perplexity / greedy 生成
  - 可选梯度裁剪与 LR scheduler

**强化学习 (RL)**
- `src/deeplearning/rl/` 下提供最小表格式强化学习模块：
  - `TicTacToeEnv`：井字棋环境（状态编码、合法动作、随机/最优对手、渲染）
  - `TabularQLearning`：Q-learning（ε-greedy、TD 更新、训练回合、评估胜率）
- `rl_tictactoe` demo 默认训练 30000 局 vs 随机对手，greedy 评估约 **99%+ 胜率**（对随机）且对最优对手可稳守和棋。

**可选绘图**
- `src/drawtool/matplot_draw.h` 中的 `drawtool::MatplotDraw` 在使用 `_MATPLOTLIB_CPP_LOAD_` 编译时绘制损失曲线。
- 该宏由 CMake 选项 `ENABLE_DRAW` 控制（见 `src/CMakeLists.txt`）。

## 2) 构建与命令

### 构建（CMake 项目根目录是 `src/`）
顶级 CMakeLists 是 `src/CMakeLists.txt`。

**通过脚本构建（仓库推荐）**
- 从 `src/` 目录：
  - `./build.sh [ENABLE_DRAW] [BUILD_TYPE]`
  - `ENABLE_DRAW` 作为 `-DENABLE_DRAW=...` 传递给 CMake（如果省略，默认为 `false`）。
  - `BUILD_TYPE` 如果提供，将变为 `-DCMAKE_BUILD_TYPE=<value>`（如果省略，则为空）。
  - 脚本创建/使用 `src/build/` 并运行 `cmake ..` 然后 `make`。

**手动构建**
- 从 `src/` 目录：
  - `mkdir -p build && cd build && cmake .. && make`

### 输出
- 可执行文件配置为通过子项目中的 `EXECUTABLE_OUTPUT_PATH` 放置在 `src/bin/` 下：
  - 测试：`src/bin/test_bin`（来自 `src/test/CMakeLists.txt`）
  - MNIST 演示：`src/bin/mnist`（来自 `src/demo/mnist/CMakeLists.txt`）
  - CNN on MNIST 演示：`src/bin/cnn_mnist`（来自 `src/demo/cnn_mnist/CMakeLists.txt`）
  - 字符级 Transformer 演示：`src/bin/transformer_char`（来自 `src/demo/transformer_char/CMakeLists.txt`）
  - 字符级 RNN 演示：`src/bin/rnn_char`（来自 `src/demo/rnn_char/CMakeLists.txt`）
  - 优化器对比 benchmark: `src/bin/optimizer_bench`（来自 `src/demo/optimizer_bench/CMakeLists.txt`）
  - 井字棋 Q-learning：`src/bin/rl_tictactoe`（来自 `src/demo/rl_tictactoe/CMakeLists.txt`）
  - word2vec 词向量训练：`src/bin/word2vec`（来自 `src/demo/word2vec/CMakeLists.txt`）

### 运行

**MNIST 演示**
- 在工作目录 `src/` 下运行（`src/demo/mnist/main.cpp` 中的路径是相对的，如 `./demo/mnist/mnist/...`）：
  - `./bin/mnist`
- 输入像素由 `mnist_data.h` 内部归一化到 `[0,1]`（不是二值化），`optimizer_bench`
  也共用同一份预处理。
- 训练结束会把模型保存到 `./demo/mnist/mnist/demo.v2.param`；下一次运行检测到该
  文件会自动加载续训（`lr` 自动降到 `1e-4` 微调）。想冷启动直接 `rm` 它。
- 详细配置与升级历程见 `docs/mnist-demo.md`。

**Transformer 字符级演示**
- 在工作目录 `src/` 下运行：
  - `./bin/transformer_char`
- 常用参数：
  - `--prompt`
  - `--generate-num`
  - `--temperature`
  - `--top-k`
  - `--top-p`
  - `--epochs`
  - `--learning-rate`
  - `--rand-seed`
  - `--backbone encoder|decoder`
  - `--model-dim`
  - `--head-num`
  - `--feed-forward-dim`
  - `--block-num`
  - `--context-size`
  - `--block-learning-rate-scale`
  - `--model-file`
  - `--config-file`
  - `--attention-export-file`
  - `--corpus`
  - `--corpus-file`
  - `--save-model`
  - `--no-save-model`
  - `--eval-only`
  - `--force-train`

**CNN on MNIST 演示**
- 在工作目录 `src/` 下运行：
  - `./bin/cnn_mnist`
- 常用参数：
  - `--epochs`
  - `--learning-rate`
  - `--train-limit`
  - `--test-limit`
  - `--rand-seed`
  - `--conv-channels`
  - `--kernel-size`

**RNN 字符级演示**
- 在工作目录 `src/` 下运行：
  - `./bin/rnn_char`
- 常用参数：
  - `--prompt`
  - `--generate-num`
  - `--epochs`
  - `--learning-rate`
  - `--rand-seed`
  - `--hidden-dim`
  - `--context-size`
  - `--gradient-clip-norm`
  - `--corpus`
  - `--corpus-file`

**井字棋 Q-learning 演示**
- 在工作目录 `src/` 下运行：
  - `./bin/rl_tictactoe`
  - `./bin/word2vec`（`--query-word cat --compare-word dog` 等，见 `docs/word2vec-demo.md`）
- 常用参数：
  - `--episodes`
  - `--eval-games`
  - `--rand-seed`
  - `--alpha`
  - `--gamma`
  - `--epsilon`
  - `--epsilon-min`
  - `--epsilon-decay`
  - `--opponent random|optimal`
  - `--show-sample`
  - `--play`（stdin 人机对战，你是 O）
- 详见 `docs/rl-tictactoe-demo.md`。

**word2vec 演示**
- 在工作目录 `src/` 下运行：
  - `./bin/word2vec`
- 常用参数：
  - `--corpus-file`
  - `--mode skip-gram|cbow`
  - `--embed-dim`
  - `--window-size`
  - `--negative-num`
  - `--epochs`
  - `--query-word`
  - `--compare-word`
  - `--show-pairs`
- 详见 `docs/word2vec-demo.md`。

**测试**
- 在工作目录 `src/` 下运行：
  - `./bin/test_bin`
- 可选的单个参数启用测试/基准名称的正则过滤：
  - `./bin/test_bin <regex>`
  - 过滤由 `src/test/main.cpp` 中的 `REGEX_FILT_TEST(argv[1])` 实现。

### 安装
- 安装由 CMake 定义：
  - `install(TARGETS deeplearning DESTINATION lib)`
  - `install(DIRECTORY ./deeplearning DESTINATION include FILES_MATCHING PATTERN "*.h")`
- 仓库 README 展示了使用 `make install` 的安装流程：
  - `mkdir build; cmake ..; sudo make install`（从包含 CMakeLists 的目录运行：`src/`）。

## 3) 代码风格

### 本仓库可见的约定
- 头文件使用 `#pragma once`。
- 命名空间组织：
  - 核心库：`namespace deeplearning { ... }`
  - 绘图工具：`namespace drawtool { ... }`
  - 测试框架：`namespace cpptest { ... }` (`src/third_party/cpptest/test.h`)
- 命名：
  - 类型/类：`PascalCase`（帕斯卡命名法，例如 `NeuralNetwork`, `SoftmaxFactory`）。
  - 方法：`PascalCase`（例如 `Train`, `Predict`, `CalcLoss`）。
  - 成员字段：尾随下划线（例如 `learning_rate_`, `neuron_weight_`）。
- 错误处理：
  - 许多操作返回 `RC` 枚举并设置字符串错误消息（`err_msg_`），例如 `NeuralNetwork::Train()`。
  - 调用者通常检查返回代码并可能打印 `err_msg()`。

### 扩展模式
- 新的激活函数/损失函数/优化器等遵循现有模式：
  - 实现相关的基础接口。
  - 添加一个枚举值。
  - 在相应的 `*Factory::Create(...)` switch 中注册它。
- 新优化器额外注意:
  - 如果需要 step 计数 (Adam 风格), 重载 `BeforeStep()`.
  - 如果支持 weight decay, 在 `CalcChangeValue` 里根据 `weight_pos != -1` 判定
    是否是 weight (只对 weight 衰减, bias 不动), 并按 Adam (L2-in-grad) 或
    AdamW (decoupled) 选择耦合方式.
- 新激活函数:
  - 简单激活实现 `Activate` + `DerivActivate(output)` 即可.
  - 需要 pre-activation 的复杂激活 (如 GELU) 额外重载
    `DerivActivate(input, output)`, 默认会回退到 `DerivActivate(output)`.

## 4) 测试

### 框架
- 使用捆绑在 `src/third_party/cpptest/test.h` 中的基于宏的框架。
- 本仓库使用的关键宏：
  - `TEST(group, name)` 定义测试。
  - `INIT(name)` / `END(name)` 通过静态生命周期定义全局设置/拆卸。
  - `MUST_EQUAL(...)`, `MUST_TRUE(...)` 用于断言。
  - `DEFER(...)` 用于清理。

### 仓库约定
- 测试主要是 `src/test/` 下的头文件，被 `src/test/main.cpp` 包含。
- 运行子集：
  - 传递一个 CLI 参数，它被视为 `std::regex` 与 `"<Group> <TestName>"` 匹配。
- 一些测试在工作目录中创建和删除临时文件（例如 `demo.param`, `demo.data`, `demo.test`）。
- `NeuralNetwork` 测试不再调用绘图弹窗；图像绘制保留在 demo 路径中。

## 5) 安全性

### 模型文件安全 (`*.param`)
- `NeuralNetworkLoader` 通过以下方式读取/写入二进制格式：
  - `NetworkOption` 作为原始字节 (`ofs.write((const char *)&option, sizeof(option))`)，
  - 然后是大小 (`ParamSizeMsg`)，
  - 然后是层/偏置/权重的原始 int/double (`src/deeplearning/neural_network_loader.h`)。
- 将 `*.param` 文件视为不可信输入：
  - 精心制作的文件可能通过大小字段强制进行非常大的分配，可能耗尽内存。
  - 原始结构体写入使得格式对编译器/ABI 差异敏感。

### 可选的 Python 链接
- 当 `ENABLE_DRAW=ON` 时，CMake 定位并链接 Python (`src/CMakeLists.txt` 中的 `find_package(PythonInterp REQUIRED)`, `find_package(PythonLibs REQUIRED)`)。
- 请记住，这会向构建中添加额外的原生依赖项和头文件。

## 6) 配置

### CMake 选项
- `ENABLE_DRAW`（默认为 `OFF`）在 `src/CMakeLists.txt` 中：
  - 添加包含路径 `src/third_party/matplotlib-cpp`。
  - 定义 `_MATPLOTLIB_CPP_LOAD_` 和 `WITHOUT_NUMPY`。
  - 查找 Python include/libs 并链接它们。

### 构建类型和工具
- `src/CMakeLists.txt` 将 `CMAKE_BUILD_TYPE` 设置为 `"Debug"`。
- `src/build.sh` 可选地传递 `-DCMAKE_BUILD_TYPE=<value>`。
- **clangd / LSP**: `src/CMakeLists.txt` 开启 `CMAKE_EXPORT_COMPILE_COMMANDS`；`./build.sh` 或 `cmake ..` 后会在 `src/build/compile_commands.json` 生成编译数据库，并复制到 `src/compile_commands.json` 与仓库根目录（均在 `.gitignore`，需本地生成）。根目录与 `src/.clangd` 已指向 `src/build` 作为 `CompilationDatabase`。

### Sanitizers (消毒剂)
- `src/CMakeLists.txt` 包含注释掉的 Address/Leak/UB sanitizers 标志。

## 7) 交互式电子书 (`docs/dl-book/`)

配套深度学习入门电子书，静态站点位于 `docs/dl-book/`，共 **30 章**(第 0–25 章正文/代码实战 + 第 26–29 章「第七部分 · 番外」)+ 术语表，含交互示意图与实验台。番外四章分别是:第 26 章「机器学习全景图」、第 27 章「无监督与自监督学习」、第 28 章「经典机器学习:决策树与随机森林」、第 29 章「从沙子到 CPU/GPU」。
作者、来源、转载要求、电子书内容许可与代码 MIT License 说明见 `docs/dl-book/about.html`；根目录 `LICENSE` 是项目代码许可，`src/third_party/` 下第三方依赖遵循各自许可证。

### 链接
- **在线阅读**: https://deeplearning.011203.xyz/
- **GitHub Pages 备用地址**: https://chenxuan520.github.io/deeplearning/
- **GitHub 仓库**: https://github.com/chenxuan520/deeplearning
- **Gitee 仓库** (`origin`): https://gitee.com/chenxuan520/deeplearning

### 双远程与线上部署（重要）
本仓库配置了两个 git remote：
- `origin` → Gitee
- `github` → GitHub（**GitHub Pages 的部署来源**）

**修改 `docs/dl-book/` 后要让线上站点更新，必须推送到 `github` 远程。** 只推 `origin` 不会触发 Pages 部署：

```bash
git push github master
# 或两个远程一起推
git push origin master && git push github master
```

自动部署工作流：`.github/workflows/dl-book-pages.yml`  
触发条件：`master` 分支上 `docs/dl-book/**` 或该 workflow 文件有变更。

### 本地预览
```bash
cd docs/dl-book && python3 -m http.server 8765
# 浏览器打开 http://localhost:8765/index.html
```

### 跨章链接约定
书中提到其他章节的具体内容时，应在**文字上直接链到对应小节标题**，不要只写“第 N 章”：

```html
<a class="xref" href="chapter-01.html#5-导数-斜率告诉你往哪走会变大">导数</a>
```

小节锚点 id 由 `docs/dl-book/assets/book.js` 里的 `slugify()` 根据 h2/h3 标题自动生成（小写、冒号变空格、空白变连字符、去掉标点）。新增或改标题后需重新核对锚点是否仍正确。

### 校验脚本（改书后务必先跑，别靠肉眼）
`docs/dl-book/tools/` 下有一套只依赖 Python3 标准库的自查脚本，改完章节 HTML **务必在 `docs/dl-book/` 目录下**跑一遍再提交；CI(`.github/workflows/dl-book-check.yml`)也会跑前两个：

```bash
cd docs/dl-book
python3 tools/check_structure.py   # HTML 标签配平 + 每章 h2 主编号是否 1,2,3… 连续
python3 tools/check_anchors.py     # 所有跨章/跨节 #锚点 是否指向存在的 h2/h3(输出“失效: 0”才算过)
python3 tools/count_chars.py -v    # (可选) 统计篇幅变化
```

**不要再手算 slug 来核对锚点**——`check_anchors.py` 已经复刻了 `book.js` 的 `slugify()` 规则,直接跑它即可。改标题、加小节、顺延编号之后,这两个脚本任一非 0 就说明有链接断了或结构坏了。改 `assets/book.js` 后再跑 `node --check assets/book.js`。

### 改书注意事项（几处容易踩的坑）
- **h2 编号会顺延**:在某章中间插入带编号的新 h2 小节后,后面所有 h2 编号要跟着 +1,否则 `check_structure.py` 报“h2 编号不连续”。「读完这一章」「小结」「动手与思考」等无编号 h2 不计入。
- **改编号/标题前先查跨章引用**:别的章可能用 `chapter-XX.html#8-xxx` 这类**数字锚点**指过来,顺延编号会打断它们。动手前先 `grep -rn 'chapter-XX.html#' docs/dl-book/` 确认没有指向将被改动编号的引用。
- **slug 会保留圈号和汉字**:`①②③④` 属于 `\w`,不会被 `slugify()` 去掉,写锚点时别漏(例如 `#④-知识蒸馏-让小模型拜大模型为师`)。拿不准就用脚本现算:见 `tools/README.md`。
- **callout 变体只有四种**:`callout--analogy` / `callout--note` / `callout--tip` / `callout--warn`(定义在 `assets/book.css`)。别用不存在的类名(如 `callout--key`),否则不会有样式。
- **新讲的重要术语要同步进 `glossary.html`**:按现有 `<li id="g-xxx" data-aliases="别名,别名">…<a href="chapter-XX.html#锚点">→ 第 X 章</a></li>` 格式补,放进对应分组(网络基础 / 训练三件套 / 大模型 等),否则会出现“书里讲了、术语表查不到”的空缺。
- **并发编辑时可用脚本改**:若有多个 agent 同时改 dl-book 文件,内置 `Edit` 可能反复报“File has not been read”;此时可写一次性 Python 脚本做带 `assert 命中数==预期` 的字符串替换绕开状态跟踪,跑完即删。

# 推理配套实验

对应电子书[第 19 章 §9.4：跨层 KV 共享](../../../docs/dl-book/chapter-19.html#94-跨层-kv-共享-几层共用一份历史)与
[第 21 章 §7：推测解码](../../../docs/dl-book/chapter-21.html#④-推测解码-草稿模型先猜主模型再验)。
两个程序只使用 C++17 标准库，不下载模型、不训练，不需要 GPU。

## 编译运行

从仓库根目录独立编译运行：

    c++ -std=c++17 -O2 src/demo/inference_lab/kv_sharing.cpp -o /tmp/kv_sharing
    /tmp/kv_sharing
    c++ -std=c++17 -O2 src/demo/inference_lab/speculative_decode.cpp -o /tmp/speculative_decode
    /tmp/speculative_decode

也可使用项目构建（首次配置完整项目需初始化子模块）：

    git submodule update --init --recursive
    cmake -S src -B src/build -DENABLE_DRAW=OFF
    cmake --build src/build --target kv_sharing speculative_decode
    ./src/bin/kv_sharing
    ./src/bin/speculative_decode

两个程序都内置正确性检查，通过后输出 PASS，失败返回非零退出码。

| 程序 | 实验与检查 | 结果边界 |
| --- | --- | --- |
| kv_sharing | 四层固定权重网络；独立、全部共享、分组共享；增量路径与整段/各前缀重算对照；未来扰动不影响历史；缓存账本 | 统计有效 K/V 数据，不含容器容量、权重或激活；不同架构输出不要求相同 |
| speculative_decode | 三词表 Markov 模型；概率质量恒等式；零概率、全拒绝、全接受；普通/固定/动态生成各 10 万条六词序列，对照 729 种序列的理论联合分布 | 调用和位置数是模拟计数，成本系数是假设值；没有真实 GPU 并行、KV 回退或训练出的置信度头 |

## 跨层 KV 共享

主要入口是 FullForward 与 CachedForward。用 owners 数组指定缓存来源，
消费者通过只读引用访问来源缓存。只有来源层追加新 K/V。
默认四层全部共享，也检查两层一组的共享方式。

| 输入 token 数 | 四份独立缓存的有效数据 | 一份共享缓存的有效数据 |
| --- | --- | --- |
| 8 | 2,048 B | 512 B |
| 32 | 8,192 B | 2,048 B |
| 128 | 32,768 B | 8,192 B |

以上按本环境中 8 字节的 double 计算。增量计算与同结构重算的最大误差小于 1e-12；
本次运行显示 0。独立结构与共享结构的输出差约为 0.0175，改变结构后不要求输出相同。
这是单头、固定权重的简化网络，保留注意力和残差，简化前馈部分、省略 LayerNorm，
用于验证缓存机制，不评测语言能力。

## 推测解码

主要入口是 Residual、Generate、ChooseLength。目标分布 p、草稿分布 q 都以同一前缀为条件；
按 min(1,p/q) 接受，首次拒绝时按归一化的 max(p-q,0) 补采样并丢弃后缀；
全部接受时，再从目标模型的额外位置抽一个 token。
概率表的每一行表示“给定上一个 token 时，下一个 token 的分布”。

动态长度在抽取本轮草稿前确定，只取决于已提交前缀；它按当前一轮的期望收益选择长度，
不保证整个有限任务的全局最优成本。改变成本时只需修改文件开头
kRoundCost、kTargetRowCost、kDraftTokenCost，调度和统计会同时使用新值。

默认参数、固定随机种子的运行结果：

| 策略 | 目标调用轮数 | 目标位置数 | joint_TV | 模拟成本 / 输出 token |
| --- | --- | --- | --- | --- |
| 普通生成 | 600,000 | 600,000 | 0.02224 | 1.15000 |
| 固定上限 5 | 239,218 | 1,059,032 | 0.02288 | 0.70445 |
| 动态长度 | 265,702 | 986,322 | 0.02328 | 0.72545 |

joint_TV 是采样得到的六词序列联合分布与理论分布的总变差距离，有限抽样下通常不为零；
默认回归阈值为 0.04。改变样本数或序列长度后，应重新评估抽样误差。
动态策略在这组短任务中少算一些目标位置，但增加了调用轮数，模拟成本略高于固定长度。
成本统计不包含选择长度本身的计算；这里是理解取舍的实验，不是 DSpark 的实现或性能复现。

## 进一步阅读

正文小节中的近期案例使用以下固定版本资料：

- [DeepInfer 跨层共享拓扑](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/cross_layer.py)
- [DeepInfer 推测解码部署说明](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/doc/inference_deployment_qa.md#speculative-decoding)

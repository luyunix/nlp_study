# Transformer：38 节零基础学习笔记

这套笔记覆盖原视频 **P106–P143**，重新编号为 **第 1–38 节**。每一节采用与文本预处理笔记相同的结构：问题目标、零基础解释、原创 SVG、最小可运行代码、输入输出、易错点、知识链、自测答案和前后导航。

图片只承载知识内容，不包含视频分集、课程来源或平台标识。

## 建议学习方法

1. 先写出输入形状，不看答案预测输出形状。
2. 对注意力始终标出 Q、K、V 分别来自哪里。
3. 对 mask 始终标出 True/False 的语义和可广播形状。
4. 运行最小代码，再到配套实现中定位对应类或函数。
5. 一阶段学完后，从头口述一次完整数据流。

## 一、认识整体架构

1. [Transformer 的由来：为什么不再逐词递归](./01-transformer-origin.md)（原视频 P106）
2. [总体架构文字版：Encoder 理解，Decoder 生成](./02-transformer-architecture-text.md)（原视频 P107）
3. [架构图上半部分：看懂 Encoder 数据流](./03-transformer-diagram-upper.md)（原视频 P108）
4. [架构图下半部分：Decoder 为什么有两种注意力](./04-transformer-diagram-lower.md)（原视频 P109）
5. [从零实现路线：先零件，再整机](./05-transformer-learning-roadmap.md)（原视频 P110）

## 二、输入与位置编码

6. [Token Embedding：把 ID 查成向量](./06-token-embedding-code.md)（原视频 P111）
7. [正弦位置编码原理：让模型知道先后顺序](./07-positional-encoding-principle.md)（原视频 P112）
8. [位置编码总结：内容回答“是什么”，位置回答“在哪儿”](./08-positional-encoding-summary.md)（原视频 P113）
9. [PositionalEncoding 代码：预计算并注册 buffer](./09-positional-encoding-code.md)（原视频 P114）
10. [位置编码测试：重点检查形状、广播和确定性](./10-positional-encoding-test.md)（原视频 P115）

## 三、掩码与缩放点积注意力

11. [上三角矩阵：标记当前位置之后的未来](./11-upper-triangular-matrix.md)（原视频 P116）
12. [下三角可见区：只看自己和过去](./12-lower-triangular-matrix.md)（原视频 P117）
13. [因果 Mask 可视化：学会读横纵轴](./13-mask-visualization.md)（原视频 P118）
14. [masked_fill：在 Softmax 前把未来分数压到极小](./14-masked-fill.md)（原视频 P119）
15. [缩放点积注意力：Q 找谁，V 提供什么](./15-scaled-dot-product-attention.md)（原视频 P120）

## 四、多头注意力与前馈网络

16. [多头注意力原理上：把特征空间分给多个头](./16-multi-head-attention-principle-upper.md)（原视频 P121）
17. [多头注意力原理下：拆头、计算、再合头](./17-multi-head-attention-principle-lower.md)（原视频 P122）
18. [MultiHeadedAttention 代码：四个线性层和形状重排](./18-multi-head-attention-code.md)（原视频 P123）
19. [多头注意力测试：不能只检查“能运行”](./19-multi-head-attention-test.md)（原视频 P124）
20. [Position-wise FFN：每个位置独立加工特征](./20-positionwise-feed-forward.md)（原视频 P125）

## 五、LayerNorm 与残差子层

21. [LayerNorm 代码：在每个 token 内标准化特征](./21-layer-normalization-code.md)（原视频 P126）
22. [LayerNorm 测试：均值约 0、方差约 1](./22-layer-normalization-test.md)（原视频 P127）
23. [BatchNorm 与 LayerNorm：统计方向不同](./23-batchnorm-vs-layernorm.md)（原视频 P128）
24. [SublayerConnection：残差、Dropout 与归一化](./24-sublayer-connection-code.md)（原视频 P129）
25. [子层连接测试：用 lambda 注入不同组件](./25-sublayer-connection-test.md)（原视频 P130）

## 六、Encoder 与 Decoder

26. [EncoderLayer 代码：两个子层串起来](./26-encoder-layer-code.md)（原视频 P131）
27. [EncoderLayer 测试：看模块树，也看 mask](./27-encoder-layer-test.md)（原视频 P132）
28. [Encoder 堆叠：N 层独立参数逐层提炼](./28-encoder-code-and-test.md)（原视频 P133）
29. [DecoderLayer 代码：三个子层与两种长度](./29-decoder-layer-code.md)（原视频 P134）
30. [DecoderLayer 测试：故意让源长和目标长不同](./30-decoder-layer-test.md)（原视频 P135）
31. [Decoder 堆叠：每层共享 memory，目标状态逐层变化](./31-decoder-code-and-test.md)（原视频 P136）

## 七、输出与完整模型

32. [Generator 代码：从隐藏维映射到目标词表](./32-generator-code.md)（原视频 P137）
33. [Generator 测试：指数还原后概率和应为 1](./33-generator-test.md)（原视频 P138）
34. [完整模型上：forward 如何组织编码和解码](./34-full-transformer-upper.md)（原视频 P139）
35. [完整模型下：encode/decode 接口要分清](./35-full-transformer-lower.md)（原视频 P140）
36. [完整模型组装：make_model 把所有组件接起来](./36-transformer-test-upper.md)（原视频 P141）
37. [组件总复盘：从模型树反向读出数据流](./37-transformer-components-review.md)（原视频 P142）
38. [完整模型端到端测试：最终输出不等于训练完成](./38-transformer-test-lower.md)（原视频 P143）

## 一张形状主线

```text
src/tgt ids                 [B, L]
Embedding + Position        [B, L, D]
split heads                 [B, h, L, d_k]
attention scores            [B, h, Lq, Lk]
merge heads                 [B, Lq, D]
Encoder memory              [B, Ls, D]
Decoder hidden              [B, Lt, D]
Generator log probabilities [B, Lt, Vt]
```

## 三组必须会写的 Q/K/V

| 位置 | Q | K | V | 主要 mask |
|---|---|---|---|---|
| Encoder 自注意力 | 源表示 | 源表示 | 源表示 | 源 PAD |
| Decoder 自注意力 | 目标表示 | 目标表示 | 目标表示 | 目标 PAD + 因果 |
| Decoder 交叉注意力 | 目标表示 | memory | memory | 源 PAD |

## 配套代码

- [完整实现](../../transformer_from_scratch/model.py)
- [自动测试](../../transformer_from_scratch/test_model.py)
- [代码说明](../../transformer_from_scratch/README.md)
- 运行：`python3 -m unittest transformer_from_scratch.test_model -v`

## 已校准的关键点

- token ID 必须严格小于对应词表大小。
- LayerNorm 方差使用 `unbiased=False` 才与标准实现一致。
- 本项目明确采用 Pre-LN：`x + Dropout(Sublayer(LayerNorm(x)))`。
- 多头 mask 的第一维是 batch 或 1，不是头数。
- 源/目标 Embedding 不能在词表不同时盲目共享。
- Cross-Attention 的 Q 来自目标侧，K/V 来自 Encoder memory。

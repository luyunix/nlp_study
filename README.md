# NLP 零基础学习笔记：1–189 节总目录

这是按老师原声讲解顺序整理的完整学习入口，共 189 篇。每篇都包含：

- 老师讲解改写成的连贯文章；
- 可核对遗漏的按时间戳原声音轨记录；
- 原创概念图，以及流程图、UML 或时序图；
- 最小代码、输入输出解释、易错点和自测。

## 小白学习顺序

1. 先读每节“这节解决什么问题”，不要急着背术语。
2. 顺着原创图把数据流说一遍。
3. 阅读“老师原声整理稿”，遇到形状就翻译成“几条样本 × 几个 token × 每个多少维”。
4. 运行最小代码，先猜输出，再核对结果。
5. 完成自测；答不出来就回到对应时间段，而不是整节重看。

> 原声转写用于完整性核查，可能仍有同音字；正文已对技术术语、口误和关键概念作校正。

## 1–4　课程导学

[打开专题学习说明](./notes/course-introduction/README.md)

1. [课程导学：为什么学 NLP，以及整门课怎样走](./notes/course-introduction/01-course-guide.md)
2. [阶段大纲：六章内容、重点难点与案例路线](./notes/course-introduction/02-stage-outline.md)
3. [NLP 概念与发展：从图灵测试到 Transformer](./notes/course-introduction/03-nlp-concepts-and-history.md)
4. [NLP 应用场景：语音助手、身份核验与机器翻译](./notes/course-introduction/04-nlp-applications.md)

## 5–37　文本预处理

[打开专题学习说明](./notes/text-preprocessing/README.md)

5. [文本预处理全景：先看清问题，再动手处理](./notes/text-preprocessing/01-overview.md)
6. [环境准备与分词：机器眼中的句子没有天然词界](./notes/text-preprocessing/02-environment-and-tokenization.md)
7. [jieba 精确模式：给句子一条主要切分路径](./notes/text-preprocessing/03-jieba-precise-mode.md)
8. [jieba 全模式：把可能的词尽量找出来](./notes/text-preprocessing/04-jieba-full-mode.md)
9. [jieba 搜索引擎模式：长词再拆一层，提高召回](./notes/text-preprocessing/05-jieba-search-mode.md)
10. [繁体中文分词：接口相同，词典覆盖决定效果](./notes/text-preprocessing/06-traditional-chinese.md)
11. [自定义词典：教分词器认识你的领域词](./notes/text-preprocessing/07-custom-dictionary.md)
12. [命名实体识别与词性标注：词是什么角色](./notes/text-preprocessing/08-ner-and-pos.md)
13. [文本张量表示：模型为什么只接收数字](./notes/text-preprocessing/09-text-to-tensor.md)
14. [One-Hot 生成：从词表位置得到独热向量](./notes/text-preprocessing/10-one-hot-generation.md)
15. [One-Hot 使用：加载同一映射并处理未知词](./notes/text-preprocessing/11-one-hot-usage.md)
16. [手写 One-Hot：看见稀疏表示的优点和代价](./notes/text-preprocessing/12-simple-one-hot.md)
17. [Word2Vec 的 CBOW：用上下文猜中间词](./notes/text-preprocessing/13-word2vec-cbow.md)
18. [Word2Vec 的 Skip-Gram：用中间词猜周围词](./notes/text-preprocessing/14-word2vec-skipgram.md)
19. [FastText 准备：从大语料到可训练文件](./notes/text-preprocessing/15-fasttext-setup.md)
20. [FastText 训练与保存：让语料自己产生监督信号](./notes/text-preprocessing/16-fasttext-training.md)
21. [FastText 加载、查看与评估：向量好不好要验证](./notes/text-preprocessing/17-fasttext-evaluation.md)
22. [FastText 超参数：每个旋钮改变什么](./notes/text-preprocessing/18-fasttext-hyperparameters.md)
23. [Word2Vec 与 Embedding：预训练方法和查表层的区别](./notes/text-preprocessing/19-embedding-vs-word2vec.md)
24. [Embedding 取词向量：从句子到三维张量](./notes/text-preprocessing/20-embedding-lookup.md)
25. [Embedding 可视化：把高维空间投影到屏幕](./notes/text-preprocessing/21-embedding-visualization.md)
26. [标签数量分布：先发现类别不平衡](./notes/text-preprocessing/22-label-distribution.md)
27. [map：把同一个函数依次用到每条数据](./notes/text-preprocessing/23-map-function.md)
28. [句子长度分布：为截断和补齐找依据](./notes/text-preprocessing/24-sentence-length-distribution.md)
29. [按标签比较长度：长度本身也可能泄露规律](./notes/text-preprocessing/25-length-by-label.md)
30. [chain：把嵌套词列表铺平成一个词流](./notes/text-preprocessing/26-itertools-chain.md)
31. [词汇量统计：不同词有多少、各出现几次](./notes/text-preprocessing/27-vocabulary-count.md)
32. [形容词词云：先按词性筛选，再按频率画图](./notes/text-preprocessing/28-adjective-wordcloud.md)
33. [zip：把多个等长位置对齐成元组](./notes/text-preprocessing/29-zip-function.md)
34. [n-gram：保留连续局部顺序的文本特征](./notes/text-preprocessing/30-n-gram.md)
35. [文本相似度：先对齐特征，再比较方向或集合](./notes/text-preprocessing/31-text-similarity.md)
36. [长度规范化：截断与补齐组成统一批次](./notes/text-preprocessing/32-length-normalization.md)
37. [回译数据增强：换一种说法，尽量保持标签](./notes/text-preprocessing/33-back-translation.md)

## 38–65　RNN

[打开专题学习说明](./notes/rnn/README.md)

38. [RNN 简介：让当前判断带上过去的信息](./notes/rnn/01-rnn-introduction.md)
39. [RNN 分类：输入输出关系与内部结构是两个维度](./notes/rnn/02-rnn-types.md)
40. [RNN 模型结构：公式、共享权重与张量形状](./notes/rnn/03-rnn-structure.md)
41. [RNN 基础代码：创建层、准备输入、运行并验形状](./notes/rnn/04-rnn-basic-code.md)
42. [修改句长：只应改变 output 的时间维](./notes/rnn/05-change-sequence-length.md)
43. [修改隐藏层与总结：维度代表模型的记忆容量](./notes/rnn/06-change-hidden-size.md)
44. [LSTM 图解（上）：遗忘门与输入门管理长期记忆](./notes/rnn/07-lstm-diagram-part1.md)
45. [LSTM 图解（下）：输出门产生当前隐藏状态](./notes/rnn/08-lstm-diagram-part2.md)
46. [Bi-LSTM：从前后两个方向理解同一位置](./notes/rnn/09-bidirectional-lstm.md)
47. [LSTM 代码：多一个细胞状态，接口如何变化](./notes/rnn/10-lstm-code.md)
48. [GRU 图解：两扇门合并记忆管理](./notes/rnn/11-gru-diagram.md)
49. [GRU 代码：替换循环层并验证接口](./notes/rnn/12-gru-code.md)
50. [姓名分类需求：从名字预测 18 个国家类别](./notes/rnn/13-name-classification-requirement.md)
51. [全局字母表与国家名：固定输入列和输出列](./notes/rnn/14-alphabet-and-countries.md)
52. [读取数据：把姓名与国家分别保存并处理异常行](./notes/rnn/15-read-name-data.md)
53. [Dataset：把变长姓名转成字符 One-Hot 张量](./notes/rnn/16-dataset.md)
54. [DataLoader：为变长姓名组织训练迭代](./notes/rnn/17-dataloader.md)
55. [LogSoftmax：旧式 NLLLoss 与现代 CrossEntropyLoss 的关系](./notes/rnn/18-log-softmax.md)
56. [RNN 分类模型：取最后时间步映射到 18 类](./notes/rnn/19-rnn-model.md)
57. [测试 RNN：用随机输入把形状链走通](./notes/rnn/20-test-rnn.md)
58. [搭建 LSTM 与 GRU 模型：复用分类头，隔离状态差异](./notes/rnn/21-lstm-gru-models.md)
59. [测试三种模型：用同一数据管道公平验形状](./notes/rnn/22-test-three-models.md)
60. [训练 RNN：外层 epoch、内层 batch 与五步反向传播](./notes/rnn/23-train-rnn.md)
61. [训练 LSTM：复用训练循环，正确管理 h 与 c](./notes/rnn/24-train-lstm.md)
62. [训练 GRU：同一协议下比较速度与效果](./notes/rnn/25-train-gru.md)
63. [可视化三模型：损失、时间和准确率要一起看](./notes/rnn/26-visualize-comparison.md)
64. [RNN 预测：姓名转张量、加载权重并取 Top-k](./notes/rnn/27-rnn-prediction.md)
65. [姓名分类总结：按单模型跑通，再统一重构](./notes/rnn/28-name-classifier-summary.md)

## 66–79　Attention

[打开专题学习说明](./notes/attention/README.md)

66. [注意力机制介绍：把有限精力分给更相关的信息](./notes/attention/01-attention-introduction.md)
67. [Q、K、V：问题、索引与实际内容](./notes/attention/02-qkv-introduction.md)
68. [注意力实现步骤：算分、归一化、加权求和](./notes/attention/03-attention-steps.md)
69. [Seq2Seq 任务：编码器把输入交给解码器逐词生成](./notes/attention/04-seq2seq-task.md)
70. [Seq2Seq 加入注意力：每个目标词拥有自己的 context](./notes/attention/05-seq2seq-with-attention.md)
71. [普通 Encoder-Decoder：一个 C 服务所有解码步骤](./notes/attention/06-plain-encoder-decoder.md)
72. [带注意力 Encoder-Decoder：从公式看 Cᵢ 怎样生成](./notes/attention/07-attentive-encoder-decoder.md)
73. [注意力概率分布：Decoder 状态如何与所有 Encoder 状态比较](./notes/attention/08-attention-probabilities.md)
74. [软注意力、硬注意力与自注意力](./notes/attention/09-attention-types.md)
75. [常见注意力计算规则：拼接式、加法式与缩放点积](./notes/attention/10-attention-scoring-rules.md)
76. [bmm：一次完成一批三维矩阵乘法](./notes/attention/11-bmm.md)
77. [注意力代码实现：从 Q/K/V 到增强后的查询](./notes/attention/12-attention-code.md)
78. [注意力测试代码：检查形状、概率和与梯度](./notes/attention/13-attention-test.md)
79. [参数解释：把 1×32、32×64、1×96 全部读成语义](./notes/attention/14-attention-parameters.md)

## 80–105　Seq2Seq 英法翻译

[打开专题学习说明](./notes/seq2seq-translation/README.md)

80. [英译法需求：先看懂 800 行项目的六个模块](./notes/seq2seq-translation/01-translation-requirements.md)
81. [CUDA 环境（上）：GPU、驱动、工具包与 PyTorch 不是同一层](./notes/seq2seq-translation/02-cuda-concepts.md)
82. [CUDA 环境实操：创建环境、安装、验证与排错](./notes/seq2seq-translation/03-cuda-practice.md)
83. [CUDA 配置总结：把可复现信息写进项目](./notes/seq2seq-translation/04-cuda-summary.md)
84. [数据清洗：规范文本，但不要改坏翻译含义](./notes/seq2seq-translation/05-data-cleaning.md)
85. [数据预处理：建两套词表并加入 SOS/EOS](./notes/seq2seq-translation/06-preprocessing.md)
86. [构建 Dataset：一条样本同时返回源 ID 与目标 ID](./notes/seq2seq-translation/07-dataset.md)
87. [获取 DataLoader：补齐、长度和 mask 一起产出](./notes/seq2seq-translation/08-dataloader.md)
88. [GRU Encoder：Embedding 后保留每个时间步输出](./notes/seq2seq-translation/09-gru-encoder.md)
89. [测试 Encoder：先验形状再运行](./notes/seq2seq-translation/10-test-encoder.md)
90. [无 Attention Decoder 思路：只靠 final hidden 生成](./notes/seq2seq-translation/11-plain-decoder-plan.md)
91. [构建无 Attention GRU Decoder](./notes/seq2seq-translation/12-plain-decoder-code.md)
92. [测试无 Attention Decoder：逐词循环与 EOS](./notes/seq2seq-translation/13-test-plain-decoder.md)
93. [有 Attention Decoder 思路：每步重新查询源句](./notes/seq2seq-translation/14-attention-decoder-plan.md)
94. [有 Attention Decoder 代码（上）：定义层与接口](./notes/seq2seq-translation/15-attention-decoder-code-part1.md)
95. [有 Attention Decoder 代码（下）：逐行完成 forward_step](./notes/seq2seq-translation/16-attention-decoder-code-part2.md)
96. [测试 Attention Decoder：权重、mask 与单步输出](./notes/seq2seq-translation/17-test-attention-decoder.md)
97. [模型搭建总结：三个模块如何对接](./notes/seq2seq-translation/18-model-summary.md)
98. [Teacher Forcing：训练时有时喂真值上一词](./notes/seq2seq-translation/19-teacher-forcing.md)
99. [模型训练（单批次）：先把一次前向和损失走通](./notes/seq2seq-translation/20-train-one-batch.md)
100. [view()：只改观察形状，不改元素顺序](./notes/seq2seq-translation/21-view-function.md)
101. [完整训练代码：epoch、验证、保存与日志](./notes/seq2seq-translation/22-full-training.md)
102. [训练总结：把 800 行压缩成一条可复述主线](./notes/seq2seq-translation/23-training-summary.md)
103. [模型预测代码：无真值时逐词生成](./notes/seq2seq-translation/24-prediction-code.md)
104. [预测代码测试：从样例翻译发现数据与模型问题](./notes/seq2seq-translation/25-prediction-test.md)
105. [绘制张量图：用图检查模块连接，不把图当性能证明](./notes/seq2seq-translation/26-tensorboard-graph.md)

## 106–143　Transformer

[打开专题学习说明](./notes/transformer/README.md)

106. [Transformer 的由来：为什么不再逐词递归](./notes/transformer/01-transformer-origin.md)
107. [总体架构文字版：Encoder 理解，Decoder 生成](./notes/transformer/02-transformer-architecture-text.md)
108. [架构图上半部分：看懂 Encoder 数据流](./notes/transformer/03-transformer-diagram-upper.md)
109. [架构图下半部分：Decoder 为什么有两种注意力](./notes/transformer/04-transformer-diagram-lower.md)
110. [从零实现路线：先零件，再整机](./notes/transformer/05-transformer-learning-roadmap.md)
111. [Token Embedding：把 ID 查成向量](./notes/transformer/06-token-embedding-code.md)
112. [正弦位置编码原理：让模型知道先后顺序](./notes/transformer/07-positional-encoding-principle.md)
113. [位置编码总结：内容回答“是什么”，位置回答“在哪儿”](./notes/transformer/08-positional-encoding-summary.md)
114. [PositionalEncoding 代码：预计算并注册 buffer](./notes/transformer/09-positional-encoding-code.md)
115. [位置编码测试：重点检查形状、广播和确定性](./notes/transformer/10-positional-encoding-test.md)
116. [上三角矩阵：标记当前位置之后的未来](./notes/transformer/11-upper-triangular-matrix.md)
117. [下三角可见区：只看自己和过去](./notes/transformer/12-lower-triangular-matrix.md)
118. [因果 Mask 可视化：学会读横纵轴](./notes/transformer/13-mask-visualization.md)
119. [masked_fill：在 Softmax 前把未来分数压到极小](./notes/transformer/14-masked-fill.md)
120. [缩放点积注意力：Q 找谁，V 提供什么](./notes/transformer/15-scaled-dot-product-attention.md)
121. [多头注意力原理上：把特征空间分给多个头](./notes/transformer/16-multi-head-attention-principle-upper.md)
122. [多头注意力原理下：拆头、计算、再合头](./notes/transformer/17-multi-head-attention-principle-lower.md)
123. [MultiHeadedAttention 代码：四个线性层和形状重排](./notes/transformer/18-multi-head-attention-code.md)
124. [多头注意力测试：不能只检查“能运行”](./notes/transformer/19-multi-head-attention-test.md)
125. [Position-wise FFN：每个位置独立加工特征](./notes/transformer/20-positionwise-feed-forward.md)
126. [LayerNorm 代码：在每个 token 内标准化特征](./notes/transformer/21-layer-normalization-code.md)
127. [LayerNorm 测试：均值约 0、方差约 1](./notes/transformer/22-layer-normalization-test.md)
128. [BatchNorm 与 LayerNorm：统计方向不同](./notes/transformer/23-batchnorm-vs-layernorm.md)
129. [SublayerConnection：残差、Dropout 与归一化](./notes/transformer/24-sublayer-connection-code.md)
130. [子层连接测试：用 lambda 注入不同组件](./notes/transformer/25-sublayer-connection-test.md)
131. [EncoderLayer 代码：两个子层串起来](./notes/transformer/26-encoder-layer-code.md)
132. [EncoderLayer 测试：看模块树，也看 mask](./notes/transformer/27-encoder-layer-test.md)
133. [Encoder 堆叠：N 层独立参数逐层提炼](./notes/transformer/28-encoder-code-and-test.md)
134. [DecoderLayer 代码：三个子层与两种长度](./notes/transformer/29-decoder-layer-code.md)
135. [DecoderLayer 测试：故意让源长和目标长不同](./notes/transformer/30-decoder-layer-test.md)
136. [Decoder 堆叠：每层共享 memory，目标状态逐层变化](./notes/transformer/31-decoder-code-and-test.md)
137. [Generator 代码：从隐藏维映射到目标词表](./notes/transformer/32-generator-code.md)
138. [Generator 测试：指数还原后概率和应为 1](./notes/transformer/33-generator-test.md)
139. [完整模型上：forward 如何组织编码和解码](./notes/transformer/34-full-transformer-upper.md)
140. [完整模型下：encode/decode 接口要分清](./notes/transformer/35-full-transformer-lower.md)
141. [完整模型组装：make_model 把所有组件接起来](./notes/transformer/36-transformer-test-upper.md)
142. [组件总复盘：从模型树反向读出数据流](./notes/transformer/37-transformer-components-review.md)
143. [完整模型端到端测试：最终输出不等于训练完成](./notes/transformer/38-transformer-test-lower.md)

## 144–154　FastText 文本分类

[打开专题学习说明](./notes/fasttext-classification/README.md)

144. [FastText 简介与环境：为什么一行 API 也值得学原理](./notes/fasttext-classification/01-fasttext-introduction.md)
145. [层次 Softmax 与哈夫曼树：从全部类别改成走一条路径](./notes/fasttext-classification/02-hierarchical-softmax-huffman.md)
146. [负采样：正样本之外，只挑少量“陪练”负样本](./notes/fasttext-classification/03-negative-sampling.md)
147. [直接训练：数据格式、预测、测试与第一版基线](./notes/fasttext-classification/04-direct-training.md)
148. [数据预处理：统一大小写、分开标点，并保持训练预测一致](./notes/fasttext-classification/05-data-preprocessing.md)
149. [训练轮数与学习率：一个决定看几遍，一个决定每步走多远](./notes/fasttext-classification/06-epochs-learning-rate.md)
150. [word N-gram 与损失函数：补一点局部词序，再选合适输出方式](./notes/fasttext-classification/07-ngrams-loss.md)
151. [自动调参：给验证集和时间预算，让程序寻找更好组合](./notes/fasttext-classification/08-autotune.md)
152. [多分类多标签：OVA、k 与 threshold 的完整含义](./notes/fasttext-classification/09-multilabel-ova.md)
153. [保存与加载模型：训练一次，部署和复现实验反复使用](./notes/fasttext-classification/10-save-load-model.md)
154. [词向量迁移：把别人训练好的语义坐标系拿来起步](./notes/fasttext-classification/11-pretrained-vectors-transfer.md)

## 155–183　迁移学习与 Transformers 任务

[打开专题学习说明](./notes/transfer-learning/README.md)

155. [迁移学习：把别人学会的能力搬到相关任务](./notes/transfer-learning/01-transfer-learning-introduction.md)
156. [常见预训练 NLP 模型：Encoder、Decoder 与 Encoder-Decoder](./notes/transfer-learning/02-pretrained-model-families.md)
157. [Transformers 库与环境：模型仓库、缓存和三种调用方式](./notes/transfer-learning/03-transformers-library-setup.md)
158. [Pipeline 文本分类：三行代码背后的标签与概率](./notes/transfer-learning/04-pipeline-text-classification.md)
159. [Pipeline 特征提取：没有任务头的“半成品”怎样读形状](./notes/transfer-learning/05-pipeline-feature-extraction.md)
160. [Pipeline 完形填空：`[MASK]`、候选概率与多个空位](./notes/transfer-learning/06-pipeline-fill-mask.md)
161. [Pipeline 阅读理解：从 context 中预测答案起止位置](./notes/transfer-learning/07-pipeline-question-answering.md)
162. [Pipeline 文本摘要：生成长度、截断与事实一致性](./notes/transfer-learning/08-pipeline-summarization.md)
163. [Pipeline NER：token 标签怎样合并成人名、地点和组织](./notes/transfer-learning/09-pipeline-ner.md)
164. [Auto 模型文本分类：手工完成分词、前向与 argmax](./notes/transfer-learning/10-auto-text-classification.md)
165. [Auto 模型特征提取：last_hidden_state 与 masked mean pooling](./notes/transfer-learning/11-auto-feature-extraction.md)
166. [Auto 模型完形填空：定位 mask，再从词表 logits 取 top-k](./notes/transfer-learning/12-auto-fill-mask.md)
167. [Auto 模型阅读理解：start_logits 与 end_logits 组合答案](./notes/transfer-learning/13-auto-question-answering.md)
168. [Auto 模型文本摘要：tokenize、generate 与 decode 分工](./notes/transfer-learning/14-auto-summarization.md)
169. [Auto 模型 NER：subword 标签对齐与实体聚合](./notes/transfer-learning/15-auto-ner.md)
170. [具体模型类做完形填空：显式使用 BertTokenizer 与 BertForMaskedLM](./notes/transfer-learning/16-specific-model-fill-mask.md)
171. [中文分类案例（一）：先把原始数据加载成 text/label 样本](./notes/transfer-learning/17-classification-data-loading.md)
172. [中文分类案例（二）：批量分词、padding、truncation 与 attention_mask](./notes/transfer-learning/18-classification-preprocessing.md)
173. [中文分类案例（三）：自定义 BERT + Linear(768→2) 网络](./notes/transfer-learning/19-classification-model.md)
174. [中文分类案例（四）：训练循环、梯度更新和学习率调度](./notes/transfer-learning/20-classification-training.md)
175. [中文分类案例（五）：eval/no_grad、准确率与保存最佳模型](./notes/transfer-learning/21-classification-evaluation.md)
176. [中文填空案例（一）：固定遮罩第 16 个位置的数据整理](./notes/transfer-learning/22-mlm-preprocessing.md)
177. [中文填空案例（二）：自定义 BERT + Linear(768→词表大小)](./notes/transfer-learning/23-mlm-model.md)
178. [中文填空案例（三）：过滤长文本并复用分类训练循环](./notes/transfer-learning/24-mlm-training.md)
179. [中文填空案例（四）：加载 FillMask 模型并计算固定位置准确率](./notes/transfer-learning/25-mlm-evaluation.md)
180. [NSP 案例（一）：自定义句对数据，构造真下一句与随机下一句](./notes/transfer-learning/26-nsp-custom-dataset.md)
181. [NSP 案例（二）：句对编码、token_type_ids 与特殊 token](./notes/transfer-learning/27-nsp-preprocessing.md)
182. [NSP 案例（三）：复用自定义 BERT + Linear(768→2)](./notes/transfer-learning/28-nsp-model.md)
183. [NSP 案例（四）：训练、评估与数据捷径检查](./notes/transfer-learning/29-nsp-training-evaluation.md)

## 184–189　BERT、ELMo 与 GPT

[打开专题学习说明](./notes/foundation-models/README.md)

184. [BERT 介绍：为什么它能成为通用的文本理解底座](./notes/foundation-models/01-bert-introduction.md)
185. [BERT 架构：三种 Embedding、Encoder 堆叠与关键形状](./notes/foundation-models/02-bert-architecture.md)
186. [BERT 预训练任务：MLM 与 NSP 怎样共同制造监督](./notes/foundation-models/03-bert-mlm-nsp.md)
187. [BERT 总结：MLM/NSP 复盘，以及 GLUE 与 CLUE 公共评测](./notes/foundation-models/04-bert-summary.md)
188. [ELMo：在 Transformer 之前，双向语言模型怎样生成动态词向量](./notes/foundation-models/05-elmo-introduction.md)
189. [GPT：因果注意力怎样把“预测下一个 token”扩展成文本生成](./notes/foundation-models/06-gpt-introduction.md)

## 配套可运行代码

- [文本预处理练习](./text_preprocessing_from_scratch/README.md)
- [RNN 练习](./rnn_from_scratch/README.md)
- [Attention 练习](./attention_from_scratch/README.md)
- [Seq2Seq 练习](./seq2seq_from_scratch/README.md)
- [Transformer 练习](./transformer_from_scratch/README.md)
- [FastText 原理练习](./fasttext_from_scratch/README.md)

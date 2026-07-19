# 文本预处理：33 节零基础学习笔记

这套笔记覆盖课程原视频 **P5–P37**，按学习顺序重新编号为 **第 1–33 节**。P38 已进入循环神经网络，不属于本专题。

每一节都包含：问题目标、零基础解释、原创 SVG 示意图、最小可运行代码、输入输出说明、易错点、知识链、自测与答案。图片均为本项目原创绘制，**没有使用课程截图**。

## 建议学习方法

1. 先看图并用自己的话复述箭头。
2. 不看输出，先手算代码会打印什么。
3. 运行代码，若结果不同，找出自己漏掉的规则。
4. 做自测并回答“什么时候会用错”。
5. 每学完一个阶段，再把该阶段的代码连起来做一次。

## 一、基础方法

1. [文本预处理全景：先看清问题，再动手处理](./01-overview.md)（原视频 P5）
2. [环境准备与分词：机器眼中的句子没有天然词界](./02-environment-and-tokenization.md)（原视频 P6）
3. [jieba 精确模式：给句子一条主要切分路径](./03-jieba-precise-mode.md)（原视频 P7）
4. [jieba 全模式：把可能的词尽量找出来](./04-jieba-full-mode.md)（原视频 P8）
5. [jieba 搜索引擎模式：长词再拆一层，提高召回](./05-jieba-search-mode.md)（原视频 P9）
6. [繁体中文分词：接口相同，词典覆盖决定效果](./06-traditional-chinese.md)（原视频 P10）
7. [自定义词典：教分词器认识你的领域词](./07-custom-dictionary.md)（原视频 P11）
8. [命名实体识别与词性标注：词是什么角色](./08-ner-and-pos.md)（原视频 P12）

## 二、文本数值表示

9. [文本张量表示：模型为什么只接收数字](./09-text-to-tensor.md)（原视频 P13）
10. [One-Hot 生成：从词表位置得到独热向量](./10-one-hot-generation.md)（原视频 P14）
11. [One-Hot 使用：加载同一映射并处理未知词](./11-one-hot-usage.md)（原视频 P15）
12. [手写 One-Hot：看见稀疏表示的优点和代价](./12-simple-one-hot.md)（原视频 P16）
13. [Word2Vec 的 CBOW：用上下文猜中间词](./13-word2vec-cbow.md)（原视频 P17）
14. [Word2Vec 的 Skip-Gram：用中间词猜周围词](./14-word2vec-skipgram.md)（原视频 P18）
15. [FastText 准备：从大语料到可训练文件](./15-fasttext-setup.md)（原视频 P19）
16. [FastText 训练与保存：让语料自己产生监督信号](./16-fasttext-training.md)（原视频 P20）
17. [FastText 加载、查看与评估：向量好不好要验证](./17-fasttext-evaluation.md)（原视频 P21）
18. [FastText 超参数：每个旋钮改变什么](./18-fasttext-hyperparameters.md)（原视频 P22）
19. [Word2Vec 与 Embedding：预训练方法和查表层的区别](./19-embedding-vs-word2vec.md)（原视频 P23）
20. [Embedding 取词向量：从句子到三维张量](./20-embedding-lookup.md)（原视频 P24）
21. [Embedding 可视化：把高维空间投影到屏幕](./21-embedding-visualization.md)（原视频 P25）

## 三、语料分析

22. [标签数量分布：先发现类别不平衡](./22-label-distribution.md)（原视频 P26）
23. [map：把同一个函数依次用到每条数据](./23-map-function.md)（原视频 P27）
24. [句子长度分布：为截断和补齐找依据](./24-sentence-length-distribution.md)（原视频 P28）
25. [按标签比较长度：长度本身也可能泄露规律](./25-length-by-label.md)（原视频 P29）
26. [chain：把嵌套词列表铺平成一个词流](./26-itertools-chain.md)（原视频 P30）
27. [词汇量统计：不同词有多少、各出现几次](./27-vocabulary-count.md)（原视频 P31）
28. [形容词词云：先按词性筛选，再按频率画图](./28-adjective-wordcloud.md)（原视频 P32）

## 四、特征处理

29. [zip：把多个等长位置对齐成元组](./29-zip-function.md)（原视频 P33）
30. [n-gram：保留连续局部顺序的文本特征](./30-n-gram.md)（原视频 P34）
31. [文本相似度：先对齐特征，再比较方向或集合](./31-text-similarity.md)（原视频 P35）
32. [长度规范化：截断与补齐组成统一批次](./32-length-normalization.md)（原视频 P36）

## 五、数据增强

33. [回译数据增强：换一种说法，尽量保持标签](./33-back-translation.md)（原视频 P37）

## 配套代码

- [纯 Python 原理实现与测试](../../text_preprocessing_from_scratch/README.md)
- 运行测试：`python3 -m unittest text_preprocessing_from_scratch.test_core -v`

## 课程内容中的版本提醒

- 课程演示的部分 Keras/TensorFlow、Seaborn 和翻译接口会随时间变化；笔记保留原理，并标出更稳妥的现代写法。
- FastText 的重要能力之一是用字符 n-gram 组合未登录词（OOV）向量，不只是“找一个最相近的已知词”。
- TensorBoard Projector 是降维后的探索工具，图上的二维距离不能代替原空间的量化评估。

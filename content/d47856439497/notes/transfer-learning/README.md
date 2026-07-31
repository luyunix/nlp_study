# 迁移学习与 Transformers 任务：29 节零基础文章笔记

本专题按老师原声的实际讲解顺序整理。口头重复、停顿和现场操作被改写为连贯文章，但保留重要类比、代码解释、易错点和纠错过程。

每节包含原创 SVG 概念图；图片中不含视频分集、课程来源或平台标识。

## 学习方法

1. 先读“老师讲解整理成文章”，理解为什么要学。
2. 再沿原创图复述数据流。
3. 不看输出先运行或手算最小代码。
4. 完成自测，再进入下一节。

## 一、迁移学习、模型家族与环境

1. [迁移学习：把别人学会的能力搬到相关任务](./01-transfer-learning-introduction.md)（原视频 P155）
2. [常见预训练 NLP 模型：Encoder、Decoder 与 Encoder-Decoder](./02-pretrained-model-families.md)（原视频 P156）
3. [Transformers 库与环境：模型仓库、缓存和三种调用方式](./03-transformers-library-setup.md)（原视频 P157）

## 二、Pipeline 六类任务

4. [Pipeline 文本分类：三行代码背后的标签与概率](./04-pipeline-text-classification.md)（原视频 P158）
5. [Pipeline 特征提取：没有任务头的“半成品”怎样读形状](./05-pipeline-feature-extraction.md)（原视频 P159）
6. [Pipeline 完形填空：`[MASK]`、候选概率与多个空位](./06-pipeline-fill-mask.md)（原视频 P160）
7. [Pipeline 阅读理解：从 context 中预测答案起止位置](./07-pipeline-question-answering.md)（原视频 P161）
8. [Pipeline 文本摘要：生成长度、截断与事实一致性](./08-pipeline-summarization.md)（原视频 P162）
9. [Pipeline NER：token 标签怎样合并成人名、地点和组织](./09-pipeline-ner.md)（原视频 P163）

## 三、Auto 与具体类

10. [Auto 模型文本分类：手工完成分词、前向与 argmax](./10-auto-text-classification.md)（原视频 P164）
11. [Auto 模型特征提取：last_hidden_state 与 masked mean pooling](./11-auto-feature-extraction.md)（原视频 P165）
12. [Auto 模型完形填空：定位 mask，再从词表 logits 取 top-k](./12-auto-fill-mask.md)（原视频 P166）
13. [Auto 模型阅读理解：start_logits 与 end_logits 组合答案](./13-auto-question-answering.md)（原视频 P167）
14. [Auto 模型文本摘要：tokenize、generate 与 decode 分工](./14-auto-summarization.md)（原视频 P168）
15. [Auto 模型 NER：subword 标签对齐与实体聚合](./15-auto-ner.md)（原视频 P169）
16. [具体模型类做完形填空：显式使用 BertTokenizer 与 BertForMaskedLM](./16-specific-model-fill-mask.md)（原视频 P170）

## 四、中文分类完整微调

17. [中文分类案例（一）：先把原始数据加载成 text/label 样本](./17-classification-data-loading.md)（原视频 P171）
18. [中文分类案例（二）：批量分词、padding、truncation 与 attention_mask](./18-classification-preprocessing.md)（原视频 P172）
19. [中文分类案例（三）：自定义 BERT + Linear(768→2) 网络](./19-classification-model.md)（原视频 P173）
20. [中文分类案例（四）：训练循环、梯度更新和逐轮保存](./20-classification-training.md)（原视频 P174）
21. [中文分类案例（五）：加载 checkpoint、全量准确率与评估边界](./21-classification-evaluation.md)（原视频 P175）

## 五、中文 MLM 完形填空

22. [中文填空案例（一）：固定遮罩下标 16 的数据整理](./22-mlm-preprocessing.md)（原视频 P176）
23. [中文填空案例（二）：自定义 BERT + Linear(768→词表大小)](./23-mlm-model.md)（原视频 P177）
24. [中文填空案例（三）：过滤长文本并复用分类训练循环](./24-mlm-training.md)（原视频 P178）
25. [中文填空案例（四）：加载 FillMask 模型并计算固定位置准确率](./25-mlm-evaluation.md)（原视频 P179）

## 六、NSP 句对任务

26. [NSP 案例（一）：自定义句对数据，构造真下一句与随机下一句](./26-nsp-custom-dataset.md)（原视频 P180）
27. [NSP 案例（二）：句对编码、token_type_ids 与特殊 token](./27-nsp-preprocessing.md)（原视频 P181）
28. [NSP 案例（三）：复用自定义 BERT + Linear(768→2)](./28-nsp-model.md)（原视频 P182）
29. [NSP 案例（四）：训练、评估与数据捷径检查](./29-nsp-training-evaluation.md)（原视频 P183）



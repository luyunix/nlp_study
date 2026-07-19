# RNN、LSTM、GRU 与姓名分类：28 节零基础文章笔记

本专题按老师原声的实际讲解顺序整理。口头重复、停顿和现场操作被改写为连贯文章，但保留重要类比、代码解释、易错点和纠错过程。

每节包含原创 SVG 概念图；图片中不含视频分集、课程来源或平台标识。

## 学习方法

1. 先读“老师讲解整理成文章”，理解为什么要学。
2. 再沿原创图复述数据流。
3. 不看输出先运行或手算最小代码。
4. 完成自测，再进入下一节。

## 一、普通 RNN：结构、形状与代码

1. [RNN 简介：让当前判断带上过去的信息](./01-rnn-introduction.md)（原视频 P38）
2. [RNN 分类：输入输出关系与内部结构是两个维度](./02-rnn-types.md)（原视频 P39）
3. [RNN 模型结构：公式、共享权重与张量形状](./03-rnn-structure.md)（原视频 P40）
4. [RNN 基础代码：创建层、准备输入、运行并验形状](./04-rnn-basic-code.md)（原视频 P41）
5. [修改句长：只应改变 output 的时间维](./05-change-sequence-length.md)（原视频 P42）
6. [修改隐藏层与总结：维度代表模型的记忆容量](./06-change-hidden-size.md)（原视频 P43）

## 二、LSTM、双向与 GRU

7. [LSTM 图解（上）：遗忘门与输入门管理长期记忆](./07-lstm-diagram-part1.md)（原视频 P44）
8. [LSTM 图解（下）：输出门产生当前隐藏状态](./08-lstm-diagram-part2.md)（原视频 P45）
9. [Bi-LSTM：从前后两个方向理解同一位置](./09-bidirectional-lstm.md)（原视频 P46）
10. [LSTM 代码：多一个细胞状态，接口如何变化](./10-lstm-code.md)（原视频 P47）
11. [GRU 图解：两扇门合并记忆管理](./11-gru-diagram.md)（原视频 P48）
12. [GRU 代码：替换循环层并验证接口](./12-gru-code.md)（原视频 P49）

## 三、全球姓名分类项目

13. [姓名分类需求：从名字预测 18 个国家类别](./13-name-classification-requirement.md)（原视频 P50）
14. [全局字母表与国家名：固定输入列和输出列](./14-alphabet-and-countries.md)（原视频 P51）
15. [读取数据：把姓名与国家分别保存并处理异常行](./15-read-name-data.md)（原视频 P52）
16. [Dataset：把变长姓名转成字符 One-Hot 张量](./16-dataset.md)（原视频 P53）
17. [DataLoader：为变长姓名组织训练迭代](./17-dataloader.md)（原视频 P54）
18. [LogSoftmax：旧式 NLLLoss 与现代 CrossEntropyLoss 的关系](./18-log-softmax.md)（原视频 P55）
19. [RNN 分类模型：取最后时间步映射到 18 类](./19-rnn-model.md)（原视频 P56）
20. [测试 RNN：用随机输入把形状链走通](./20-test-rnn.md)（原视频 P57）
21. [搭建 LSTM 与 GRU 模型：复用分类头，隔离状态差异](./21-lstm-gru-models.md)（原视频 P58）
22. [测试三种模型：用同一数据管道公平验形状](./22-test-three-models.md)（原视频 P59）
23. [训练 RNN：外层 epoch、内层 batch 与五步反向传播](./23-train-rnn.md)（原视频 P60）
24. [训练 LSTM：复用训练循环，正确管理 h 与 c](./24-train-lstm.md)（原视频 P61）
25. [训练 GRU：同一协议下比较速度与效果](./25-train-gru.md)（原视频 P62）
26. [可视化三模型：损失、时间和准确率要一起看](./26-visualize-comparison.md)（原视频 P63）
27. [RNN 预测：姓名转张量、加载权重并取 Top-k](./27-rnn-prediction.md)（原视频 P64）
28. [姓名分类总结：按单模型跑通，再统一重构](./28-name-classifier-summary.md)（原视频 P65）

## 配套代码

- [rnn_from_scratch 配套实现](../../rnn_from_scratch/README.md)
- 测试命令：`python3 -m unittest rnn_from_scratch.test_model -v`


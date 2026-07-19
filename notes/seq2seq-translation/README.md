# GRU Seq2Seq 英法翻译：26 节零基础文章笔记

本专题按老师原声的实际讲解顺序整理。口头重复、停顿和现场操作被改写为连贯文章，但保留重要类比、代码解释、易错点和纠错过程。

每节包含原创 SVG 概念图；图片中不含视频分集、课程来源或平台标识。

## 学习方法

1. 先读“老师讲解整理成文章”，理解为什么要学。
2. 再沿原创图复述数据流。
3. 不看输出先运行或手算最小代码。
4. 完成自测，再进入下一节。

## 一、需求、设备与数据管道

1. [英译法需求：先看懂 800 行项目的六个模块](./01-translation-requirements.md)（原视频 P80）
2. [CUDA 环境（上）：GPU、驱动、工具包与 PyTorch 不是同一层](./02-cuda-concepts.md)（原视频 P81）
3. [CUDA 环境实操：创建环境、安装、验证与排错](./03-cuda-practice.md)（原视频 P82）
4. [CUDA 配置总结：把可复现信息写进项目](./04-cuda-summary.md)（原视频 P83）
5. [数据清洗：规范文本，但不要改坏翻译含义](./05-data-cleaning.md)（原视频 P84）
6. [数据预处理：建两套词表并加入 SOS/EOS](./06-preprocessing.md)（原视频 P85）
7. [构建 Dataset：一条样本同时返回源 ID 与目标 ID](./07-dataset.md)（原视频 P86）
8. [获取 DataLoader：补齐、长度和 mask 一起产出](./08-dataloader.md)（原视频 P87）

## 二、Encoder 与两类 Decoder

9. [GRU Encoder：Embedding 后保留每个时间步输出](./09-gru-encoder.md)（原视频 P88）
10. [测试 Encoder：先验形状再运行](./10-test-encoder.md)（原视频 P89）
11. [无 Attention Decoder 思路：只靠 final hidden 生成](./11-plain-decoder-plan.md)（原视频 P90）
12. [构建无 Attention GRU Decoder](./12-plain-decoder-code.md)（原视频 P91）
13. [测试无 Attention Decoder：逐词循环与 EOS](./13-test-plain-decoder.md)（原视频 P92）
14. [有 Attention Decoder 思路：每步重新查询源句](./14-attention-decoder-plan.md)（原视频 P93）
15. [有 Attention Decoder 代码（上）：定义层与接口](./15-attention-decoder-code-part1.md)（原视频 P94）
16. [有 Attention Decoder 代码（下）：逐行完成 forward_step](./16-attention-decoder-code-part2.md)（原视频 P95）
17. [测试 Attention Decoder：权重、mask 与单步输出](./17-test-attention-decoder.md)（原视频 P96）
18. [模型搭建总结：三个模块如何对接](./18-model-summary.md)（原视频 P97）

## 三、Teacher Forcing、训练与预测

19. [Teacher Forcing：训练时有时喂真值上一词](./19-teacher-forcing.md)（原视频 P98）
20. [模型训练（单批次）：先把一次前向和损失走通](./20-train-one-batch.md)（原视频 P99）
21. [view()：只改观察形状，不改元素顺序](./21-view-function.md)（原视频 P100）
22. [完整训练代码：epoch、验证、保存与日志](./22-full-training.md)（原视频 P101）
23. [训练总结：把 800 行压缩成一条可复述主线](./23-training-summary.md)（原视频 P102）
24. [模型预测代码：无真值时逐词生成](./24-prediction-code.md)（原视频 P103）
25. [预测代码测试：从样例翻译发现数据与模型问题](./25-prediction-test.md)（原视频 P104）
26. [绘制张量图：用图检查模块连接，不把图当性能证明](./26-tensorboard-graph.md)（原视频 P105）

## 配套代码

- [seq2seq_from_scratch 配套实现](../../seq2seq_from_scratch/README.md)
- 测试命令：`python3 -m unittest seq2seq_from_scratch.test_model -v`


# 注意力机制与 Seq2Seq 框架：14 节零基础文章笔记

本专题按老师原声的实际讲解顺序整理。口头重复、停顿和现场操作被改写为连贯文章，但保留重要类比、代码解释、易错点和纠错过程。

每节包含原创 SVG 概念图；图片中不含视频分集、课程来源或平台标识。

## 学习方法

1. 先读“老师讲解整理成文章”，理解为什么要学。
2. 再沿原创图复述数据流。
3. 不看输出先运行或手算最小代码。
4. 完成自测，再进入下一节。

## 一、注意力直觉与 Q/K/V

1. [注意力机制介绍：把有限精力分给更相关的信息](./01-attention-introduction.md)（原视频 P66）
2. [Q、K、V：问题、索引与实际内容](./02-qkv-introduction.md)（原视频 P67）
3. [注意力实现步骤：算分、归一化、加权求和](./03-attention-steps.md)（原视频 P68）

## 二、在 Seq2Seq 中逐步定位信息

4. [Seq2Seq 任务：编码器把输入交给解码器逐词生成](./04-seq2seq-task.md)（原视频 P69）
5. [Seq2Seq 加入注意力：每个目标词拥有自己的 context](./05-seq2seq-with-attention.md)（原视频 P70）
6. [普通 Encoder-Decoder：一个 C 服务所有解码步骤](./06-plain-encoder-decoder.md)（原视频 P71）
7. [带注意力 Encoder-Decoder：从公式看 Cᵢ 怎样生成](./07-attentive-encoder-decoder.md)（原视频 P72）
8. [注意力概率分布：Decoder 状态如何与所有 Encoder 状态比较](./08-attention-probabilities.md)（原视频 P73）

## 三、类型、公式、bmm 与完整代码

9. [软注意力、硬注意力与自注意力](./09-attention-types.md)（原视频 P74）
10. [常见注意力计算规则：拼接式、加法式与缩放点积](./10-attention-scoring-rules.md)（原视频 P75）
11. [bmm：一次完成一批三维矩阵乘法](./11-bmm.md)（原视频 P76）
12. [注意力代码实现：从 Q/K/V 到增强后的查询](./12-attention-code.md)（原视频 P77）
13. [注意力测试代码：检查形状、概率和与梯度](./13-attention-test.md)（原视频 P78）
14. [参数解释：把 1×32、32×64、1×96 全部读成语义](./14-attention-parameters.md)（原视频 P79）

## 配套代码

- [attention_from_scratch 配套实现](../../attention_from_scratch/README.md)
- 测试命令：`python3 -m unittest attention_from_scratch.test_model -v`


# FastText 文本分类：11 节零基础文章笔记

本专题按老师原声的实际讲解顺序整理。口头重复、停顿和现场操作被改写为连贯文章，但保留重要类比、代码解释、易错点和纠错过程。

每节包含原创 SVG 概念图；图片中不含视频分集、课程来源或平台标识。

## 学习方法

1. 先读“老师讲解整理成文章”，理解为什么要学。
2. 再沿原创图复述数据流。
3. 不看输出先运行或手算最小代码。
4. 完成自测，再进入下一节。

## 一、原理：模型为什么快

1. [FastText 简介与环境：为什么一行 API 也值得学原理](./01-fasttext-introduction.md)（原视频 P144）
2. [层次 Softmax 与哈夫曼树：从全部类别改成走一条路径](./02-hierarchical-softmax-huffman.md)（原视频 P145）
3. [负采样：正样本之外，只挑少量“陪练”负样本](./03-negative-sampling.md)（原视频 P146）

## 二、基线与数据

4. [直接训练：数据格式、预测、测试与第一版基线](./04-direct-training.md)（原视频 P147）
5. [数据预处理：统一大小写、分开标点，并保持训练预测一致](./05-data-preprocessing.md)（原视频 P148）

## 三、调优与多标签

6. [训练轮数与学习率：一个决定看几遍，一个决定每步走多远](./06-epochs-learning-rate.md)（原视频 P149）
7. [word N-gram 与损失函数：补一点局部词序，再选合适输出方式](./07-ngrams-loss.md)（原视频 P150）
8. [自动调参：给验证集和时间预算，让程序寻找更好组合](./08-autotune.md)（原视频 P151）
9. [多分类多标签：OVA、k 与 threshold 的完整含义](./09-multilabel-ova.md)（原视频 P152）

## 四、保存与迁移

10. [保存与加载模型：训练一次，部署和复现实验反复使用](./10-save-load-model.md)（原视频 P153）
11. [词向量迁移：把别人训练好的语义坐标系拿来起步](./11-pretrained-vectors-transfer.md)（原视频 P154）

## 配套代码

- [FastText 原理配套练习包](../../fasttext_from_scratch/README.md)
- 测试命令：`python -m unittest discover -s fasttext_from_scratch -p 'test_*.py'`


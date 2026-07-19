# 第 11 节：无 Attention Decoder 思路：只靠 final hidden 生成

> 笔记编号 11/26 · 对应原视频 P90 · [打开这一集](https://www.bilibili.com/video/BV14mdfBDE4Q?p=90)

[← 上一节：10 测试 Encoder：先验形状再运行](./10-test-encoder.md) · [返回总目录](./README.md) · [下一节：12 构建无 Attention GRU Decoder →](./12-plain-decoder-code.md)

## 这节解决什么问题

无注意力版本怎样工作，它为什么是理解有注意力版本的对照组？

![第 11 节原创概念图](./diagrams/11-concept.svg)

图从左向右读。先跟着数据或推理过程走一遍，再学习下面的术语。

## 辅助流程图

```mermaid
flowchart LR
    N0["输入上一个目标 token"]
    N1["目标 Embedding"]
    N2["与 hidden 送 GRU"]
    N3["Linear→目标词表"]
    N4["预测下一 token"]
    N0 --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
```

### Seq2Seq 模块 UML

```mermaid
classDiagram
    class EncoderGRU {
      source_embedding
      gru
      forward(source_ids)
    }
    class AttentionDecoderGRU {
      target_embedding
      attention
      gru
      classifier
      forward_step(token, hidden, encoder_outputs)
    }
    class Seq2Seq {
      teacher_forcing_ratio
      forward(source, target)
      greedy_decode(source)
    }
    Seq2Seq *-- EncoderGRU
    Seq2Seq *-- AttentionDecoderGRU
```

## 老师原声整理稿（按讲解顺序）

### 0:00–2:58　单步 Decoder

每步输入一个目标 token ID 和上一 hidden；Embedding 后经 GRU，Linear 输出目标词表 logits。

### 2:58–5:19　信息来源的限制

源句只通过 Encoder final hidden 进入 Decoder。长句所有细节都压在一个状态里，这是固定 C 瓶颈。

## 完整原声逐段记录

[查看本节按时间戳整理的完整音轨转写](./transcripts/p090.md)

逐段记录用于核查老师讲解是否遗漏；正文会进一步纠正口误和语音识别中的技术术语。

## 零基础先记住

- Decoder 每步只预测一个词
- 输出维等于目标词表大小
- 无注意力只接 final hidden

## 最小可运行代码

下面代码默认从项目根目录运行；专题配套实现见 [seq2seq_from_scratch 配套实现](../../seq2seq_from_scratch/README.md)。

```python
print("token + hidden → GRU → Linear(target_vocab)")
```

### 输入和输出怎么看

显示无注意力单步的最短数据流。

## 最容易踩的坑

目标 Embedding 的词表大小必须用法语词表，而非英语词表。

## 本节知识链

`输入上一个目标 token → 目标 Embedding → 与 hidden 送 GRU → Linear→目标词表 → 预测下一 token`

## 自测

**问题：无注意力 Decoder 从哪里得到源句信息？**

<details>
<summary>点开核对答案</summary>

Encoder 的最终 hidden。

</details>

## 学完检查

- [ ] 我能用自己的话复述老师的讲解顺序
- [ ] 我能在运行前预测关键输出或张量形状
- [ ] 我知道这节方法最容易用错的地方
- [ ] 我能独立回答自测题

[← 上一节：10 测试 Encoder：先验形状再运行](./10-test-encoder.md) · [返回总目录](./README.md) · [下一节：12 构建无 Attention GRU Decoder →](./12-plain-decoder-code.md)

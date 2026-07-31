# 第 3 节：BERT 预训练任务：MLM 与 NSP 怎样共同制造监督

> 笔记编号 3/6 · 对应原视频 P186 · [打开这一集](https://www.bilibili.com/video/BV14mdfBDE4Q?p=186)

[← 上一节：2 BERT 架构：三种 Embedding、Encoder 堆叠与关键形状](./02-bert-architecture.md) · [返回总目录](./README.md) · [下一节：4 BERT 总结：MLM/NSP 复盘，以及 GLUE 与 CLUE 公共评测 →](./04-bert-summary.md)

## 这节解决什么问题

没有人工标签的大规模文本，BERT 从哪里得到可计算的训练目标？

![第 3 节原创概念图](./diagrams/03-concept.svg)

图从左向右读。先跟着数据或推理过程走一遍，再学习下面的术语。

## 辅助流程图

```mermaid
flowchart LR
    N0["原始连续文本"]
    N1["随机遮罩 token"]
    N2["构造真假句对"]
    N3["同时预测词与句间关系"]
    N4["联合损失更新 Encoder"]
    N0 --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
```

### BERT 从输入到预训练目标

```mermaid
flowchart LR
    A["文本或句对"] --> B["Token + Position + Segment Embedding"]
    B --> C["多层双向 Transformer Encoder"]
    C --> D["每个 token 的上下文表示 [B,L,H]"]
    D --> E["MLM：还原被遮罩 token"]
    D --> F["NSP：判断句 B 是否紧接句 A"]
```

### BERT 一批预训练数据的时序

```mermaid
sequenceDiagram
    participant D as 原始句对
    participant C as MLM/NSP 数据构造
    participant B as BertForPreTraining
    participant M as MLM Loss
    participant N as NSP Loss
    D->>C: 连续文本
    C-->>B: input_ids / masks / segments / 两类 labels
    B-->>M: prediction_logits [B,L,V]
    B-->>N: seq_relationship_logits [B,2]
    M-->>B: MLM loss
    N-->>B: NSP loss
    B-->>B: 总损失反向传播
```

## 真正看懂自监督：标签其实藏在原文里

假设原文是：

```text
小明 今天 去 学校
```

MLM 数据构造程序选中“学校”作为本次预测位置。送进模型的输入可能变成：

```text
input_ids: 小明 今天 去 [MASK]
labels:    -100 -100 -100 学校
```

`labels` 中的 `学校` 实际保存的是它的 token ID；`-100` 是常见实现里的忽略值，表示这些位置不参与交叉熵。模型输出 `[B,L,V]`，但本例只在最后一个位置的 V 个词表分数上计算 loss。原词来自原始文本，所以不需要人工逐条标注。

### 15% 与 80/10/10 是两层抽样

经典 BERT 做法不是把 15% 的词全部换成 `[MASK]`：

1. 先从全部 token 中选约 15% 作为**需要预测的位置**；
2. 对这些已选位置，约 80% 换 `[MASK]`，约 10% 换随机词，约 10% 保持原词；
3. 无论表面上怎样替换，这些位置的 label 都是替换前的原 token。

所以“保持原词”的 10% 仍会产生 loss；“没有被选中”的 85% 才是不监督的位置。比例是经典 BERT 的训练配方，不是 MLM 定义本身，其他模型可以采用不同遮罩策略。

### NSP 再看一个正负样本

原文连续两句：

```text
A：外面开始下雨。
B：小明拿了一把伞。
```

正样本可构造成 `[CLS] A [SEP] B [SEP]`，标签为 `IsNext`。负样本则把 B 换成另一篇文档的随机句子，标签为 `NotNext`。模型通常读取 `[CLS]` 的最终表示，通过二分类头输出 `[B,2]`。

NSP 的限制也由这个例子暴露出来：若随机负句主题差得太远，模型只比较主题就可能答对，不一定真的学到句间连续关系。因此后续模型可以改进负样本或取消 NSP；这不影响 MLM 的基本思想。

### 两个 loss 如何共同更新一个 Encoder

设本批 `L_MLM=2.1`、`L_NSP=0.4`，经典未加权写法得到总损失 `2.5`。反向传播分别计算两项任务通过共享 Encoder 产生的梯度，并在共享参数处相加。优化器随后根据合成梯度更新参数。Loss 负责量化错误，反向传播负责求梯度，优化器才执行更新——三者不要混成一句“loss 自动学习”。

## 老师原声整理稿（按讲解顺序）

### 0:00–6:30　MLM：从原文自己造标签

随机选取一部分 token 作为预测目标，常见 BERT 方案约 15%。选中位置中约 80% 换 `[MASK]`、10% 换随机 token、10% 保持原 token；labels 保存原 token ID，其他位置设忽略值。模型输出 `[B,L,V]`，只在选中位置算词表交叉熵。

### 6:30–12:30　为什么不能把输入全遮住

模型需要上下文才能推断目标；全部遮罩会丢失句意。只替换 MASK 又会造成预训练看到特殊符号、下游不看到的差距，因此混入随机词和原词。经典比例是设计选择，不是自然定律。

### 12:30–18:30　NSP：句 B 是否真接句 A

从文档取句 A；一半用真实下一句 B，一半换随机句，标签常为 IsNext/NotNext。输入 `[CLS] A [SEP] B [SEP]`，句对头输出 `[B,2]`。NSP 希望学习句间关系，但负样本过简单会让模型走主题匹配捷径。后续 RoBERTa 等研究说明去掉 NSP 也能很好，因此不要把 NSP 当成所有 BERT 类模型不可缺少的真理。

### 18:30–24:00　联合损失

这里先说明公式解决什么问题：同一 Encoder 要同时学 token 级恢复和句对级关系。经典训练把两项损失相加：`L = L_MLM + L_NSP`（也可带权）。反向传播让共享 Encoder 同时服务两个目标。预训练完成后通常丢弃这两个头，换上下游任务头。

## 完整原声逐段记录

[查看本节按时间戳整理的完整音轨转写](./transcripts/p186.md)

逐段记录用于核查老师讲解是否遗漏；正文会进一步纠正口误和语音识别中的技术术语。

## 零基础先记住

- MLM 与 NSP 标签都可从原始文本自动构造
- MLM 只监督被选中的位置
- NSP 并非所有后续模型都保留

## 最小可运行代码

下面代码是帮助理解本节概念的最小示例，默认从项目根目录运行。

```python
from transformers import AutoModelForPreTraining
model=AutoModelForPreTraining.from_pretrained("your-bert-pretraining-checkpoint")
out=model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    token_type_ids=batch.get("token_type_ids"),
    labels=mlm_labels,
    next_sentence_label=nsp_labels,
)
print(out.loss,out.prediction_logits.shape,out.seq_relationship_logits.shape)
```

### 输入和输出怎么看

得到总 loss、MLM `[B,L,V]` logits 和 NSP `[B,2]` logits。

## 最容易踩的坑

把 MLM labels 设成遮罩后的输入 ID，而不是遮罩前的真实 ID。

## 本节知识链

`原始连续文本 → 随机遮罩 token → 构造真假句对 → 同时预测词与句间关系 → 联合损失更新 Encoder`

## 自测

**问题：BERT 预训练为什么不需要人工逐条标注？**

<details>
<summary>点开核对答案</summary>

MLM 的原词和 NSP 的句子顺序都能从原始连续文本自动构造监督信号。

</details>

## 学完检查

- [ ] 我能用自己的话复述老师的讲解顺序
- [ ] 我能在运行前预测关键输出或张量形状
- [ ] 我知道这节方法最容易用错的地方
- [ ] 我能独立回答自测题

[← 上一节：2 BERT 架构：三种 Embedding、Encoder 堆叠与关键形状](./02-bert-architecture.md) · [返回总目录](./README.md) · [下一节：4 BERT 总结：MLM/NSP 复盘，以及 GLUE 与 CLUE 公共评测 →](./04-bert-summary.md)

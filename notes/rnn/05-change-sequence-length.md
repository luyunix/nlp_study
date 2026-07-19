# 第 5 节：修改句长：只应改变 output 的时间维

> 笔记编号 5/28 · 对应原视频 P42 · [打开这一集](https://www.bilibili.com/video/BV14mdfBDE4Q?p=42)

[← 上一节：4 RNN 基础代码：创建层、准备输入、运行并验形状](./04-rnn-basic-code.md) · [返回总目录](./README.md) · [下一节：6 修改隐藏层与总结：维度代表模型的记忆容量 →](./06-change-hidden-size.md)

## 这节解决什么问题

序列从 3 步改为更多步时，哪些张量维度应该变化？

![第 5 节原创概念图](./diagrams/05-concept.svg)

图从左向右读。先跟着数据或推理过程走一遍，再学习下面的术语。

## 辅助流程图

```mermaid
flowchart LR
    N0["改 seq_len"]
    N1["重建 input"]
    N2["复用 RNN"]
    N3["比较 output"]
    N4["确认 hn 不带时间维"]
    N0 --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
```

### PyTorch 循环层的张量形状

```mermaid
flowchart LR
    X["input<br/>[batch, seq, input_size]"] --> R["RNN / LSTM / GRU"]
    H0["h0<br/>[layers×directions, batch, hidden]"] --> R
    C0["仅 LSTM：c0<br/>形状同 h0"] -.-> R
    R --> O["output<br/>每个时间步<br/>[batch, seq, directions×hidden]"]
    R --> HN["hn<br/>每层最终状态"]
    R -.-> CN["仅 LSTM：cn"]
```

## 老师原声整理稿（按讲解顺序）

### 0:00–1:53　只改一个变量做对照

老师把输入的句子长度改长，其他 input_size、hidden_size、batch 和层数不动，然后要求运行前先猜形状。

### 1:53–3:26　观察结果

output 要为每个时间步保留一个隐藏输出，所以时间维随句长改变；h_n 只保存各层最终状态，不保留整条时间轴，因此形状不变。这个小实验是后面“取最后一步做分类”的基础。

## 完整原声逐段记录

[查看本节按时间戳整理的完整音轨转写](./transcripts/p042.md)

逐段记录用于核查老师讲解是否遗漏；正文会进一步纠正口误和语音识别中的技术术语。

## 零基础先记住

- seq_len 决定 output 的时间维
- h_n 不包含完整时间轴
- 控制变量实验比背结果可靠

## 最小可运行代码

下面代码默认从项目根目录运行；专题配套实现见 [rnn_from_scratch 配套实现](../../rnn_from_scratch/README.md)。

```python
import torch
rnn = torch.nn.RNN(5, 6, batch_first=True)
for length in (3, 7):
    out, hn = rnn(torch.randn(2, length, 5))
    print(length, out.shape, hn.shape)
```

### 输入和输出怎么看

output 从 [2,3,6] 变 [2,7,6]；h_n 始终 [1,2,6]。

## 最容易踩的坑

真实 batch 中句长不同还需 padding/packing；这里只是所有样本同时改长。

## 本节知识链

`改 seq_len → 重建 input → 复用 RNN → 比较 output → 确认 hn 不带时间维`

## 自测

**问题：为什么 h_n 不需要 seq_len 这一维？**

<details>
<summary>点开核对答案</summary>

它定义上就是每层、每方向的最终状态，而不是所有时间步。

</details>

## 学完检查

- [ ] 我能用自己的话复述老师的讲解顺序
- [ ] 我能在运行前预测关键输出或张量形状
- [ ] 我知道这节方法最容易用错的地方
- [ ] 我能独立回答自测题

[← 上一节：4 RNN 基础代码：创建层、准备输入、运行并验形状](./04-rnn-basic-code.md) · [返回总目录](./README.md) · [下一节：6 修改隐藏层与总结：维度代表模型的记忆容量 →](./06-change-hidden-size.md)

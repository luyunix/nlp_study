# 第 5 节：Seq2Seq 加入注意力：每个目标词拥有自己的 context

> 笔记编号 5/14 · 对应原视频 P70 · [打开这一集](https://www.bilibili.com/video/BV14mdfBDE4Q?p=70)

[← 上一节：4 Seq2Seq 任务：编码器把输入交给解码器逐词生成](./04-seq2seq-task.md) · [返回总目录](./README.md) · [下一节：6 普通 Encoder-Decoder：一个 C 服务所有解码步骤 →](./06-plain-encoder-decoder.md)

## 这节解决什么问题

固定 C 为什么不够，C₁、C₂、C₃ 又怎样随解码步变化？

![第 5 节原创概念图](./diagrams/05-concept.svg)

图从左向右读。先跟着数据或推理过程走一遍，再学习下面的术语。

## 辅助流程图

```mermaid
flowchart LR
    N0["编码器保存 h1…hL"]
    N1["当前解码状态作 Q"]
    N2["算本步 weights"]
    N3["得到本步 C_t"]
    N4["预测目标 token"]
    N0 --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
```

### Encoder、Attention、Decoder 的模块关系

```mermaid
classDiagram
    class Encoder {
      输入 token 序列
      输出每个位置 h1...hL
    }
    class Attention {
      Decoder状态作为Q
      Encoder状态作为K和V
      输出context
    }
    class Decoder {
      上一步输出
      上一步隐藏状态
      context
      预测下一个token
    }
    Encoder --> Attention
    Attention --> Decoder
    Decoder --> Attention : 下一时间步更新Q
```

### 注意力的三步主流程

```mermaid
flowchart LR
    Q["Q：当前想找什么"] --> S["Q 与每个 K 算分"]
    K["K：每份内容的检索标签"] --> S
    S --> P["Softmax：变成总和为 1 的权重"]
    P --> W["按权重汇总 V"]
    V["V：真正要取的内容"] --> W
    W --> C["context：本次专属信息"]
```

## 真正看懂动态 context：同一源句算三次权重

源句是 `welcome / to / Wuhan`。Encoder 为三个位置分别产生 `h₁、h₂、h₃`。假设 Decoder 要依次生成 `欢迎 / 来 / 武汉`，每一步的权重可能是：

| 解码步 | 当前要生成 | 对 welcome | 对 to | 对 Wuhan | 本步 context |
|---|---|---:|---:|---:|---|
| 1 | 欢迎 | 0.75 | 0.15 | 0.10 | `0.75h₁+0.15h₂+0.10h₃` |
| 2 | 来 | 0.20 | 0.60 | 0.20 | `0.20h₁+0.60h₂+0.20h₃` |
| 3 | 武汉 | 0.05 | 0.10 | 0.85 | `0.05h₁+0.10h₂+0.85h₃` |

这些数字只是便于手算的示意，不是课程实验结果。真正的权重由模型训练得到。每行经过 Softmax 后和为 1，但不要求只有一个位置非零：翻译短语、调换语序或消解歧义时，一个目标词可能同时依赖多个源词。

### 为什么每一步的权重会变

源句的 `K/V` 在一次翻译中通常不变，变化的是 Decoder 状态，也就是 Q：

```text
第 1 步 Q₁：只知道 SOS，寻找第一个目标词需要的信息
第 2 步 Q₂：已经生成“欢迎”，寻找接下来需要的信息
第 3 步 Q₃：已经生成“欢迎 来”，寻找地点信息
```

Q 变化，`score(Q,Kᵢ)` 就变化，Softmax 权重和 context 也随之变化。这里的 context 不是“把最相关词复制出来”，而是对所有 V 做加权求和后的连续向量。

### 形状上发生了什么

若 `B=2、源长 L=3、隐藏维 H=4`：

```text
Encoder states / K / V: [2,3,4]
当前 Decoder query Q:  [2,4]
本步 weights:          [2,3]
本步 context:          [2,4]
```

到下一个目标时间步时，再产生一份新的 `[2,3]` 权重。若一共生成 `T` 步，把各步权重保存起来，就得到 `[2,T,3]` 的注意力矩阵；热力图的每一行对应一个目标位置，每一列对应一个源位置。

## 老师原声整理稿（按讲解顺序）

### 0:00–4:58　固定 C 的问题

普通 Encoder-Decoder 每次预测都使用同一个 C，等于认为所有源词对所有目标词贡献相同。长句信息被挤进一个向量。

### 4:58–9:58　动态 C_t

加入注意力后，第 1、2、3 个目标词分别使用 C_1、C_2、C_3。每个 C_t 都由当前解码状态与编码器各位置重新打分得到。

### 9:58–14:57　翻译对齐直觉

生成“欢迎”时 welcome 权重大，生成“来”时 to 权重大；但仍会给其他词小权重以处理短语、一词多义和语序变化。

### 14:57–18:03　计算闭环

Decoder 状态→Q；Encoder 状态→K/V；Softmax 得权重；加权 V 得 C_t；Decoder 用 C_t 和历史生成新词，下一步状态再形成新 Q。

## 完整原声逐段记录

[查看本节按时间戳整理的完整音轨转写](./transcripts/p070.md)

逐段记录用于核查老师讲解是否遗漏；正文会进一步纠正口误和语音识别中的技术术语。

## 零基础先记住

- 每个解码步都有独立 context
- 注意力是软对齐，不要求一一对应
- 当前 Decoder 状态决定本步查询

## 最小可运行代码

下面代码默认从项目根目录运行；专题配套实现见 [attention_from_scratch 配套实现](../../attention_from_scratch/README.md)。

```python
contexts=["C_欢迎","C_来","C_武汉"]
for step,c in enumerate(contexts,1): print(step,c)
```

### 输入和输出怎么看

三个目标时间步明确使用不同 context。

## 最容易踩的坑

注意力不要求源词数与目标词数相同。

## 本节知识链

`编码器保存 h1…hL → 当前解码状态作 Q → 算本步 weights → 得到本步 C_t → 预测目标 token`

## 自测

**问题：为什么生成下一个词时权重会变化？**

<details>
<summary>点开核对答案</summary>

Decoder 状态和已生成历史变化，使 Q 变化。

</details>

## 学完检查

- [ ] 我能用自己的话复述老师的讲解顺序
- [ ] 我能在运行前预测关键输出或张量形状
- [ ] 我知道这节方法最容易用错的地方
- [ ] 我能独立回答自测题

[← 上一节：4 Seq2Seq 任务：编码器把输入交给解码器逐词生成](./04-seq2seq-task.md) · [返回总目录](./README.md) · [下一节：6 普通 Encoder-Decoder：一个 C 服务所有解码步骤 →](./06-plain-encoder-decoder.md)

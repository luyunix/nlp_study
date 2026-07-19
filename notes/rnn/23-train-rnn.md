# 第 23 节：训练 RNN：外层 epoch、内层 batch 与五步反向传播

> 笔记编号 23/28 · 对应原视频 P60 · [打开这一集](https://www.bilibili.com/video/BV14mdfBDE4Q?p=60)

[← 上一节：22 测试三种模型：用同一数据管道公平验形状](./22-test-three-models.md) · [返回总目录](./README.md) · [下一节：24 训练 LSTM：复用训练循环，正确管理 h 与 c →](./24-train-lstm.md)

## 这节解决什么问题

一轮训练中数据、损失、梯度和参数到底按什么顺序流动？

![第 23 节原创概念图](./diagrams/23-concept.svg)

图从左向右读。先跟着数据或推理过程走一遍，再学习下面的术语。

## 辅助流程图

```mermaid
flowchart LR
    N0["取一批 x/y"]
    N1["forward"]
    N2["计算 NLLLoss"]
    N3["zero_grad/backward/step"]
    N4["累计指标并保存"]
    N0 --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
```

### 一批数据的训练时序

```mermaid
sequenceDiagram
    participant D as DataLoader
    participant M as 循环模型
    participant L as 损失函数
    participant O as 优化器
    D->>M: x 与 label
    M->>L: log_probs 与 label
    O->>O: zero_grad()
    L->>M: backward()
    O->>M: step() 更新参数
```

## 老师原声整理稿（按讲解顺序）

### 0:00–7:19　训练函数输入与准备

老师定义训练轮数、模型、损失函数、优化器，并准备记录损失、准确率和耗时。模型末尾为 LogSoftmax，因此使用 NLLLoss。

### 7:51–15:59　双重循环

外层控制 epoch，内层遍历 DataLoader。每批取 x/y，初始化状态或由模型内部处理，然后前向得到 18 类 log-probabilities。

### 16:11–27:14　反向传播固定顺序

先 optimizer.zero_grad() 清掉上批梯度；loss.backward() 计算梯度；optimizer.step() 更新参数。若漏清零，PyTorch 默认会累积梯度。

### 27:14–37:14　损失与准确率统计

每批损失累加；argmax 预测与标签比较得到正确数。epoch 平均损失应按样本或批次口径明确，不能混算。进度条只负责显示。

### 37:14–44:38　保存模型与返回曲线数据

训练结束保存 state_dict，并返回损失、准确率、耗时等用于对比。保存目录要存在，模型结构配置也应一并记录。

## 完整原声逐段记录

[查看本节按时间戳整理的完整音轨转写](./transcripts/p060.md)

逐段记录用于核查老师讲解是否遗漏；正文会进一步纠正口误和语音识别中的技术术语。

## 零基础先记住

- zero_grad→backward→step 顺序固定
- 训练指标口径要明确
- 保存 state_dict 还需保存模型配置

## 最小可运行代码

下面代码默认从项目根目录运行；专题配套实现见 [rnn_from_scratch 配套实现](../../rnn_from_scratch/README.md)。

```python
# 训练核心顺序
optimizer.zero_grad()
logp = model(x)
loss = criterion(logp, y)
loss.backward()
optimizer.step()
```

### 输入和输出怎么看

每批先算预测和损失，再反传并更新一次参数。

## 最容易踩的坑

不要在 forward 前忘记 model.train()，评估时要 model.eval()。

## 本节知识链

`取一批 x/y → forward → 计算 NLLLoss → zero_grad/backward/step → 累计指标并保存`

## 自测

**问题：为什么每批都要 zero_grad？**

<details>
<summary>点开核对答案</summary>

PyTorch 梯度默认累积，不清零会把多批梯度叠在一起。

</details>

## 学完检查

- [ ] 我能用自己的话复述老师的讲解顺序
- [ ] 我能在运行前预测关键输出或张量形状
- [ ] 我知道这节方法最容易用错的地方
- [ ] 我能独立回答自测题

[← 上一节：22 测试三种模型：用同一数据管道公平验形状](./22-test-three-models.md) · [返回总目录](./README.md) · [下一节：24 训练 LSTM：复用训练循环，正确管理 h 与 c →](./24-train-lstm.md)

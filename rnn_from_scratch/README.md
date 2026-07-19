# RNN、LSTM、GRU 与人名分类配套代码

这组代码服务于课程 P38–P65 的文章笔记，重点是让输入、隐藏状态和输出形状容易观察。

包含：

- 按公式手写的单步 `SimpleRNNCell`
- 可切换 `RNN`、`LSTM`、`GRU` 的 `NameClassifier`
- 双向 LSTM 输出维度检查
- LogSoftmax 概率性质测试

运行测试：

```bash
python3 -m pip install torch
python3 -m unittest rnn_from_scratch.test_model -v
```

这些代码只演示循环网络主干。完整人名分类数据管道、变长序列打包、训练循环和预测流程会在对应逐节文章中展开。

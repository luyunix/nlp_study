# GRU Seq2Seq 英法翻译配套代码

这组代码服务于 P80–P105 的文章笔记，采用统一的 `batch_first=True` 形状：

- `EncoderGRU`：词 ID → Embedding → GRU 全部隐藏状态
- `AttentionDecoderGRU`：当前查询对 Encoder 输出做点积注意力
- `Seq2Seq.forward`：支持 Teacher Forcing 的训练前向
- `Seq2Seq.greedy_decode`：逐词贪心预测并返回注意力矩阵

运行测试：

```bash
python3 -m unittest seq2seq_from_scratch.test_model -v
```

实现刻意保持小而清晰，用于理解数据流和形状；实际翻译系统还要加入词表持久化、长度打包、验证集、beam search、指标与检查点管理。

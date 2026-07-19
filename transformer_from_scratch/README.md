# 从零实现 Transformer

本目录是第 111–143 节课程代码的清理、校准和整合版本。实现采用 PyTorch，并使用较易训练的 Pre-LN 残差结构。

## 包含模块

```text
Embeddings
PositionalEncoding
subsequent_mask
attention
MultiHeadedAttention
PositionwiseFeedForward
LayerNorm
SublayerConnection
EncoderLayer / Encoder
DecoderLayer / Decoder
Generator
EncoderDecoder
make_model
```

## 运行测试

```bash
python3 -m pip install torch
python3 -m unittest transformer_from_scratch.test_model -v
```

## 最小使用示例

```python
import torch

from transformer_from_scratch import (
    make_model,
    subsequent_mask,
)

model = make_model(
    src_vocab=1000,
    tgt_vocab=1200,
    n=2,
    d_model=64,
    d_ff=128,
    h=8,
    dropout=0.1,
)

src = torch.randint(0, 1000, (2, 6))
tgt = torch.randint(0, 1200, (2, 5))
src_mask = torch.ones(2, 1, 6, dtype=torch.bool)
tgt_mask = subsequent_mask(5)

log_probs = model(src, tgt, src_mask, tgt_mask)
print(log_probs.shape)  # [2, 5, 1200]
```

这里只验证架构前向传播。若要训练翻译模型，还需要数据管道、padding mask、目标右移、损失函数、优化器、学习率调度和自回归解码。

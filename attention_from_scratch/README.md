# 注意力机制配套代码

这组代码服务于课程 P66–P79，并会在英译法 Seq2Seq 专题中继续复用。

核心步骤：

```text
query 与 keys 计算匹配分数
→ mask 屏蔽无效位置
→ Softmax 得到概率
→ 概率对 values 加权求和
→ context
```

实现包含：

- Dot-Product Attention
- Luong General Attention
- `torch.bmm` 的批量矩阵乘法
- mask、概率和与输出形状测试

运行：

```bash
python3 -m pip install torch
python3 -m unittest attention_from_scratch.test_model -v
```

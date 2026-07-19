# 文本预处理配套代码

这组代码把课程中的关键数据变换写成了无第三方依赖、可单步调试的
Python 函数，适合配合笔记理解：

- 词表构建与 One-hot
- CBOW / Skip-Gram 训练样本构造
- n-gram 特征
- 余弦相似度
- 截断与补齐
- 标签分布与词频统计

运行测试：

```bash
python3 -m unittest text_preprocessing_from_scratch.test_core -v
```

课程中的 jieba、FastText、PyTorch Embedding、pandas 和词云等第三方库
示例会保留在对应逐节笔记中；这里不把库的内部实现伪装成几行教学代码。

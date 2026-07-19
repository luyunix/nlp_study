# FastText 原理配套练习包

这是一组只依赖 Python 标准库的小练习，用于把专题中的三个抽象概念变成可运行代码：

- `data.py`：清洗文本、生成和解析 `__label__` 行；
- `huffman.py`：构造哈夫曼编码并计算带权路径长度；
- `sampling.py`：从正样本以外按权重抽取少量负样本。

它不是对官方 FastText 的重写，也不负责真正训练分类模型。文章中的 `fasttext.train_supervised` 示例需要另行安装 FastText。

运行测试：

```bash
python -m unittest discover -s fasttext_from_scratch -p 'test_*.py'
```

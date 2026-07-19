import fs from "node:fs";
import path from "node:path";

const root = "/Users/lyn/Documents/nlp_study";
const outDir = path.join(root, "notes", "text-preprocessing");
const diagramDir = path.join(outDir, "diagrams");
fs.mkdirSync(diagramDir, { recursive: true });

const lessons = [
  {
    p: 5, slug: "overview", title: "文本预处理全景：先看清问题，再动手处理",
    plain: "原始文字不能直接、稳定地送进模型。我们要先把文字切成有意义的单位，再变成数字，检查数据有没有偏斜，最后才做特征加工和数据增强。",
    points: ["基础方法：分词、词性标注、命名实体识别", "数值表示：One-Hot、Word2Vec、Embedding", "语料分析：标签、长度、词频；特征与增强：n-gram、相似度、回译"],
    code: `pipeline = ["分词", "数字化", "语料分析", "特征处理", "数据增强"]\nfor step, name in enumerate(pipeline, 1):\n    print(step, name)`,
    output: "程序会按顺序打印 5 个阶段。真正项目不一定每一步都用，但“先分析、后决定”不能省。",
    pit: "预处理不是越多越好。比如情感分类中删掉“没”“不”，会把意思反过来；应让任务决定规则。",
    quiz: "为什么不能看到文本很长，就直接统一截成 128 个词？", answer: "因为还没看长度分布；128 可能截掉大量关键信息，也可能浪费绝大多数计算。",
    diagram: ["原始文本", "基础处理", "数字表示", "分析与特征", "可训练数据"]
  },
  {
    p: 6, slug: "environment-and-tokenization", title: "环境准备与分词：机器眼中的句子没有天然词界",
    plain: "英文常用空格提示词界，中文通常连在一起。分词就是决定“南京市长江大桥”应切成哪些词，让后续统计和建模有稳定单位。",
    points: ["课程使用 Python 3.10、jieba 0.42.1；版本不同可能有小差异", "虚拟环境把本项目依赖与其他项目隔离", "token（词元）是模型处理的基本单位，不一定等于日常所说的“词”"],
    code: `# 安装一次：pip install jieba==0.42.1\nimport jieba\ntext = "我正在学习自然语言处理"\nprint(list(jieba.cut(text)))`,
    output: "输入是一整句中文；输出是若干词元组成的列表。具体边界由词典和统计规则共同决定。",
    pit: "不要把“分词结果”当绝对真理。同一句话在搜索、命名实体识别和语言模型中可能需要不同粒度。",
    quiz: "虚拟环境解决的是分词准确率问题吗？", answer: "不是。它解决依赖隔离和版本复现问题；准确率仍取决于算法、词典和语料。",
    diagram: ["连续中文", "分词器", "词元边界", "词元列表"]
  },
  {
    p: 7, slug: "jieba-precise-mode", title: "jieba 精确模式：给句子一条主要切分路径",
    plain: "精确模式尽量把句子切成一组互不重叠、适合一般文本分析的词。它是最常用的默认模式。",
    points: ["jieba.cut(text, cut_all=False) 返回生成器", "jieba.lcut(text) 直接返回列表", "生成器节省内存，但通常只能从头到尾消费一次"],
    code: `import jieba\ntext = "小明硕士毕业于中国科学院计算所"\nwords = jieba.lcut(text, cut_all=False)\nprint(words)\nprint("/".join(words))`,
    output: "输入句子被切成一条主要路径；第二行用 / 显示边界，便于人工检查。",
    pit: "把 cut 的结果打印出来可能只看到 generator 地址。需要 list(...)，或直接使用 lcut。",
    quiz: "为什么同一个生成器第二次 list(...) 可能为空？", answer: "生成器是惰性的一次性迭代器，第一次遍历已经把其中元素消费完了。",
    diagram: ["一句中文", "精确模式", "唯一主路径", "词元序列"]
  },
  {
    p: 8, slug: "jieba-full-mode", title: "jieba 全模式：把可能的词尽量找出来",
    plain: "全模式会扫描句子中所有可能成词的片段，因此召回多、速度快，但会产生互相重叠甚至互相冲突的词。",
    points: ["调用 jieba.lcut(text, cut_all=True)", "适合快速枚举候选词，不适合作为严格的句子切分结果", "它回答“可能包含哪些词”，不是“唯一该怎么切”"],
    code: `import jieba\ntext = "南京市长江大桥"\nprint("精确：", jieba.lcut(text, cut_all=False))\nprint("全模式：", jieba.lcut(text, cut_all=True))`,
    output: "比较两行输出：全模式一般包含更多且重叠的候选，如“南京”“南京市”“长江”“大桥”等。",
    pit: "全模式词多不代表更准确。把重叠候选直接按顺序拼回去，已经不再是原句的一次切分。",
    quiz: "做普通词频统计时，为什么通常先选精确模式？", answer: "词频需要每个位置尽量只归属一个词；全模式的重叠会重复计数。",
    diagram: ["原句字符", "全模式扫描", "多个重叠候选", "高召回/有歧义"]
  },
  {
    p: 9, slug: "jieba-search-mode", title: "jieba 搜索引擎模式：长词再拆一层，提高召回",
    plain: "搜索模式先做较准确的切分，再把较长的词补充拆成短词。用户只搜到其中一部分时，也更容易命中文档。",
    points: ["使用 jieba.lcut_for_search(text)", "兼顾长词语义与短词检索入口", "典型目标是提高搜索召回率，而非生成最干净的语言序列"],
    code: `import jieba\ntext = "小明毕业于中国科学院计算所"\nprint(jieba.lcut(text))\nprint(jieba.lcut_for_search(text))`,
    output: "搜索模式通常在精确结果基础上补充较短片段；长词和短词可能同时出现。",
    pit: "搜索模式的输出也可能重叠。训练序列模型时直接使用，会把原始位置关系弄复杂。",
    quiz: "电商搜索为什么希望“华为手机壳”还能被“手机壳”命中？", answer: "用户查询常只包含长词的一部分；补充短词索引可以减少漏检。",
    diagram: ["长词文本", "精确切分", "长词二次拆分", "建立多粒度索引"]
  },
  {
    p: 10, slug: "traditional-chinese", title: "繁体中文分词：接口相同，词典覆盖决定效果",
    plain: "jieba 可以直接接收繁体中文字符串，调用方法不变。真正需要关注的是所用词典是否覆盖领域词和地区用语。",
    points: ["Python 3 字符串天然支持 Unicode", "简繁转换不是所有任务的必选步骤", "保留原文还是统一字形，要看检索、分类或生成任务的目标"],
    code: `import jieba\ntext = "我來到北京清華大學"\nprint(jieba.lcut(text, cut_all=False))\nprint(jieba.lcut_for_search(text))`,
    output: "输入繁体文本，仍得到词元列表；不同版本词典可能导致个别切分不同。",
    pit: "机械简繁转换可能合并本来不同的用词，或损伤人名、地名。转换前应保留原始字段并抽样检查。",
    quiz: "看到繁体文本，第一步必须转成简体吗？", answer: "不一定。若工具已支持、任务需要保留原貌，就可以直接处理；统一字形应是有目的的选择。",
    diagram: ["繁体文本", "Unicode 字符串", "jieba 词典匹配", "繁体词元"]
  },
  {
    p: 11, slug: "custom-dictionary", title: "自定义词典：教分词器认识你的领域词",
    plain: "通用词典不认识公司名、产品名或专业术语时，我们可以补一份用户词典，让这些片段更倾向于作为整体出现。",
    points: ["每行格式：词语 词频 词性；只有词语必填", "jieba.load_userdict(path) 加载文件", "词频越高，算法越倾向保留整体，但不是无限制越高越好"],
    code: `import jieba\n# userdict.txt 示例：创新办 10000 n\nprint(jieba.lcut("创新办主任正在开会"))\njieba.add_word("创新办", freq=10000, tag="n")\nprint(jieba.lcut("创新办主任正在开会"))`,
    output: "比较添加词语前后结果；后者更可能把“创新办”保留成一个词。",
    pit: "词频不是实际业务出现次数的严格统计值，而是参与分词概率的权重。乱设超大值可能破坏其他句子的边界。",
    quiz: "用户词典中的词性可以省略吗？", answer: "可以；词语必填，频率和词性通常可选。若后续依赖词性过滤，建议认真填写。",
    diagram: ["领域文本", "通用词典漏词", "加入词/频率/词性", "边界得到修正"]
  },
  {
    p: 12, slug: "ner-and-pos", title: "命名实体识别与词性标注：词是什么角色",
    plain: "分词只告诉我们边界；词性标注告诉我们它像名词、动词还是形容词；命名实体识别进一步找人名、地名、机构名等现实对象。",
    points: ["jieba.posseg 可同时分词并给出词性标记", "NER 与 POS 有关联但不是同一任务", "规则/词典系统会误判，领域数据和上下文模型能继续改善"],
    code: `import jieba.posseg as pseg\nfor item in pseg.lcut("李明在北京大学学习人工智能"):\n    print(item.word, item.flag)`,
    output: "每行是“词语 + 词性代码”。代码如 n、v、a 等应查表理解，不必死背所有标记。",
    pit: "不能因为某词被标成地名就认定它一定是实体。课程中的示例也会出现专名误判，这正说明结果要验证。",
    quiz: "“苹果”是水果还是公司，单靠固定词典能总是判断对吗？", answer: "不能。它依赖上下文；NER 往往需要结合整句语义。",
    diagram: ["句子", "分词", "词性角色", "人/地/机构实体"]
  },
  {
    p: 13, slug: "text-to-tensor", title: "文本张量表示：模型为什么只接收数字",
    plain: "神经网络做的是加法、乘法和求导，不能直接计算“天气”这两个字。文本表示就是建立稳定规则，把词元映射成数字向量。",
    points: ["先建立去重词表，再给每个词固定编号", "One-Hot 向量长度等于词表大小，只有对应位置为 1", "Word2Vec 与 Embedding 用较短的稠密向量表达词"],
    code: `tokens = ["我", "爱", "自然语言", "我"]\nvocab = {word: i for i, word in enumerate(dict.fromkeys(tokens))}\nprint(vocab)\nprint([vocab[word] for word in tokens])`,
    output: "先得到词到编号的映射，再把原序列变为整数序列；重复词必须得到同一编号。",
    pit: "编号 7 不表示这个词比编号 3“大”。ID 只是地址，不能直接当连续数值特征。",
    quiz: "词表有 10,000 个词，One-Hot 每个词向量多长？", answer: "10,000 维，其中仅一个位置为 1，其余为 0。",
    diagram: ["词元", "固定词表", "整数 ID", "向量/张量"]
  },
  {
    p: 14, slug: "one-hot-generation", title: "One-Hot 生成：从词表位置得到独热向量",
    plain: "给每个词一个唯一列号。看到某词，就创建全零数组，再把它对应的那一列设为 1。",
    points: ["训练集和验证集必须共用同一词表", "映射要保存，预测时不能重新按新顺序编号", "课程展示旧版 Keras Tokenizer；理解原理比绑定旧 API 更重要"],
    code: `from text_preprocessing_from_scratch.core import one_hot_corpus\nvocab, matrix = one_hot_corpus(["我", "爱", "NLP", "我"])\nprint(vocab)\nfor row in matrix:\n    print(row)`,
    output: "输入 4 个词、3 个不同词，输出形状为 4×3 的矩阵；两个“我”的行完全相同。",
    pit: "先对训练集建词表，再冻结。若把测试集也用于建词表，会造成数据泄漏。",
    quiz: "为什么要把 tokenizer/词表持久化保存？", answer: "为了保证训练和以后预测时，同一个词始终落在同一个位置。",
    diagram: ["去重词表", "词→列号", "全零向量", "对应列置 1"]
  },
  {
    p: 15, slug: "one-hot-usage", title: "One-Hot 使用：加载同一映射并处理未知词",
    plain: "模型上线后必须加载训练时的词表。新句子中的已知词按旧位置编码，没见过的词要有明确的 OOV（未知词）策略。",
    points: ["保存的是映射规则，不只是某批向量", "常设置 <UNK> 接住词表外词", "生产中还需 <PAD> 表示补齐位置"],
    code: `vocab = {"<UNK>": 0, "我": 1, "爱": 2, "NLP": 3}\ndef encode(words):\n    return [vocab.get(word, vocab["<UNK>"]) for word in words]\nprint(encode(["我", "爱", "新词"]))`,
    output: "结果为 [1, 2, 0]；“新词”没有报错，而是安全映射到 <UNK>。",
    pit: "直接写 vocab[word] 遇到新词会 KeyError。更糟的是临时把新词追加进词表，模型的输入维度和含义都会变化。",
    quiz: "<UNK> 和 <PAD> 能共用一个编号吗？", answer: "通常不要。一个表示未知语义，一个表示没有内容；模型往往需要区分它们。",
    diagram: ["保存的词表", "新句子", "已知词查表", "未知词→UNK"]
  },
  {
    p: 16, slug: "simple-one-hot", title: "手写 One-Hot：看见稀疏表示的优点和代价",
    plain: "手写一次能彻底看懂：One-Hot 不学习语义，它只是一个不会混淆的身份牌。",
    points: ["优点：简单、确定、容易解释", "缺点：词表大时维度巨大且绝大多数为 0", "任意两个不同词的 One-Hot 距离相同，无法表达语义相近"],
    code: `words = ["猫", "狗", "汽车"]\nvocab = {w: i for i, w in enumerate(words)}\nfor word in words:\n    vector = [0] * len(vocab)\n    vector[vocab[word]] = 1\n    print(word, vector)`,
    output: "猫、狗、汽车各占一列。虽然猫与狗更相近，向量本身并没有体现这一点。",
    pit: "One-Hot 不是“落后的错误方法”。小型类别特征仍很合适；只是大型词表的语义表示通常需要 Embedding。",
    quiz: "词表从 1 万扩大到 100 万，One-Hot 的主要问题是什么？", answer: "每个向量维度同步变成 100 万，存储和计算极其稀疏且不含语义关系。",
    diagram: ["词的身份", "唯一列", "大量零", "无语义距离"]
  },
  {
    p: 17, slug: "word2vec-cbow", title: "Word2Vec 的 CBOW：用上下文猜中间词",
    plain: "把一句话切成许多小窗口，遮住中间词，让模型根据左右邻居猜它。长期训练后，经常出现在相似上下文里的词会得到相近向量。",
    points: ["训练样本形式：(上下文词集合 → 中心词)", "输入经过共享的隐层权重，再预测整个词表概率", "训练完成后通常取隐层权重矩阵作为词向量"],
    code: `from text_preprocessing_from_scratch.core import cbow_examples\ntokens = "我 爱 自然 语言 处理".split()\nfor context, target in cbow_examples(tokens, window_size=1):\n    print(context, "->", target)`,
    output: "例如 ('爱','语言') → '自然'。这些只是训练样本；真正 Word2Vec 会用大量语料学习权重。",
    pit: "CBOW 不是把相邻词简单拼接成最终词向量；相邻预测任务只是学习表示的训练信号。",
    quiz: "窗口为 1 时，“我 爱 NLP”中以“爱”为中心的 CBOW 输入和标签是什么？", answer: "输入是上下文（我，NLP），标签是中心词“爱”。",
    diagram: ["左邻词 + 右邻词", "共享隐层", "词表概率", "预测中心词"]
  },
  {
    p: 18, slug: "word2vec-skipgram", title: "Word2Vec 的 Skip-Gram：用中间词猜周围词",
    plain: "Skip-Gram 与 CBOW 方向相反：给一个中心词，分别预测窗口里的各个上下文词。一处中心会产生多对训练样本。",
    points: ["样本形式：(中心词 → 一个上下文词)", "同一中心对应多个损失，再共同更新权重", "计算通常更多，但对低频词往往更有帮助"],
    code: `from text_preprocessing_from_scratch.core import skipgram_examples\ntokens = "我 爱 自然 语言 处理".split()\nfor center, context in skipgram_examples(tokens, window_size=1):\n    print(center, "->", context)`,
    output: "“爱”会分别产生 爱→我、爱→自然；序列边缘只有一侧邻居。",
    pit: "不要把 Skip-Gram 理解为跳着取词的 n-gram。它是一个预测任务，skip 指由中心预测窗口上下文。",
    quiz: "CBOW 和 Skip-Gram 最核心的方向差异是什么？", answer: "CBOW：上下文→中心；Skip-Gram：中心→上下文。",
    diagram: ["中心词", "共享隐层", "左上下文", "右上下文"]
  },
  {
    p: 19, slug: "fasttext-setup", title: "FastText 准备：从大语料到可训练文件",
    plain: "FastText 能从普通文本语料学习词向量。训练前最重要的不是立刻调参，而是确认语料编码、分词质量、体量和文件路径。",
    points: ["课程使用预处理后的百科语料并切分大文件", "安装 fasttext 可能涉及本机编译环境", "流程分为：准备数据、训练保存、加载评估、调参"],
    code: `from pathlib import Path\ncorpus = Path("wiki_sample.txt")\ncorpus.write_text("自然 语言 处理 很 有趣\\n机器 学习 需要 数据\\n", encoding="utf-8")\nprint(corpus.read_text(encoding="utf-8"))`,
    output: "FastText 无监督训练文件通常一行一段已分词文本，词之间由空格分开。",
    pit: "不要把未切分的整句中文直接当作一个“词”。这会让词表几乎全是句子，学不到期望的词向量。",
    quiz: "语料越大就一定越好吗？", answer: "不一定。大量乱码、重复或领域错位文本会放大噪声；质量与任务匹配同样重要。",
    diagram: ["原始语料", "清洗与分词", "UTF-8 文本", "FastText 输入"]
  },
  {
    p: 20, slug: "fasttext-training", title: "FastText 训练与保存：让语料自己产生监督信号",
    plain: "无监督训练不需要人工标签。窗口中的词互相充当预测目标，模型边猜边调整向量；训练结束要保存模型，才能复用同一空间。",
    points: ["fasttext.train_unsupervised(input, model='skipgram')", "常见默认概念：dim=100、epoch=5、lr=0.05（以安装版本文档为准）", "日志关注进度、学习率、平均损失和预计剩余时间"],
    code: `# 需要：pip install fasttext\nimport fasttext\nmodel = fasttext.train_unsupervised(\n    "wiki_sample.txt", model="skipgram", dim=50, epoch=5, lr=0.05\n)\nmodel.save_model("wiki_skipgram.bin")`,
    output: "输入分词语料，输出二进制模型文件。该文件不仅含词向量，还含生成词向量所需的模型参数。",
    pit: "一次训练成功不代表向量就好。小样本仅适合跑通流程，语义质量需要足够且匹配领域的语料。",
    quiz: "为什么训练后必须保存模型而不能只保存几个查询结果？", answer: "未来要查询任意词、做近邻或继续使用统一向量空间，需要完整模型或完整向量表。",
    diagram: ["分词语料", "窗口预测", "反向更新", "保存 .bin 模型"]
  },
  {
    p: 21, slug: "fasttext-evaluation", title: "FastText 加载、查看与评估：向量好不好要验证",
    plain: "加载模型后可以取单词向量，也可以查最近邻。最近邻看起来合理是直观检查，但正式项目还应有与任务相关的量化评估。",
    points: ["fasttext.load_model(path) 恢复模型", "get_word_vector(word) 返回长度为 dim 的 NumPy 向量", "get_nearest_neighbors(word) 返回（相似度，词）列表"],
    code: `import fasttext\nmodel = fasttext.load_model("wiki_skipgram.bin")\nvector = model.get_word_vector("语言")\nprint(vector.shape)\nfor score, word in model.get_nearest_neighbors("语言", k=5):\n    print(round(score, 3), word)`,
    output: "先看到向量形状，再看到若干相近词及分数。分数高只表示向量空间相近，不等于事实关系成立。",
    pit: "只挑一个成功案例容易自我欺骗。应准备固定词表、类比/相似度数据或下游任务指标批量评估。",
    quiz: "最近邻都是常见词，是否足以证明模型适合情感分类？", answer: "不足。它只做内在检查；还需在情感分类验证集上比较实际效果。",
    diagram: ["加载模型", "查询词向量", "计算近邻", "人工+任务评估"]
  },
  {
    p: 22, slug: "fasttext-hyperparameters", title: "FastText 超参数：每个旋钮改变什么",
    plain: "调参不是盲目把数字变大，而是理解资源、语料与目标之间的交换：更高维、更久训练通常更贵，也更可能过拟合噪声。",
    points: ["model：cbow 更快，skipgram 常更照顾低频词", "dim 控制向量容量和存储；epoch 控制遍历语料次数", "lr 控制每次更新幅度；thread 控制并行线程数"],
    code: `config = {\n    "model": "skipgram",\n    "dim": 100,\n    "epoch": 10,\n    "lr": 0.05,\n    "thread": 4,\n}\nfor key, value in config.items():\n    print(f"{key:>6} = {value}")`,
    output: "这段先把实验配置明确打印出来。真实训练应连同随机种子、语料版本和评估结果一起记录。",
    pit: "每次同时改 5 个参数，结果变好也不知道原因。初学时固定其他项，一次只比较一个关键变量。",
    quiz: "维度从 100 提到 1000，为什么不保证效果提升？", answer: "容量和成本都增加，但数据可能不足、噪声可能被拟合，下游任务也未必需要这么多维度。",
    diagram: ["语料与任务", "模型/维度", "轮次/学习率", "速度与质量权衡"]
  },
  {
    p: 23, slug: "embedding-vs-word2vec", title: "Word2Vec 与 Embedding：预训练方法和查表层的区别",
    plain: "Embedding 广义上指把离散对象变成稠密向量；在 PyTorch 中也特指一张可训练查找表。Word2Vec 是用上下文预测任务预训练这张向量表的一类方法。",
    points: ["Word2Vec 向量可先在大语料学好，再放入下游模型", "nn.Embedding 通常随当前任务的损失继续更新", "预训练提供起点，任务内训练提供针对性；两者可以结合"],
    code: `import torch\nlayer = torch.nn.Embedding(num_embeddings=5, embedding_dim=3)\nids = torch.tensor([1, 3, 1])\nvectors = layer(ids)\nprint(vectors.shape)\nprint(torch.equal(vectors[0], vectors[2]))`,
    output: "输出形状是 [3, 3]；相同 ID 查到相同向量。训练后权重会被梯度更新。",
    pit: "课程把未知词近似为“找已知最近词”容易误解。真正 FastText 可用字符 n-gram 为 OOV 词组合向量，这是它的重要特点。",
    quiz: "nn.Embedding 会自动理解“猫”和“狗”相近吗？", answer: "不会。初始只是参数表，必须通过训练目标或加载预训练权重才逐渐形成语义。",
    diagram: ["离散词 ID", "Embedding 查表", "稠密向量", "随任务更新"]
  },
  {
    p: 24, slug: "embedding-lookup", title: "Embedding 取词向量：从句子到三维张量",
    plain: "先分词并映射成 ID，再把每个 ID 当作行号去 Embedding 权重表中取一行。批量句子最终常形成“批次×长度×维度”的张量。",
    points: ["num_embeddings 是词表行数，embedding_dim 是每个词的向量长度", "输入形状可以是任意 ID 网格，输出会在末尾多一个 embedding 维", "同一个 ID 在同一次权重状态下查到同一行"],
    code: `import torch\nvocab = {"我": 0, "爱": 1, "NLP": 2}\nids = torch.tensor([[vocab["我"], vocab["爱"], vocab["NLP"]]])\nembedding = torch.nn.Embedding(len(vocab), 4)\nprint("输入：", ids.shape)\nprint("输出：", embedding(ids).shape)`,
    output: "输入形状 [1, 3]，输出形状 [1, 3, 4]：1 个句子、3 个词、每词 4 维。",
    pit: "词表编号必须从 0 到 num_embeddings-1；任何越界 ID 都会报错。",
    quiz: "输入是 [32, 20]，embedding_dim=128，输出形状是什么？", answer: "[32, 20, 128]，分别是批量、序列长度和词向量维度。",
    diagram: ["句子分词", "词→ID", "按行查权重表", "B×L×D 张量"]
  },
  {
    p: 25, slug: "embedding-visualization", title: "Embedding 可视化：把高维空间投影到屏幕",
    plain: "词向量通常有几十到几百维，肉眼看不了。TensorBoard Projector 用 PCA、t-SNE 等方法投影到 2D/3D，帮助观察聚类和离群点。",
    points: ["SummaryWriter.add_embedding 写入向量和词标签", "TensorBoard 读取 runs 日志目录并在浏览器展示", "投影会丢失信息，只适合探索，不是最终质量证明"],
    code: `import torch\nfrom torch.utils.tensorboard import SummaryWriter\nwords = ["猫", "狗", "汽车"]\nvectors = torch.randn(3, 8)\nwriter = SummaryWriter("runs/words")\nwriter.add_embedding(vectors, metadata=words, tag="demo")\nwriter.close()\n# 终端运行：tensorboard --logdir=runs`,
    output: "浏览器打开通常是 http://localhost:6006，在 Projector 中可切换投影方法并搜索词。",
    pit: "若张量需要梯度，直接 .numpy() 会报错；先 tensor.detach().cpu().numpy()。另外，二维近邻不一定等于原高维近邻。",
    quiz: "投影图中两个词靠得近，能断言它们在原空间也最近吗？", answer: "不能。降维会扭曲距离，需要回到原向量计算余弦相似度验证。",
    diagram: ["高维词向量", "PCA/t-SNE 投影", "2D/3D 点", "观察后再量化验证"]
  },
  {
    p: 26, slug: "label-distribution", title: "标签数量分布：先发现类别不平衡",
    plain: "分类前先数每类有多少样本。如果 99% 都是正类，一个永远猜正类的模型也有 99% 准确率，却几乎没有用。",
    points: ["Pandas 读取 TSV 时明确 sep='\\t'", "countplot 或 value_counts 展示每类数量", "不平衡时同时关注 precision、recall、F1、混淆矩阵"],
    code: `from text_preprocessing_from_scratch.core import label_distribution\nlabels = ["正面", "正面", "负面", "正面", "中性"]\nprint(label_distribution(labels))`,
    output: "输出每个标签的样本数。真实数据还应计算比例，并分别查看训练/验证/测试集。",
    pit: "不要为了“平衡”就随意删除大量多数类。可考虑分层划分、类别权重、重采样，并用验证集比较。",
    quiz: "正类 950、负类 50，恒猜正类准确率多少？", answer: "95%。这说明只看 accuracy 会掩盖模型完全识别不了负类的问题。",
    diagram: ["读取标签", "分类计数", "发现不平衡", "调整评估/训练策略"]
  },
  {
    p: 27, slug: "map-function", title: "map：把同一个函数依次用到每条数据",
    plain: "map 像一条流水线：左边不断来元素，中间执行同一个函数，右边才产生结果。它默认惰性计算，所以通常要 list 才能直接查看。",
    points: ["map(function, iterable) 返回迭代器", "lambda 适合非常短的一次性函数", "Pandas Series 也可用 .map 或 .apply，但语义稍有差别"],
    code: `sentences = ["我爱NLP", "文本预处理很重要", "短句"]\nlengths = map(len, sentences)\nprint(list(lengths))\nprint(list(lengths))`,
    output: "第一行输出长度列表；第二行是空列表，因为同一个 map 迭代器已经被消费。",
    pit: "惰性迭代器节省内存，但重复使用前要转成列表，或重新创建 map。",
    quiz: "map 会在创建的一刻把所有结果算完吗？", answer: "不会。通常在迭代取值时才逐个计算，这叫惰性求值。",
    diagram: ["可迭代数据", "统一函数", "惰性 map", "消费后得到结果"]
  },
  {
    p: 28, slug: "sentence-length-distribution", title: "句子长度分布：为截断和补齐找依据",
    plain: "先测每条文本多长，再看直方图和分位数。目标长度应覆盖大多数样本，同时避免被少数极长异常值拖得太大。",
    points: ["长度可以按字符、词元或子词计算，必须与模型输入单位一致", "直方图看整体形状，分位数给出可操作阈值", "Seaborn 的 distplot 已弃用，可用 histplot(kde=True)"],
    code: `lengths = [5, 8, 8, 10, 12, 13, 20, 100]\nlengths = sorted(lengths)\ndef percentile(p):\n    i = round((len(lengths) - 1) * p)\n    return lengths[i]\nprint("中位数", percentile(0.5))\nprint("约95分位", percentile(0.95))`,
    output: "少数 100 的长文本不会迫使所有样本都补到 100；可结合 95/99 分位与任务损失选择长度。",
    pit: "用 Python len(中文字符串) 得到字符数，不是 jieba 词数，更不是模型 tokenizer 的子词数。",
    quiz: "为什么通常不直接用数据集中最大长度？", answer: "最大值常是极少数异常样本，会造成大量 padding、浪费显存和计算。",
    diagram: ["计算每条长度", "直方图/分位数", "检查长尾", "选择目标长度"]
  },
  {
    p: 29, slug: "length-by-label", title: "按标签比较长度：长度本身也可能泄露规律",
    plain: "把正负样本的长度分别画出来，可以看到某类是否明显更长、异常点集中在哪类，也能发现采集流程带来的偏差。",
    points: ["stripplot 能展示每个样本点，箱线图可概括中位数与四分位", "长度差异可能是真实规律，也可能是数据采集漏洞", "极长、极短样本要回到原文人工复查"],
    code: `samples = [("正", 8), ("正", 10), ("正", 80), ("负", 7), ("负", 9)]\nfor label in sorted({x[0] for x in samples}):\n    values = [length for y, length in samples if y == label]\n    print(label, "数量", len(values), "平均长度", sum(values) / len(values))`,
    output: "按类输出数量和平均长度；正类中的 80 会明显拉高平均值，提示进一步检查。",
    pit: "看到两类长度不同，不能立即删除长度信息。先判断它是有意义的语言信号还是标签泄漏。",
    quiz: "若所有正样本来自长评论、负样本来自短标题，模型可能学到什么捷径？", answer: "只按长度猜标签，而不是理解语义；换数据来源后性能会崩。",
    diagram: ["文本+标签", "分组长度散点", "发现差异/离群", "回看原文原因"]
  },
  {
    p: 30, slug: "itertools-chain", title: "chain：把嵌套词列表铺平成一个词流",
    plain: "每句话分词后得到一个列表，整个语料就是“列表的列表”。chain.from_iterable 能把它惰性串起来，便于做全局词频。",
    points: ["itertools.chain(a, b, c) 依次遍历多个可迭代对象", "chain.from_iterable(nested) 适合列表的列表", "它不复制全部数据，但仍是一次性迭代器"],
    code: `from itertools import chain\nsentences = [["我", "爱", "NLP"], ["我", "学", "分词"]]\nwords = chain.from_iterable(sentences)\nprint(list(words))\nprint(list(words))`,
    output: "第一次得到一个平坦词列表，第二次为空。大语料中可直接把它交给 Counter，避免额外复制。",
    pit: "不要用 sum(nested_lists, []) 铺平超大数据，它会反复复制列表，效率很差。",
    quiz: "chain 会把字符串当一个整体还是逐字符遍历？", answer: "字符串本身也是可迭代对象，会逐字符遍历；所以输入结构要确认正确。",
    diagram: ["句1词列表", "句2词列表", "chain 串接", "单一词流"]
  },
  {
    p: 31, slug: "vocabulary-count", title: "词汇量统计：不同词有多少、各出现几次",
    plain: "把全语料词流交给 Counter，可以同时得到词频和词汇表。词汇量不是越大越好，低频噪声会让模型和词表膨胀。",
    points: ["总词数包含重复，词汇量只数不同词", "Counter.most_common 查看高频词", "可设置最低频率并为其余词保留 <UNK>"],
    code: `from collections import Counter\nwords = "我 爱 NLP 我 爱 分词".split()\ncounts = Counter(words)\nprint("总词数", sum(counts.values()))\nprint("词汇量", len(counts))\nprint("高频词", counts.most_common())`,
    output: "总词数为 6，词汇量为 4；“我”和“爱”各出现 2 次。",
    pit: "在分割数据前用全量语料统计词表会窥见测试集。严格实验应只用训练集建立词表和频率阈值。",
    quiz: "总词数 100 万、不同词 5 万，Embedding 表有多少行？", answer: "通常约 5 万行，再加 PAD、UNK 等特殊词；不是 100 万行。",
    diagram: ["全语料词流", "Counter 计数", "总词数≠词汇量", "频率过滤词表"]
  },
  {
    p: 32, slug: "adjective-wordcloud", title: "形容词词云：先按词性筛选，再按频率画图",
    plain: "词云把出现次数映射成字体大小。若我们只关心评价用语，可先用词性标注筛出形容词，再统计频率。",
    points: ["jieba.posseg 产生词和词性，常按 a 开头标记筛形容词", "Counter 计算频率，WordCloud.generate_from_frequencies 绘制", "中文词云必须指定可显示中文的字体文件"],
    code: `from collections import Counter\nimport jieba.posseg as pseg\ntexts = ["画面非常漂亮", "操作简单但是速度很慢"]\nadjectives = []\nfor text in texts:\n    adjectives.extend(w.word for w in pseg.lcut(text) if w.flag.startswith("a"))\nprint(Counter(adjectives))`,
    output: "先得到形容词及次数；安装 wordcloud 后可把这个 Counter 传给 generate_from_frequencies。",
    pit: "词云适合探索和展示，不适合精确比较。字体面积会误导，重要结论要配词频表或条形图。",
    quiz: "为什么中文词云常出现方框？", answer: "默认字体没有中文字形；需要提供有效的中文 font_path。",
    diagram: ["评论分词+词性", "筛选形容词", "Counter 频率", "字号映射词云"]
  },
  {
    p: 33, slug: "zip-function", title: "zip：把多个等长位置对齐成元组",
    plain: "zip 像拉拉链：第一个序列的第 1 项和第二个序列的第 1 项合在一起，然后继续下一位置。n-gram 的简洁写法正利用了这种对齐。",
    points: ["zip(a, b) 返回惰性迭代器", "默认在最短输入结束处停止", "解包 zip(*pairs) 可把成对数据重新拆列"],
    code: `words = ["我", "爱", "NLP"]\nlabels = ["代词", "动词"]\npairs = zip(words, labels)\nprint(list(pairs))`,
    output: "只输出两对，第三个词被静默忽略，因为 labels 更短。",
    pit: "长度不等时 zip 不报错，容易悄悄丢数据。Python 3.10+ 可用 zip(a, b, strict=True) 让不等长时报错。",
    quiz: "zip([1,2,3], ['a']) 会产生几项？", answer: "1 项：(1, 'a')；默认由最短序列决定长度。",
    diagram: ["序列 A", "序列 B", "同位置拉链", "元组流/最短截止"]
  },
  {
    p: 34, slug: "n-gram", title: "n-gram：保留连续局部顺序的文本特征",
    plain: "只数单词会丢掉顺序；n-gram 把连续 n 个词或字符绑成一个特征。“不 好”作为 bigram 能比两个孤立词更清楚地表达否定。",
    points: ["unigram=1 个单位，bigram=2 个，trigram=3 个", "滑动窗口每次向右移动一格，不能跳词", "词 n-gram 与字 n-gram 是不同粒度，需按任务选择"],
    code: `from text_preprocessing_from_scratch.core import ngrams\ntokens = ["这个", "电影", "不", "好看"]\nprint("bigram:", ngrams(tokens, 2))\nprint("1~2 gram:", ngrams(tokens, 2, include_lower_orders=True))`,
    output: "bigram 有 3 个连续词对；include_lower_orders=True 会同时保留 4 个 unigram。",
    pit: "课程示例若用 set 去重，会丢失出现次数和原顺序。做计数特征时通常应保留列表，再交给 Counter。",
    quiz: "长度为 L 的序列有多少个连续 n-gram（L≥n）？", answer: "L-n+1 个，因为窗口起点可从 0 移到 L-n。",
    diagram: ["词元序列", "长度 n 窗口", "逐格右移", "局部顺序特征"]
  },
  {
    p: 35, slug: "text-similarity", title: "文本相似度：先对齐特征，再比较方向或集合",
    plain: "两段文字先要放进同一个特征空间，再计算相似程度。词频向量常用余弦相似度；只关心共有特征时也可用 Jaccard。",
    points: ["词表取两段文本特征的并集，向量各列含义才一致", "余弦比较方向，对不同长度文本通常比原始欧氏距离稳健", "Jaccard=交集大小/并集大小，忽略频次时很直观"],
    code: `from text_preprocessing_from_scratch.core import cosine_similarity\n# 同一词表：[机器, 学习, 很, 有趣]\na = [1, 1, 1, 1]\nb = [1, 1, 0, 0]\nprint(round(cosine_similarity(a, b), 3))`,
    output: "输出约 0.707。1 表示方向完全相同，0 表示正交；词频向量非负时通常在 0 到 1。",
    pit: "绝不能分别给两句话独立编号后直接比较。例如两边第 0 列若代表不同词，算出的距离毫无意义。",
    quiz: "为什么长度差异很大时，余弦常比欧氏距离合适？", answer: "余弦更关注特征比例方向，整体词频规模变大对它影响较小。",
    diagram: ["两段文本", "共同特征词表", "两个同维向量", "余弦/Jaccard"]
  },
  {
    p: 36, slug: "length-normalization", title: "长度规范化：截断与补齐组成统一批次",
    plain: "一个批次要堆成规则张量，句子长度必须相同。短句补 PAD，长句截断；从左还是从右操作，取决于关键信息通常出现在哪里。",
    points: ["padding='pre/post' 决定 PAD 加在前还是后", "truncating='pre/post' 决定长文本丢前部还是后部", "目标长度应来自长度分布和验证集效果"],
    code: `from text_preprocessing_from_scratch.core import normalize_length\nprint(normalize_length([4, 8], 5, 0, padding="right"))\nprint(normalize_length([1, 2, 3, 4], 3, 0, truncation="left"))`,
    output: "第一行得到 [4,8,0,0,0]；第二行从左截掉 1，保留 [2,3,4]。",
    pit: "PAD 必须有独立 ID，并在注意力或损失中被 mask；否则模型会把大量补齐位置当真实内容学习。",
    quiz: "评论结论常在末尾时，长文本更适合丢前面还是丢后面？", answer: "通常更倾向丢前面（pre truncation），但应通过样本检查和验证集确认。",
    diagram: ["长短不一序列", "长句截断", "短句补 PAD", "统一 B×L 张量"]
  },
  {
    p: 37, slug: "back-translation", title: "回译数据增强：换一种说法，尽量保持标签",
    plain: "把中文译成另一种语言再译回中文，常能得到措辞不同但语义相近的新句子，用来扩充训练数据。",
    points: ["原文→中间语言→原语言；可使用一个或少量中间语言", "增强后要去重、检查语义漂移，并确认分类标签仍成立", "短句容易原样返回，过多翻译链会累积错误、延迟和费用"],
    code: `pairs = [\n    ("这家店服务很好", "这家商店的服务非常棒"),\n    ("完全不好用", "使用体验并不理想"),\n]\nfor original, augmented in pairs:\n    print("原文：", original)\n    print("增强：", augmented)`,
    output: "示例展示“标签大致不变、表达方式改变”。真实回译需调用合规翻译服务或本地模型。",
    pit: "课程中的旧公共翻译接口可能已失效。不要硬编码密钥；还要考虑隐私、费用和服务条款。回译结果必须抽检。",
    quiz: "负面句“这个产品并非不好”回译后最危险的变化是什么？", answer: "否定范围被改坏，变成明确负面或正面，导致原标签不再成立。",
    diagram: ["原始样本+标签", "译到中间语言", "译回中文", "去重/语义/标签过滤"]
  }
];

const detailedArticles = {
  5: `### 0:00–3:51　先建立五项任务的全景

老师开场说明，文本预处理不只是“分一下词”。这一章分成五类工作：基本方法、文本张量表示、语料分析、特征处理和数据增强。先看全景，是为了后面每学一个 API 都知道它在整条流水线中的位置。

基本方法包括分词、词性标注、命名实体识别；表示方法包括 One-Hot、Word2Vec 和 Embedding。老师特别预告 Word2Vec 的 CBOW 与 Skip-Gram 是当天难点，需要提前把“谁预测谁”分清。

### 3:51–7:42　模型为什么不能直接吃原始文字

神经网络底层进行矩阵乘法、加法和求导，不能直接对“天气很好”四个字运算。文本必须先切成 token，再按词表变成 ID 或向量。

原始数据还可能缺失、格式不一致、长度不同。老师用截断与补齐说明：若统一长度为 10，超过 10 的序列截掉后部，不足 10 的补 PAD。但长度 10 不能凭感觉指定，后面要先看长度分布。

预处理的目的不是让数据“看起来整齐”，而是得到模型需要的 X 与标签 Y，同时尽量保留任务相关信息。

### 7:42–10:22　基本方法：边界、角色和现实对象

分词把句段切成词元；词性标注判断名词、动词、形容词等语法角色；命名实体识别提取人名、地名、机构名等现实对象。

老师用“我爱北京天安门”等句子让同学找实体。三项任务互相关联但不相同：有了词边界，不代表已经知道词性；知道某词是名词，也不等于确认它是机构实体。

### 10:22–15:15　稀疏 One-Hot 与稠密词向量

老师先举五词句子。One-Hot 的维度等于词表大小，只有对应词位置为 1，其余全为 0。词表扩大后，大量零造成稀疏表示，而且“猫”和“狗”的向量不会天然比“猫”和“汽车”更接近。

Word2Vec/Embedding 用较短的稠密向量表示词，每一维都可参与计算。老师强调读音应为 Word-to-Vector，并预告 CBOW、Skip-Gram 两种训练方向。稠密不等于每个数字都有人工可解释含义，而是模型从任务中学习分布式特征。

### 15:15–20:12　拿到语料先分析，再决定怎么处理

老师用班级男女数量解释标签分布：类别严重不平衡时，只看准确率会产生错觉。还要分析句子长度、词频与缺失数据。

不同句长会影响 batch 训练，因此需要选择截断/补齐长度；但应根据直方图、分位数和任务需求决定。若大部分只有 10 个 token，少量 200 token，直接按最大值补齐会浪费计算。

### 20:12–26:05　特征处理：绘图、n-gram 和长度规范

词云本质是按词频绘制大小不同的词，适合快速观察常见词，但不能替代定量分析。

n-gram 把连续 n 个 token 组合为局部特征：

- unigram：单个 token；
- bigram：相邻两个；
- trigram：相邻三个。

老师用“今天天气很好适合……”逐步组合说明。n 越大，局部语序信息越多，但组合数量上升、数据更稀疏，通常先从 1–3 gram 尝试。

文本长度规范则根据前面的分布选择截断和 Padding，使同 batch 张量形状统一。

### 26:05–33:02　数据增强与回译

数据不够时可以补采真实业务数据、构造或增强已有文本。老师重点演示回译：中文先翻成英语、法语、德语等，再翻回中文，得到意思相近但表达不同的句子。

例如“你好帅，我好喜欢”可回译为“你很英俊，我非常喜欢你”等。回译可以增加语言表述多样性，但翻译错误也会改变标签；情感、否定、数字和实体必须抽样核查。

### 33:02–36:09　最后用 X、Y 串起全流程

老师要求截图保存思维导图，并用选择题复习。整章最终围绕：

> 原始文本/标签 → 基本处理 → 数字表示 → 数据分析 → 特征/增强 → 可训练 X、Y。

不是每个项目都必须把五项全部用一遍。预处理要由任务决定：删除停用词、截断、增强等操作都可能损伤语义，必须通过验证集检查。`,

  6: `### 0:00–3:54　中文为什么需要分词

老师从“字、词、句、段、篇、章”回顾语言层级。英文常有空格提示词边界，中文连续书写，没有天然的空格分隔，因此程序需要决定哪些字符组成一个 token。

课程把分词结果先近似理解为 token。这里要补充：现代大模型 tokenizer 可能按子词、字节或字符切分，不一定等于日常词语；jieba 解决的是传统中文词分词。

### 3:54–6:54　常见中文分词工具

老师介绍 jieba，并扩展 SnowNLP、HanLP、LTP、THULAC、IK 等工具。它们面向的场景不同：有的包含完整 NLP 管线，有的常用于搜索引擎。学习这份课程先掌握 jieba 的模式与词典，其他工具做到知道名称和用途。

工具多不代表需要全部安装。选型要看语言、领域、速度、部署方式和下游模型。

### 6:54–10:53　环境版本与课程接口

课程建议 Python 3.10 与 jieba 0.42.1，并说明 jieba 支持简体、繁体以及用户自定义词典。老师强调若版本差异造成接口或结果变化，可参考课堂版本。

虚拟环境解决依赖隔离：一个项目升级包，不影响另一个项目。它不会直接提高分词准确率。

### 10:53–16:52　用 Conda 创建独立环境

老师现场演示大致命令：

\`\`\`bash
conda create -n nlp_base python=3.10
conda activate nlp_base
pip install jieba==0.42.1
\`\`\`

创建时 Conda 会列出待安装包并询问确认。环境名可自定，但应有语义。安装完成后可导出依赖版本，便于其他机器复现。

### 16:52–23:56　把 IDE 解释器切到新环境

只在终端创建环境还不够，PyCharm/IDE 项目必须选择对应 Python Interpreter。老师演示新增 Conda 环境、切换已有环境、删除错误解释器和在右下角快速切换。

最常见问题是“明明 pip install 了，代码仍提示找不到 jieba”，通常因为终端 pip 和 IDE Python 不属于同一环境。可同时打印：

\`\`\`python
import sys, jieba
print(sys.executable)
print(jieba.__version__)
\`\`\`

确认实际解释器与包版本。`,

  7: `### 0:00–3:55　先回顾模式与评估术语插曲

老师开始 jieba 精确模式前，回顾精确率、召回率等旧知识，并列出 jieba 的精确、全、搜索引擎、繁体与自定义词典能力。课堂插曲较多，学习主线是先掌握三种模式的切分粒度。

精确模式力图给句子一条主要、不重叠的切分路径，适合普通文本分析。

### 3:55–9:50　模式差异本质是切分粒度

老师再次提到 SnowNLP、LTP、THULAC、IK 等工具，并解释精确模式尽量消除歧义；全模式枚举更多候选、速度快但有重叠；搜索模式为检索补充短词。

同一句话没有唯一绝对正确的切法。新闻分析、搜索索引、实体识别可能需要不同粒度，后续任务决定选择。

### 9:50–13:47　jieba.cut 与 cut_all=False

代码先导入 jieba，准备中文文本，再调用：

\`\`\`python
result = jieba.cut(content, cut_all=False)
\`\`\`

cut_all=False 表示精确模式。返回值不是列表，而是生成器。生成器惰性地产生 token，适合大文本逐个处理，但直接 print 只会看到对象信息。

\`jieba.lcut(...)\` 则相当于直接返回 list。

### 13:47–18:44　三种读取生成器的方法与“指针跑完”

老师先用 next(result) 逐个取词，再用 for 循环继续遍历。for 会从当前未消费位置开始，不会自动回到开头。

生成器遍历完后，再调用 list(result) 得到空列表。课堂把它类比为“指针已经跑到末尾”。若需要多次使用：

- 第一次就转成列表保存；
- 或重新调用 jieba.cut 创建新生成器。

### 18:44–21:04　lcut 更适合课堂查看

\`\`\`python
words = jieba.lcut(content, cut_all=False)
\`\`\`

直接得到列表，能重复遍历、拼接和检查。代价是一次把全部 token 放入内存。小文本和调试通常用 lcut，大语料流式处理可保留 cut 生成器。

精确模式结果应人工用 “/” 连接检查边界。代码运行不报错，只说明接口正确，不说明领域词一定切对。`,

  8: `### 0:00–1:57　全模式解决的不是“唯一切法”

老师从精确模式转到全模式。全模式会扫描文本中尽可能多的成词片段，适合只希望把候选关键词尽量找全的场景。它召回高，但不能消除歧义。

例如“南京市长江大桥”可能同时出现“南京”“南京市”“市长”“长江”“大桥”等重叠候选。它们不能按输出顺序直接拼回原句。

### 1:57–3:55　cut_all=True 与生成器

代码只是把上一节改为：

\`\`\`python
result = jieba.cut(content, cut_all=True)
\`\`\`

返回仍是生成器，打印地址值每次运行可能不同，不必纠结。需要 list 或 for 才能看到 token。

复制上一节测试代码时，老师提醒注释/修改旧测试，避免精确与全模式输出混在一起，误判是哪段代码产生结果。

### 3:55–4:42　比较结果而不是比较数量

老师运行全模式并与精确模式对照。全模式词更多只是因为包含重叠候选，并不等于更准确。

普通词频统计通常使用精确模式，否则同一字符范围可能被多次计数；搜索召回或候选生成可以考虑全模式。选择标准是“下游是否允许重叠”。`,

  9: `### 0:00–1:58　搜索模式为什么为检索服务

老师说明搜索引擎模式适合建立索引。精确模式保留较完整长词；搜索模式在此基础上把较长词再切出短片段，使用户只输入长词的一部分时也可能命中文档。

调用：

\`\`\`python
jieba.cut_for_search(content)
# 或 jieba.lcut_for_search(content)
\`\`\`

### 1:58–4:54　用“黑马程序员”解释召回

若文档只按完整长词建立索引，用户搜索“程序”可能匹配不到“黑马程序员”。搜索模式同时保存长词和部分短词，相当于建立多个检索入口。

这里的“智能”不是模型理解了用户全部意图，而是索引粒度更丰富，减少了漏检。

### 4:54–7:52　“苹果手机保护套”示例

老师继续问：用户只搜“手机壳/保护套”，能否找到“苹果手机保护套”。精确切分可能只保留长词组合，搜索模式会补充“苹果、手机、保护、保护套”等候选，提升召回。

召回提高也可能带来无关结果增加，所以真实搜索还需要相关性排序、同义词、字段权重等，不能仅靠分词。

### 7:52–10:43　复制代码最容易留下旧变量

课堂从上一文件复制函数并改成搜索模式，提醒检查函数名、测试名和实际调用。若测试仍调用旧函数，输出正常却不是本节逻辑。

搜索模式返回的生成器同样可 list 转换。保存为列表后才能重复打印或比较。

### 10:43–12:03　三种模式最终对比

- 精确模式：主要切分路径，适合一般分析；
- 全模式：尽量枚举候选，重叠最多；
- 搜索模式：精确结果上补短词，面向索引召回。

三者差异是粒度与使用目标，不是“高级模式一定比低级模式准确”。训练序列模型通常需要稳定的不重叠序列，不能直接把搜索模式重叠输出当作时间步。`,

  10: `### 0:00–1:54　繁体文本不需要换一套 API

老师继续完成 jieba 的繁体中文演示。Python 3 字符串使用 Unicode，繁体文字可直接传给 jieba；精确模式仍是 \`cut_all=False\`，全模式改 True，搜索模式使用 cut_for_search。

接口相同不表示效果必然相同。实际边界取决于词典是否包含当地用语、繁体词和领域专名。

### 1:54–3:05　三种模式在繁体上的调用

课堂分别调用精确、全和搜索模式，并打印结果。代码重点是：

\`\`\`python
text = "我來到北京清華大學"
print(jieba.lcut(text, cut_all=False))
print(jieba.lcut(text, cut_all=True))
print(jieba.lcut_for_search(text))
\`\`\`

是否先做简繁转换取决于任务。检索若希望简繁互通，可同时保留原文和规范化字段；生成、档案或姓名任务可能需要保留原貌。机械转换会误伤专名，因此要抽样验证。`,

  11: `### 0:00–3:52　用户词典每行三个字段

老师介绍 jieba 用户自定义词典。每行通常是：

\`\`\`text
词语 词频 词性
\`\`\`

词语必填，词频与词性可选。词频不是“这个词在业务文件中真实出现了多少次”的严格统计，而是参与分词概率的权重；数值越高，分词器越倾向把它保留为整体。

词性如 n、v 等来自标记体系，课程建议知道常见大类即可，需要时查对照表，不必一次背完。

### 3:52–8:49　词性表与词典文件创建

老师展示词性对照，并新建 user_dict.txt，把课堂领域词、频率和词性写进去。用户词典应该只补充通用词典缺少的公司名、产品名、术语、人名等。

若几乎所有普通词都手工写入，说明方案难维护；词频乱设极大还会破坏其他边界。用户词典要版本管理，并配回归样句。

### 8:49–12:46　加载前后必须对比

先在未加载时分词：

\`\`\`python
before = jieba.lcut(text)
jieba.load_userdict("user_dict.txt")
after = jieba.lcut(text)
\`\`\`

老师比较加载前后，观察原本被拆开的领域词是否合并。路径错误、编码错误或格式多空格都可能导致加载失败。

### 12:46–14:15　词频与词性分别影响什么

高词频提高整体成词倾向；词性给 posseg 等后续标注提供提示。词性不是强制，若任务只关心边界可省略；若后面按名词/动词过滤，应认真填写并验证。

还可用 \`jieba.add_word\` 动态添加单词。无论文件还是动态方式，最终都要在包含歧义的句子上测试，而不是只看词典是否成功加载。`,

  12: `### 0:00–3:50　NER 找“现实世界中的谁”

老师把命名实体识别与词性标注放在一起讲。NER（Named Entity Recognition）从文本中识别人名、地名、机构名、时间等实体。

以鲁迅介绍句为例，“鲁迅”是人名，“浙江绍兴”是地名。实体通常是名词性短语，但“名词”不自动等于“命名实体”；普通概念“桌子”是名词，却不是特定实体。

### 3:50–7:41　POS 标注词的语法角色

POS（Part-of-Speech tagging）先分词，再为每个 token 标注名词、动词、形容词等角色。同一个字/词在不同语境可能承担不同词性，老师用中文绕口句说明语境的重要性。

NER 与 POS 相关但任务不同。面试回答应先分别定义，再说 POS 可为 NER 提供特征，但现代上下文模型往往联合利用整句表示。

### 7:41–10:39　jieba.posseg 的返回结果

课程导入：

\`\`\`python
import jieba.posseg as pseg
items = pseg.lcut("我爱自然语言处理")
\`\`\`

每个 item 含 word 与 flag。flag 是词性代码，需要查 jieba 标记表解释。不要看到字母就猜含义。

### 10:39–17:35　写函数并暴露工具局限

老师定义 NER/POS 演示函数，遍历：

\`\`\`python
for word, flag in pseg.lcut(text):
    print(word, flag)
\`\`\`

课堂结果没有把某些地名正确识别出来。老师借此说明词典/规则工具有误判，用户词典可改善部分专名，但复杂 NER 还需要标注数据和上下文模型。

严格说，posseg 的主要输出是分词+词性，并不是完整的学习型 NER 系统。可把特定词性标记当作实体候选，但不能把它等同于可靠 NER。

### 17:35–24:02　总结与选择题

老师复习分词三种模式、繁体、用户词典、NER 与 POS，并用选择题检查：

- 分词：把句子切成 token；
- NER：识别人/地/机构等实体；
- POS：为每个词标语法角色。

大模型任务最终要理解人类语言，边界、角色和实体都是不同层面的辅助信息。`,

  13: `### 0:00–2:57　文本表示解决“文字不能做矩阵运算”

老师再次强调模型只处理数字。学习目标包括文本向量概念、One-Hot 原理，以及 Word2Vec 的 CBOW/Skip-Gram 推理方向。

单个词可表示为向量；一句话中多个词向量按顺序排列，就形成词向量矩阵。batch 后再多一个批次维。

### 2:57–6:53　用成绩向量类比人物特征

老师用张三多门考试成绩表示一个人：一串数字可从多个维度描述对象。类似地，一个词用若干数字表示；句子中每个词都有一行，合起来成为矩阵。

类比的边界是：词向量维度通常不是人工命名的“语文、数学”，而是通过训练学习到的分布式特征。

### 6:53–9:50　One-Hot 的身份编码

若类别为红、绿、蓝，可用 [1,0,0]、[0,1,0]、[0,0,1]。词表中每个词同样占唯一位置。优点是确定、不会混淆；缺点是词表越大向量越长，而且不同词之间没有语义距离。

老师由稀疏问题引出 Word2Vec 与 Embedding 稠密表示。

### 9:50–13:15　固定顺序、Tokenizer 与 OOV

One-Hot 的列顺序必须固定。训练时“我”若在第 1 列，预测时也必须在同一列。课程展示旧版 Keras Tokenizer/编码接口，把文本拟合为词到索引映射，再生成矩阵。

新词不在词表中会找不到。真实系统应设置 OOV/UNK 策略，并保存 tokenizer，而不是预测时重新拟合。

ID 本身只是地址：编号 5 的词不比编号 2 的词“更大”。只有进一步 One-Hot 或 Embedding 查表后才成为模型特征。`,

  14: `### 0:00–3:55　先把 One-Hot 用一句话讲清

老师定义 One-Hot：先分词，建立固定词表，把每个词映射为长度等于词表大小的 0/1 向量；对应列为 1，其余为 0。句子最终成为二维文本张量，可作为模型输入。

### 3:55–6:50　稀疏表示与稠密表示对比

老师联系以前的情感分析与神经网络输入层：One-Hot 是稀疏词向量；Word2Vec/Embedding 是稠密表示。Embedding 层把词 ID 当行号查参数表，不是自动从字符串理解语义。

### 6:50–12:48　课程旧版依赖与 Tokenizer

课堂尝试安装/导入 Keras 相关 Tokenizer，不同版本出现模块路径变化与安装速度差异。这个现场过程说明：旧教程 API 可能随版本更新，理解“拟合词表→固定索引→生成编码”比死记具体包路径更重要。

现在若复现，可用自己保存的 Python 字典或当前 tokenizer 库；不要为了旧 API 盲目降级整个环境。

### 12:48–21:49　拟合词表后索引如何变化

Tokenizer 根据训练文本建立 word_index。索引通常从 1 开始，0 留给补齐或保留位，因此手写 One-Hot 时要注意列号与数组下标是否需要减 1。

课堂打印过程中出现大量 0 和耗时变化。关键不是控制台每个数字，而是确认：

- 词表大小 V；
- 输入 token 数 L；
- 输出矩阵 shape [L,V]；
- 每行恰好一个 1（无 OOV 时）。

### 21:49–26:40　手工修改对应位置为 1

老师用全零数组演示：找到 token 的索引，将相应位置置 1。若索引从 1 开始而数组从 0 开始，需明确偏移，否则会越界或错一列。

更稳健的代码可直接按自己定义的 0-based vocab 生成。

### 26:40–29:46　为什么必须保存 tokenizer

老师最后强调每次重新训练/拟合词表，索引可能不同。模型权重已经把每列含义固定下来，预测时必须加载当时的 tokenizer/词表。

只保存模型而不保存词表，等于保留了一台机器却丢了输入插头定义。未知词应映射到 UNK，而不是临时追加新列改变维度。`,

  15: `### 0:00–1:59　加载的不只是模型，还有词到索引的规则

老师继续演示 One-Hot 的使用：先加载上一节保存的 tokenizer/词汇映射，再取 \`word_index\`。预测阶段的列含义必须与训练阶段一致。

序列化文件只应从可信来源加载；Python pickle 能执行对象反序列化逻辑，不能随意打开陌生文件。

### 1:59–4:54　给指定词构造 One-Hot

根据词表长度创建全零列表，查出词索引，再把对应位置改为 1。若 tokenizer 索引从 1 开始，数组下标需要做一致处理，或预留第 0 列。

核心不是写循环，而是保持三个不变量：向量长度固定、已知词列固定、未知词策略固定。

### 4:54–5:47　陌生词报错揭示 OOV 问题

课堂输入词表中不存在的名字/词，查表时报错。真实系统不应让任意新词导致崩溃，应在训练建词表时设置 \`<UNK>\`，预测时：

\`\`\`python
idx = vocab.get(word, vocab["<UNK>"])
\`\`\`

\`<PAD>\` 与 \`<UNK>\` 通常分开：PAD 表示没有内容，UNK 表示有内容但词表不认识。临时追加新词会改变模型输入含义，不能作为上线补救。`,

  16: `### 0:00–2:57　不用旧库也能手写 One-Hot

老师扩展最朴素实现：先得到稳定、不重复的词列表，为每个词建立索引。若直接用 set 去重，遍历顺序在不同运行/实现中可能不适合作为持久规则，因此应使用固定排序或首次出现顺序。

\`\`\`python
vocab = {w: i for i, w in enumerate(dict.fromkeys(tokens))}
\`\`\`

### 2:57–4:56　每个词生成全零列表并置 1

循环每个 token：

\`\`\`python
vec = [0] * len(vocab)
vec[vocab[word]] = 1
\`\`\`

老师强调这个实现不需要复杂工具：全零、找位置、改成 1。它适合帮助理解和处理很小的类别特征。

### 4:56–7:50　优点和两个核心缺点

优点是简单、确定、可解释。缺点一是维度随词表线性增长：10 万词就需要 10 万维。缺点二是极度稀疏，大多数元素为 0，浪费存储/计算。

此外，不同词的 One-Hot 彼此正交，不能表达“猫”比“汽车”更接近“狗”。这引出 Word2Vec 等稠密表示。

### 7:50–10:31　什么情况下仍然适合

One-Hot 也叫独热/零一编码，有该类别为 1、没有为 0。类别数量很少时，它仍是合理方案；例如几个颜色或有限标签。

老师用选择题让同学判断：类别取值非常多时不适合 One-Hot。结论不是“One-Hot 已淘汰”，而是词表规模和语义关系决定是否换 Embedding。`,

  17: `### 0:00–3:55　Word2Vec 用预测任务学习稠密向量

老师定义 Word2Vec：用浅层神经网络和上下文预测目标训练权重，再把某层权重矩阵作为词向量。没有人工标签也能构造监督信号，因为句子中的邻近词互相充当输入与标签。

课程指出两套方向：CBOW 用两边上下文预测中间词；Skip-Gram 用中心词预测周围词。本节重点先画 CBOW。

### 3:55–9:50　输入层、隐藏层、输出层的维度

假设语料库只有 5 个词，输入和输出都是 5 维 One-Hot；隐藏层设为 3 维。输入→隐藏权重矩阵把 5 维投到 3 维，隐藏→输出再投回 5 维词表 logits。

隐藏维 3 只是教学方便，实际可用 100、300 等。输入/输出维由词表大小决定，不能随意改成与词表不一致。

### 9:50–15:43　最终取哪张权重当词向量

老师围绕矩阵方向反复纠正 5×3 / 3×5。若采用行向量约定，One-Hot [1,V]×W[V,D] 得 [1,D]，W 的每一行对应一个词的 D 维向量；采用列向量时矩阵写法会转置。

不要只背“5×3”。先写输入在左还是右，再做矩阵乘法维度检查。课程最终要表达的是：输入到隐藏的权重表为每个词保存稠密表示。

### 15:43–22:42　CBOW 样本怎样构造

对长度窗口，选中中心词，两侧上下文作为输入，中心词为标签。例如：

\`\`\`text
上下文（爱，语言） → 中心词“自然”
\`\`\`

多个上下文 One-Hot 可求和/平均后送入网络。前向得到词表预测，和中心词真实 One-Hot 计算损失。

老师回顾深度学习训练三件事：前向传播得预测、损失函数衡量差异、反向传播更新权重。

### 22:42–25:41　CBOW 与 Skip-Gram 的一句话区别

老师在图旁写：

- CBOW：context → center；
- Skip-Gram：center → context。

无论哪种，训练的目标是让出现在相似上下文中的词形成有用向量，而不是最终部署一个“猜中间词”的产品。

### 25:41–40:34　手推一次矩阵乘法

老师放大输入—隐藏连接，逐项写 W11、W12 等。One-Hot 的作用是选中权重矩阵对应行：只有一个输入位置为 1，其余为 0，矩阵乘法后恰好取出该词向量。

上下文含多个词时，多个被选行会组合/平均形成隐藏表示，再乘隐藏→输出权重，Softmax 得到中心词概率。

### 40:34–45:29　损失、反向更新与遍历语料

预测分布与真实中心词计算损失，反向传播同时更新两张权重。移动窗口，重复构造下一组样本，直到多轮遍历语料。

训练完成后取输入权重（或按实现组合两套权重）作为词向量。老师最后强调这里讲的是 CBOW 的词向量获取；下一节把输入输出方向反过来讲 Skip-Gram。`,

  18: `### 0:00–2:56　Skip-Gram 从中心预测多个上下文

老师紧接 CBOW 反转任务。给定中心词，分别预测窗口左侧和右侧词。一个中心词可产生多个训练对，例如：

\`\`\`text
爱 → 我
爱 → 自然
\`\`\`

### 2:56–5:54　网络结构可复用，样本方向不同

输入、隐藏、输出三层与词表维度都可沿用。区别是输入 One-Hot 现在代表中心词，标签是某个上下文词。

老师复制上一张图修改箭头，提醒不要误以为 Skip-Gram 是全新网络；它主要改变训练样本与损失目标。

### 5:54–7:51　多个上下文损失怎样处理

中心词预测左词与右词会得到多个损失，可求和或平均，再反向更新同一套权重。实际高效 Word2Vec 常使用负采样等方法，不会每次对超大词表做完整 Softmax；课程先掌握基础原理。

### 7:51–9:53　最终拿到谁的词向量

Skip-Gram 同样从输入到隐藏的权重中取得中心词稠密表示。老师用“基于中间预测两边”作结。

常见经验是 CBOW 训练较快，Skip-Gram 对低频词更友好，但效果依赖语料、窗口、负采样等，不能把经验当绝对定律。`,

  19: `### 0:00–3:54　从手推原理转向 FastText 工具

老师说明 CBOW/Skip-Gram 原理已经讲完，接下来不手写完整高效训练器，而使用 Facebook Research 开源的 FastText 工具。Word2Vec 由 Tomas Mikolov 等人在 Google 推动；FastText 是另一套加入子词信息的模型/库，不能把两者作者和实现完全等同。

### 1:56–5:53　准备百科语料

课程使用预处理后的维基百科中文语料。原始压缩/解压体积较大，已通过正则清洗、分词，并用空格隔开 token。

FastText 无监督训练输入通常一行一段已分词文本。若中文整句没有空格，库可能把整句当一个 token，无法学到预期词向量。

### 5:53–8:53　大文件拆分与编辑器显示限制

老师展示资料中的文本文件。文件过大时 IDE 只显示一部分，不代表数据被截断；应用命令行统计行数/大小，并用流式读取抽样。

拆成多个小文件便于演示和调试，正式训练要明确到底传入哪份语料、编码是否 UTF-8、是否有空行/乱码。

### 8:53–11:50　把原理写进项目说明

老师再次记录：Word2Vec/稠密词向量通过上下文预测训练，无论 CBOW 还是 Skip-Gram，都会利用隐藏层权重形成词表示。项目文档应写清数据来源、预处理步骤、训练模式和维度。

### 11:50–14:33　安装 fasttext 与编译问题

课程建议安装 fasttext，课堂版本约 0.9.2。部分系统需要 C++ 编译工具，直接 pip 安装可能失败；可根据官方发行方式、Python/系统版本选择兼容构建。

不要只追求和老师版本完全一致。先在独立环境中确认：

\`\`\`python
import fasttext
print(fasttext.__file__)
\`\`\`

安装完成后再用极小语料跑通，避免在数 GB 数据上才发现环境问题。`,

  20: `### 0:00–2:55　train_unsupervised 开始无监督训练

老师进入 FastText 训练与保存。调用 \`fasttext.train_unsupervised\`，输入分词语料文件。无监督表示“不需要人工类别标签”，不是没有训练目标；窗口里的上下文仍构造预测信号。

默认模型/参数应以当前安装版本文档为准。课堂查看函数定义，看到 model、dim、epoch、lr、thread 等默认值。

### 2:55–4:50　先用一份文件跑通流程

老师选择拆分语料中的一个文件作为 input，并指定保存路径。小文件用于验证代码与环境；正式质量需要更大且干净的语料。

训练函数会读取词表、统计 token，再多轮更新。文件路径、编码和分词格式错误会在这里暴露。

### 4:50–8:47　怎样读训练日志

控制台日志包含进度、词数/词表规模、每秒处理词数、学习率、损失与预计时间。老师解释“去重后不同词数量”和总 token 不是同一个概念。

速度受 CPU 线程、文件缓存、语料大小和参数影响；不要只看每秒数字判断模型好坏。损失是否可比较也要求语料和配置一致。

### 8:47–9:04　更大文件通常信息更多，但不是必然更好

老师提出用 file9 等更大语料训练，通常比很小样例得到更丰富的词关系。但若大文件包含乱码、重复、错误分词或领域不匹配，规模会放大噪声。

训练完成后必须 \`save_model("xxx.bin")\`。模型文件保存统一向量空间，后面才能加载查询与评估。`,

  21: `### 0:00–2:54　评估先分“直观看”与“任务验证”

老师准备加载已保存模型，查看单词向量和最近邻。最近邻能直观判断相似词是否合理，但属于内在/主观检查；真正项目还要在下游分类、检索等验证集上量化。

### 2:54–5:52　get_word_vector 返回 dim 个数字

\`\`\`python
model = fasttext.load_model(path)
vector = model.get_word_vector("目标词")
\`\`\`

若 dim=100，返回形状 [100]。老师打印具体数值并尝试一个个人名字。标准 FastText 可用字符 n-gram 为未登录词组合向量，因此 OOV 不一定报错；得到向量也不等于语义一定可靠。

### 5:52–8:46　get_nearest_neighbors 看近邻

\`\`\`python
model.get_nearest_neighbors(word, k=10)
\`\`\`

返回若干（相似度，词）。老师说明默认数量并用它检验模型是否学到语义关联。

相似度通常基于向量空间方向，表示分布式用法接近，不保证事实关系、因果或同义。

### 8:46–11:37　多查几组，不要挑成功案例

课堂观察前几个近邻“还不错”，并建议更多语料改善质量。更可靠流程是固定一批常见词、低频词、领域词和歧义词，批量记录结果。

若目标是情感分类，最近邻合理仍不足以证明有效；应把向量放入分类器，与随机/其他预训练表示比较验证 F1。`,

  22: `### 0:00–1:59　调参前先记录默认基线

老师最后讲 FastText 超参数。先保留一套默认配置和固定语料/评估集，否则每次变化没有比较基准。

### 1:59–4:57　model、dim、epoch、lr

- model：cbow 或 skipgram，决定预测方向；
- dim：词向量维度，控制容量、内存和查询成本；
- epoch：完整遍历语料次数；
- lr：每次更新步长。

课堂把默认 Skip-Gram 改为 CBOW，并指出 epoch、lr 都可调。参数越大不代表越好：维度过高可能需要更多数据，epoch 过多会拟合噪声，lr 过大可能不稳定。

### 4:57–6:54　训练、保存与评估必须成套

修改配置后重新训练并保存不同文件名，再用相同词表/下游指标比较。不要覆盖唯一模型，否则无法回退。

建议每次只改一个关键变量，同时记录语料版本、随机种子、线程数、训练时间和指标。多参数同时改变，结果变好也无法知道原因。

CBOW 通常更快，Skip-Gram 常对低频词更友好，但这是经验，不是保证；以自己的语料和任务验证。`,

  23: `### 0:00–2:55　Embedding 的广义与狭义

老师说明广义 Word Embedding 是所有把离散词变成稠密向量的技术，Word2Vec、FastText 都属于其中。狭义课堂语境常指神经网络里的词嵌入层（如 nn.Embedding）。

### 2:55–7:48　预训练 Word2Vec 与任务内 Embedding 的流程差异

Word2Vec/FastText 通常先在大语料单独训练、保存，再把向量用于下游任务。nn.Embedding 则可以作为模型第一层，与分类/翻译损失一起端到端更新。

两者也能结合：加载预训练矩阵初始化 Embedding，再选择冻结或微调。

### 7:48–12:53　再复习 CBOW、Skip-Gram 与权重矩阵

老师回到前向—损失—反向传播，强调输入到隐藏的权重矩阵可作为词向量。CBOW 用上下文预测中心，Skip-Gram 用中心预测上下文。

nn.Embedding 只完成 ID 查表，本身没有 CBOW/Skip-Gram 训练目标；它的语义来自下游损失或加载的预训练权重。

### 12:53–18:51　函数复习与选择题

课堂复习 FastText 的训练、保存、加载、取向量、最近邻和超参数，并用题目检查两种模式方向与经验差异。

窗口扩大时，CBOW 会聚合更多上下文预测中心；Skip-Gram 会让中心产生更多上下文训练对。

### 18:51–27:54　课堂扩展阅读与图示纠正

老师讨论 CBOW/Skip-Gram 哪个更适合高频/低频词，并借助外部生成的解释检查图。重要结论是：CBOW 聚合上下文时常先求和或平均，再预测中心；图中矩阵方向要以实际输入约定核对。

课程现场也说明生成式辅助内容可能写错，必须用公式和维度验证。最终可记经验：CBOW 速度快、对高频稳定；Skip-Gram 训练对更多、常照顾低频。具体效果仍以实验为准。`,

  24: `### 0:00–3:57　从表示法进入词向量可视化案例

老师总结稀疏 One-Hot 与稠密 Word2Vec/Embedding，并提出案例目标：准备两句话，分词、建词表、取 Embedding 权重，再把词向量可视化观察关系。

### 3:57–7:44　新项目与依赖

课堂新建工程，导入 jieba、TensorFlow/Keras Tokenizer 与后续可视化工具。旧版依赖今天可能变化；核心流程可用 Python 字典和 PyTorch 完成，不必绑定某个 Tokenizer 版本。

### 7:44–13:28　两句话先分词并放进列表

老师定义句子列表，遍历每句调用 jieba，把结果保存为“句子列表中的 token 列表”。要打印检查，避免把 generator 对象直接塞入后续。

若环境日志过多，可调日志级别，但不要因此忽略真正异常。

### 13:28–17:22　Tokenizer 建词表

将分词结果拟合为 token→id 映射。老师纠正输入对象：应传 token 列表而不是未切分原句。查看 word_index 的 keys/values 可理解词与编号。

词表顺序一旦用于 Embedding 必须固定保存。

### 17:22–20:20　文本序列化为 ID

每个句子中的 token 按映射变为整数序列。例如同一个“我”无论出现在何处都应得到相同 ID。

不同句长尚未补齐时是变长列表；若要组成 batch 张量，后面还需 padding。

### 20:20–24:30　Embedding 查表与权重矩阵

创建 Embedding(num_embeddings=V,embedding_dim=D)，ID [B,L] 查表后为 [B,L,D]。老师查看权重参数：矩阵每一行就是某个 ID 当前的词向量。

随机初始化的向量没有语义，只是为可视化流程准备数据；要看到可靠聚类，需要训练或加载预训练向量。本节先完成“词→ID→按行查向量”的链路。`,

  25: `### 0:00–3:53　把词与对应向量一起写入日志

老师在已取得 Embedding 权重后做可视化。需要两份一一对应的数据：

- 词向量矩阵 [V,D]；
- V 个词标签，顺序与矩阵行一致。

课堂例子约 20 个词、每词 8 维，因此矩阵 [20,8]。词数量与权重表行数若不匹配，Projector 标签会错位。

### 3:53–8:35　从 word_index 按 ID 顺序取向量

Tokenizer 的 word_index 是 word→index。老师遍历字典，将每个 index 转为张量，送入 Embedding 取对应行，并同步保存 word。

必须按索引排序，而不是依赖任意字典遍历顺序。若索引从 1 开始，Embedding 行数与第 0 行保留位也要一致。

### 8:35–12:22　SummaryWriter.add_embedding

\`\`\`python
writer.add_embedding(vectors, metadata=words, tag="demo")
writer.close()
\`\`\`

TensorBoard 日志写入 runs 目录。vectors 应是二维 [V,D]；metadata 长度为 V。

### 12:22–16:22　启动 TensorBoard 并进入 Projector

终端切到日志所在项目，运行：

\`\`\`bash
tensorboard --logdir=runs --host=127.0.0.1
\`\`\`

浏览器通常打开 6006 端口。若提示 No dashboards/data，检查 logdir 是否指到实际事件文件，而不是反复刷新空目录。

Projector 可选择 PCA、t-SNE 等投影，每个点代表一个词。搜索词只能帮助定位，不代表投影距离必然可靠。

### 16:22–22:42　高维图只能探索，不能代替评估

老师点击点观察相近词，并总结流程：句子列表→分词→建词表→ID→Embedding→日志→浏览器可视化。

降维会扭曲距离。图中靠近的词还应回到原始 D 维向量计算余弦相似度；随机未训练 Embedding 形成的“聚类”没有语义证据。

课堂最后扩展张量 reshape 的问题。无论怎样变形，都要保持元素总数不变，并明确词标签仍对应哪一行。`,

  26: `### 0:00–3:54　文本分析章节要回答哪些问题

老师进入下半章：标签数量、句长、词频、样本散点、缺失值与脏数据。分析不是为了画漂亮图，而是帮助发现会影响训练和评估的问题。

### 3:54–8:50　先看标签是否平衡

评论数据使用 0/1 表示负面/正面。老师用生活中的反讽差评说明：文本情感不能只靠星级或单个关键词。

分类数据理想情况下各类数量接近，但不必机械追求严格 1:1。若极不平衡，应考虑分层划分、类别权重、重采样，并同时看 precision、recall、F1。

### 8:50–13:46　CSV 与 TSV 的分隔符

老师现场展示数据文件，发现读取后只有一列，原因是实际以 Tab 分隔。Pandas 应：

\`\`\`python
pd.read_csv(path, sep="\\t")
\`\`\`

CSV/TSV 文件扩展名不是绝对保证，最好打开前几行或检查列数。分隔符读错会让后续标签列不存在。

### 13:46–18:42　导入分析与绘图库

课程使用 pandas、matplotlib、seaborn、jieba.posseg 等，并设置中文字体。不同系统字体名不同；字体配置只影响显示，不应改变数据。

### 18:42–23:38　countplot 绘制训练/测试标签数量

读取 train/test 后，用 countplot 或 value_counts 统计 label。hue、palette 等参数控制分组与颜色；标题和紧凑布局帮助解释。

### 23:38–26:13　读图结论

老师观察 0/1 数量很接近，认为不需大幅调整。还应分别检查训练、验证、测试比例是否一致，并保存精确计数。

图只是摘要。决定策略时应用表格记录数量/比例，并建立“恒猜多数类”的基线，避免被表面准确率欺骗。`,

  27: `### 0:00–3:59　统计句长前先补 map 基础

老师准备给每条评论计算长度，但先单独讲 map。后面将把 \`len\` 或分词长度函数应用到 DataFrame 每条文本。

### 3:59–7:52　map(function, iterable)

定义一个简单函数，再：

\`\`\`python
result = map(fn, [1,2,3,4,5])
\`\`\`

map 会把同一个 fn 依次用于每个元素，但返回迭代器，不会在创建时马上把全部结果算成列表。

### 7:52–9:51　惰性计算与一次消费

打印 map 只看到对象。调用 \`list(result)\` 才触发/消费结果；同一个迭代器再次 list 可能为空。

惰性适合大数据流，若后面要反复画图或统计，应及时转成 Series/list 保存。

### 9:51–12:41　lambda 是短函数写法

\`\`\`python
list(map(lambda x: x * x, [1,2,3]))
\`\`\`

等价于先定义一行函数再传入。lambda 只适合简单表达式；清洗逻辑复杂时应写有名字的函数，便于测试和记录异常。

回到句长任务，\`map(len, texts)\` 得到字符长度。若模型输入单位是 jieba token 或子词，就应传相应 tokenizer 计数函数，不能混用。`,

  28: `### 0:00–3:54　把 map 应用到训练集与测试集

老师回到评论数据，为 train/test 增加 sentence_length 列。对每条 sentence 计算 len；课堂说明后续可据此选择截断/补齐长度。

若中文字符串直接 len，统计的是字符数；若模型按 token 输入，应先 tokenize 再 len。

### 3:54–6:52　先查看具体长度和异常值

新增列后打印前几行，并观察最大/常见长度。老师举 48、80 等数，让同学思考目标长度。

不要直接把最大值作为 padding 长度。极少数超长评论会让所有样本浪费显存，应结合分位数、任务和长文本策略。

### 6:52–9:49　旧版 distplot 与弃用问题

课堂先展示 seaborn distplot，然后指出新版本已弃用。旧代码可能仍能运行但有警告，应改为：

\`\`\`python
sns.histplot(data=df, x="sentence_length", kde=True)
\`\`\`

### 9:49–13:48　直方图与 KDE 分别怎么看

直方图柱子表示某长度区间有多少句子；KDE 曲线是平滑密度估计，帮助看整体形状，但受带宽影响，不能代替原始计数。

训练集和测试集应使用相同 bins/坐标范围比较，避免视觉尺度造成误判。

### 13:48–15:50　从分布选择规范长度

老师观察大量句子集中在约 25 左右，少数形成长尾。实际应计算 90/95/99 分位，再比较不同 max_len 对截断比例、显存和任务指标的影响。

分布告诉你“数据是什么样”，不是自动给出唯一阈值。`,

  29: `### 0:00–2:59　为什么还要按标签看长度

总体直方图可能掩盖类别差异。老师接着画正负样本长度散点，检查某一类是否更长、极端长评论集中在哪类。

课堂还建议打印三千多字符的异常评论回看原文。异常可能是真实长文本、重复抓取、HTML 或拼接错误，处理策略不同。

### 2:59–5:57　stripplot 的 x、y 与 hue

横轴 label（0/1），纵轴 sentence_length，每个点是一条样本。hue 可按标签着色，jitter 让重叠点稍微错开。

训练集与测试集可画成并列子图，坐标范围保持一致。

### 5:57–7:03　读图不要把所有离群点当脏数据

老师解释分组参数并观察散点。发现极长点后，第一步是回到原文，不是直接删除。

若正样本全部来自长评论、负样本来自短标题，模型可能只学长度捷径；若长度差异来自真实表达，也可能是有效信号。应做去除/保留长度特征的对照实验，并检查数据采集来源。`,

  30: `### 0:00–1:57　先学 chain，再做全语料词汇统计

老师要统计训练集和测试集分别有多少种词。每句话分词后是一个列表，整个数据集就成了“列表的列表”；在计数之前，必须先把里面的词串成一条词流。因此老师暂时离开业务代码，先单独演示 \`itertools.chain\`。

\`\`\`python
from itertools import chain

left = [1, 2, 3]
right = [2, 4]
stream = chain(left, right)
\`\`\`

### 1:57–4:53　chain 返回惰性迭代器

创建 chain 对象时不会马上遍历底层数据，真正调用 \`list(stream)\`、for 循环或交给其他消费者时才逐个取元素。老师把它与前面学过的 map 对照：两者都返回惰性迭代器。

迭代器一旦消费完，同一个对象再次转成列表通常为空。若后续要重复使用结果，应第一次就保存为 list；若数据很大且只消费一次，保留迭代器更省内存。

### 4:53–7:50　从两个列表过渡到“每句话的分词列表”

普通 \`list.extend\` 会修改原列表；chain 则只是提供统一的遍历视图。老师重新定义两个句子，先用 map 对每个句子调用 jieba 分词：

\`\`\`python
tokenized = map(lambda sentence: jieba.lcut(sentence), sentences)
\`\`\`

此时 tokenized 中的每个元素仍是一个词列表，还没有铺平。

### 7:50–10:50　星号解包与去重

课堂写法用 \`chain(*tokenized)\`：星号先把 map 产生的多个词列表逐个取出，再作为多个参数传给 chain。

\`\`\`python
unique_words = set(chain(*map(jieba.lcut, sentences)))
\`\`\`

这行代码完成三件事：全部句子分词、铺平成一个词流、用 set 去重。更直接且不需要星号展开的写法是：

\`\`\`python
unique_words = set(chain.from_iterable(map(jieba.lcut, sentences)))
\`\`\`

### 10:50–12:21　一行代码的边界

老师把整段压成一行，强调以后熟练后代码会更短。但短不等于总是更好：若还要统计频率，就不能过早变成 set；应把平坦词流交给 Counter。若需要调试分词错误，也应拆开中间步骤打印抽样结果。`,

  31: `### 0:00–1:00　把 chain 用回训练集与测试集

上一节解决了“如何铺平词列表”，这一节回到业务需求：分别获取训练集和测试集的不同词汇总数。老师把函数命名为类似 \`word_count\`，提醒这里要数的是去重后的词汇量，不是包含重复的总 token 数。

### 1:00–2:59　把示例中的两句话换成整列数据

老师让同学不要重新发明代码，而是辨认示例和真实数据之间唯一的差别：示例输入是两个句子，业务输入是 DataFrame 中的整列句子。

\`\`\`python
from itertools import chain
import jieba

def vocabulary_size(sentences):
    token_lists = map(jieba.lcut, sentences)
    unique_words = set(chain.from_iterable(token_lists))
    return len(unique_words)
\`\`\`

map 对每个句子分词，chain 铺平，set 去重，len 才得到“不同词汇数”。

### 2:59–3:59　把精确数字写进可读报告

老师继续组织输出，让训练集和测试集的词汇总数以完整中文句子显示。实际项目除了打印，还应保存以下量：

- 总 token 数：\`sum(counter.values())\`；
- 不同词汇数：\`len(counter)\`；
- 只出现一次的词数；
- 测试集中训练词表未覆盖的 OOV 比例。

### 3:59–6:40　课堂互动与可靠实现的补充

后半段主要是围绕格式化输出的课堂提问和反复纠正，没有引入新的算法步骤。更稳妥的实现是使用 Counter：它既保留频率，也能随时得到词汇量；等确定最低频率阈值后再建词表，避免大量拼写错误和只出现一次的噪声词膨胀输入空间。

\`\`\`python
from collections import Counter
counts = Counter(chain.from_iterable(map(jieba.lcut, sentences)))
print("总 token 数：", sum(counts.values()))
print("不同词汇数：", len(counts))
\`\`\``,

  32: `### 0:00–1:54　任务：比较正负样本中的高频形容词

老师准备分别绘制训练集正样本、负样本中的高频形容词词云。形容词只是示例；把词性条件换成名词或动词，整条流程不变。这个任务需要三个独立函数：从一句话取形容词、从词列表绘词云、组织某个数据集和标签。

### 1:54–5:44　函数一：词性标注后筛选形容词

\`\`\`python
import jieba.posseg as pseg

def get_adjectives(text):
    result = []
    for item in pseg.lcut(text):
        if item.flag.startswith("a"):
            result.append(item.word)
    return result
\`\`\`

老师先打印 \`item.word\` 和 \`item.flag\`，让同学确认每个对象同时含词和词性，再添加判断。课程口头说“等于 a”；实际 jieba 还可能出现 ad、an 等形容词细类，按任务决定用 \`== "a"\` 还是 \`startswith("a")\`。

### 5:44–9:36　函数二：配置 WordCloud 并生成图

词云需要中文字体路径、最多显示词数和背景色。课堂先把词列表用空格 join 成字符串，再调用 \`generate\`：

\`\`\`python
cloud = WordCloud(
    font_path="可显示中文的字体.ttf",
    max_words=100,
    background_color="white",
).generate(" ".join(words))
\`\`\`

若已计算词频，\`generate_from_frequencies(Counter(words))\` 更明确，也不会让词云自己再次切分。

### 9:36–12:02　Matplotlib 只负责显示

老师依次创建画布、\`imshow(cloud, interpolation="bilinear")\`、关闭坐标轴并显示。这里的 bilinear 是图像缩放插值，不决定文字横排还是竖排；词的方向由 WordCloud 的布局参数控制。这是对课堂口头解释的一处技术校正。

### 12:02–18:46　函数三：筛标签、逐句提取、铺平后画图

先读取训练集，筛出 \`label == 1\` 的 sentence 列。对每句话调用 get_adjectives，会得到列表的列表；老师复用 map 与 chain 将它铺平，再交给词云函数。

\`\`\`python
positive_texts = train.loc[train["label"] == 1, "sentence"]
adjectives = chain.from_iterable(map(get_adjectives, positive_texts))
draw_wordcloud(list(adjectives))
\`\`\`

### 18:46–23:49　正负样本都可能出现“相反”词

正样本词云里也可能出现“差”“吵”，负样本里也可能出现“好”。老师用酒店评论解释：一句话可以先抱怨装修噪声，再称赞前台和性价比，最终标签仍为正。词云丢掉上下文和否定关系，所以只能用于找线索，不能直接判定标注错误。

将标签条件改为 0 就能查看负样本；训练/测试与正/负四种场景的结构完全相同，适合封装成可复用函数。

### 23:49–29:29　从图回到人工审核与分析流程

若某些异常词频繁出现，应回到包含这些词的原句，判断是混合情感、标注错误还是敏感内容。老师强调程序不能替代人工抽查。

随后复习文本分析的顺序：先看 X、Y、标签分布、句长、词频和词云，再决定分词、词性、NER、n-gram、长度规范或增强策略。

### 29:29–35:21　直方图、柱状图与 API 复习

柱状图比较离散类别计数；直方图统计连续数值落入各区间的数量。老师复习 \`histplot\` 取代已弃用的 \`distplot\`，map 把同一函数作用于每个元素，stripplot 比较类别下的连续值分布，并通过选择题回顾 zip 的行为。

最后的工程建议是把已经验证的分析代码保存成模板，但必须理解输入、输出和假设，不能只会机械套用。`,

  33: `### 0:00–2:54　文本特征处理包含两件事

老师进入文本特征处理：一是加入 n-gram，让模型看到相邻词组；二是把序列截断或补齐到统一长度。

n-gram 是连续 n 个词或字组成的特征。unigram 保留单词，bigram 再加入相邻两词，trigram 再加入相邻三词。n 越大并不必然越好：组合空间迅速膨胀，短句甚至没有足够长度，因此实践中常用 1–3，课程重点掌握 bigram。

### 2:54–6:49　给组合分配独立特征

老师用“水 / 敲动 / 我心”等词和整数 ID 说明：相邻两词组合后应当被视为新的特征，也需要自己的唯一编号，不能把两个 ID 直接写在一起当十进制数。

加入 bigram 后，原 unigram 通常仍保留：

\`\`\`text = [w1, w2, w3, w4, w5]
bigram = [(w1,w2), (w2,w3), (w3,w4), (w4,w5)]
features = text + bigram
\`\`\`

这样既保留单词身份，又增加局部语序。代价是特征数更多、低频组合更稀疏。

### 6:49–9:46　先补 zip：按位置组合多个迭代对象

老师先用两个不等长列表演示 zip。它返回惰性 zip 对象，转成 list 后得到元组；输出长度等于最短输入，较长列表尾部会被丢弃。

\`\`\`python
list(zip([1,2,3,4,5,6], [2,3,4]))
# [(1,2), (2,3), (3,4)]
\`\`\`

### 9:46–14:13　zip(*nested) 的星号是解包

若直接 \`zip(nested)\`，zip 只收到一个参数，于是每行会成为单元素元组；\`zip(*nested)\` 会把内层列表拆成多个位置参数，再按列组合。

\`\`\`python
rows = [[1,2,3], [3,4,5]]
list(zip(rows))   # [([1,2,3],), ([3,4,5],)]
list(zip(*rows))  # [(1,3), (2,4), (3,5)]
\`\`\`

老师反复用“整体”和“拆出里面每个元素”解释星号。这个概念将在下一节用滑动切片生成 n-gram。`,

  34: `### 0:00–2:59　先把 n-gram 写成可复用函数

老师新建脚本，先写定义：连续 n 个字或词构成小词组特征，帮助计算机观察局部文本规律。unigram、bigram、trigram 分别是连续 1、2、3 个 token；实践里 n 常设为 2 或 3。

函数接收一个 token 列表，课程示例用 \`n = 2\`，随后准备测试输入。

### 2:59–6:52　滑动切片产生错位的列表

核心是生成 n 份错位切片。bigram 时：

\`\`\`python
tokens = [1, 3, 2, 1, 5, 3]
slices = [tokens[i:] for i in range(2)]
# [1,3,2,1,5,3]
# [3,2,1,5,3]
\`\`\`

第一行从位置 0 开始，第二行从位置 1 开始。两行纵向对齐的位置，就是所有相邻二元组。

### 6:52–8:50　zip(*slices) 完成滑动窗口组合

\`\`\`python
grams = list(zip(*slices))
# [(1,3), (3,2), (2,1), (1,5), (5,3)]
\`\`\`

星号把切片列表拆成多个 zip 参数；zip 自动在最短切片结束，因此不会产生越界窗口。若 n 改成 3，就会生成三份错位切片并组合成三元组。

### 8:50–10:01　去重是否正确取决于任务

课堂最后用 set 返回无重复 n-gram，并说明集合显示顺序可能变化：

\`\`\`python
def create_ngrams(tokens, n):
    return set(zip(*(tokens[i:] for i in range(n))))
\`\`\`

若只建“有哪些特征”的词表，set 合理；若要统计 n-gram 频率、保留出现顺序或训练语言模型，就应返回 list 或 Counter，不能提前去重。`,

  35: `### 0:00–0:59　面试题：两段文本怎样比较相似

老师在长度规范前插入一个扩展问题。第一步一定是分词；例如“今天天气很好，适合旅游”和“今天天气不太好，下雨适合睡觉”，只有变成统一特征空间中的数字向量后，才能计算距离或相似度。

### 0:59–3:54　先合并词表，再把句子变成计数向量

课堂简化例子：

- 文本一分词：我、是、黑马、人、我；
- 文本二分词：你、是、黑马。

把两句词去重合并，固定词表顺序 \`[我, 你, 是, 黑马, 人]\`。文本一得到 \`[2,0,1,1,1]\`，文本二得到 \`[0,1,1,1,0]\`。关键是两向量的每一列必须代表同一个词，不能各自建词表。

### 3:54–5:52　欧氏距离是“差值平方和开根”

\`\`\`text
distance = sqrt((2-0)² + (0-1)² + (1-1)² + (1-1)² + (1-0)²)
         = sqrt(6)
\`\`\`

距离越小表示越接近；它不是百分比式“相似度”。老师还提到曼哈顿距离，每种度量都有自己的计算规则。

### 5:52–6:50　加入 bigram 后重新对齐

若加入 bigram，文本一增加“我是、是黑马、黑马人、人我”，文本二增加“你是、是黑马”。把 unigram 与 bigram 一起建立总特征表，再分别计数，最后仍可算欧氏距离。

n-gram 能区分一部分局部语序，但维度会变大。两个向量的距离数值也会随特征尺度变化，所以不能把加入前后的绝对距离直接当作同一标尺比较。

### 6:50–8:33　常见相似度与距离

老师总结通用步骤：分词→合并去重词表→统计频次向量→计算度量，并扩展：

- 欧氏距离：直线距离，越小越近；
- 曼哈顿距离：各维绝对差之和；
- Jaccard 相似度：集合交集大小 / 并集大小；
- 余弦相似度：比较向量方向，越接近 1 越相似。

真实文本常用 TF-IDF 或预训练句向量，而不是裸词频。无论用哪种表示，先保证同一特征空间，再选择与任务匹配的度量。`,

  36: `### 0:00–2:54　为什么一个 batch 必须统一长度

老师举同一批 8 个句子：长度分别为 10、13、9、11 时，不能直接堆成规则矩阵。设目标长度 \`max_len=10\` 后，长序列截断，短序列用 0 补齐。

目标长度不能随手写。应先看前面得到的句长分布和分位数，再平衡信息保留、显存和速度。

### 2:54–5:49　两种实现路线

课堂准备两个版本：

1. 使用 TensorFlow/Keras 的 \`pad_sequences\`；
2. 只用 Python 切片和列表拼接手写。

第三方 API 简洁，手写版帮助真正理解截断与补齐发生在哪里。

### 5:49–8:48　pre/post 是处理前端还是后端

\`\`\`python
pad_sequences(
    sequences,
    maxlen=10,
    truncating="pre",
    padding="pre",
)
\`\`\`

\`truncating="pre"\` 删除序列前端，保留最后 10 个；\`padding="pre"\` 在前端补 0。\`post\` 则在后端截或补。老师反复强调 pre 对应“前面”，post 对应“后面”。

### 8:48–14:43　运行比较默认值与显式参数

老师准备一条超长序列和一条短序列，先显式写 pre，再把参数删掉验证默认行为相同；随后改成 post，观察长句保留开头、短句在末尾补零。

这里最容易误解的是“从前端截断”：意思是删掉前面的元素，不是保留前面的元素。选择哪一端不能机械决定；情感关键词若常在句尾，删尾可能更危险，反之亦然。

### 14:43–18:41　纯 Python：先处理超长序列

\`\`\`python
def normalize(sequences, max_len=10):
    result = []
    for sequence in sequences:
        if len(sequence) > max_len:
            result.append(sequence[:max_len])  # 保留前端
        else:
            ...
    return result
\`\`\`

老师选择保留前 \`max_len\` 个，用 \`sequence[:max_len]\` 截断；如果要保留后端，可用 \`sequence[-max_len:]\`。

### 18:41–22:36　短序列需要补多少个零

进入 else 说明长度小于或等于目标值。补齐数量是：

\`\`\`python
missing = max_len - len(sequence)
result.append(sequence + [0] * missing)
\`\`\`

等长序列的 missing 为 0，不会增加元素；短 3、目标 10 时补 7 个。课堂通过提问让同学写出“原序列 + 若干个零”。

### 22:36–23:36　验证输出并回扣 zip

老师运行手写函数，检查长序列恰好截到指定位置、短序列补到相同长度，最后用 \`zip([1,2,3],[4,5,6])\` 回顾按位置组合会得到 \`[(1,4),(2,5),(3,6)]\`。

补充工程要点：PAD 最好使用专门 ID，并在模型中设置 mask；否则大量 0 也会被当成普通 token 参与注意力或循环计算。`,

  37: `### 0:00–2:58　回译用于扩充少量文本数据

老师把回译定义为文本数据增强：原句译到另一语言，再译回原语言，得到“意思接近、措辞不同”的新样本并加入训练集。课程原来使用的公共翻译接口已经停用，因此不继续跑旧代码；可以使用正规翻译平台、本地模型或大模型接口。

课堂建议中间语言可选择资源相对少的语言，希望增加表述变化。技术上没有“小语种一定更好”的保证，应该用语义保持率、标签一致率和多样性实测。

### 2:58–4:55　优点、短文本重复与语义失真

优点是流程简单，新句通常可读；缺点是短句如“我爱你”往往原样返回，不能有效扩大特征空间。可串联少量语言增加变化，但老师提醒中间翻译通常不要超过约三次：链太长会降低效率并像传话游戏一样累积语义失真。

### 4:55–6:52　旧接口代码只保留思路

旧示例先指定源语言和目标语言，把中文译成英文，再把英文译回中文。接口失效时不应寻找来历不明的密钥或把 API Key 写进代码库；应使用仍受支持的服务、环境变量与费用上限。

### 6:52–10:44　用提示词要求多轮改写

老师现场让大模型把“你今天真好看”依次翻到多种语言后回到中文，并要求结果不能与原句完全相同、核心含义不变。示例得到“你今天的模样也太惊艳了吧”等表达。

随后老师进一步要求扩写为十句话，展示提示词约束会显著影响结果；并借此解释提示词工程的价值。需要校正的是：直接让模型生成十条中文改写，属于释义生成，不一定真的执行了可验证的多语言回译，但同样可作为数据增强候选。

### 10:44–11:40　增强结果不能未经检查就入库

一条原始语料可产生多条变体，但数量多不等于有效。应检查：

- 与原文是否重复或过度相似；
- 否定、数字、实体和情感标签是否改变；
- 多条生成结果是否彼此重复；
- 是否引入不自然、冒犯或事实错误内容。

高风险样本可用人工审核；普通样本也应抽检并做自动相似度、规则和去重过滤。

### 11:40–13:30　全章收束

老师回顾三项内容：n-gram 把连续若干词作为整体特征；长度规范把序列截断或补齐到统一长度；回译通过表达变化增加语料。

最后再次强调回译的边界：短文本容易重复，多次翻译容易失真；如果原始数据已经足够，不必为了“做了增强”而强行制造低质量样本。是否使用增强，应由验证集对照实验决定。`,
};

function esc(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function wrap(text, max = 11) {
  const chars = [...text];
  const lines = [];
  for (let i = 0; i < chars.length; i += max) lines.push(chars.slice(i, i + max).join(""));
  return lines;
}

function textLines(text, x, y, options = {}) {
  const { size = 24, color = "#16324F", weight = 600, anchor = "middle", max = 11, gap = 34 } = options;
  return wrap(text, max).map((line, i) =>
    `<text x="${x}" y="${y + i * gap}" text-anchor="${anchor}" font-family="-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif" font-size="${size}" font-weight="${weight}" fill="${color}">${esc(line)}</text>`
  ).join("\n");
}

function makeSvg(lesson, index) {
  const colors = ["#E7F3FF", "#EAF8F0", "#FFF4D9", "#F6EAFF", "#FFECE7"];
  const strokes = ["#3B82C4", "#31966F", "#D69424", "#8C62C7", "#D6674F"];
  const nodes = lesson.diagram;
  const count = nodes.length;
  const gap = 34;
  const margin = 70;
  const usable = 1200 - margin * 2 - gap * (count - 1);
  const width = usable / count;
  const y = 245;
  const height = 190;
  const cards = nodes.map((node, i) => {
    const x = margin + i * (width + gap);
    const center = x + width / 2;
    const lineCount = wrap(node, 10).length;
    const startY = y + height / 2 - ((lineCount - 1) * 34) / 2 + 8;
    return `<g>
      <rect x="${x}" y="${y}" width="${width}" height="${height}" rx="24" fill="${colors[i]}" stroke="${strokes[i]}" stroke-width="3"/>
      <circle cx="${center}" cy="${y + 40}" r="19" fill="${strokes[i]}"/>
      <text x="${center}" y="${y + 47}" text-anchor="middle" font-family="sans-serif" font-size="19" font-weight="700" fill="white">${i + 1}</text>
      ${textLines(node, center, startY + 24, { size: 23, max: 10 })}
    </g>`;
  }).join("\n");
  const arrows = nodes.slice(0, -1).map((_, i) => {
    const x1 = margin + i * (width + gap) + width + 6;
    const x2 = margin + (i + 1) * (width + gap) - 8;
    return `<line x1="${x1}" y1="${y + height / 2}" x2="${x2}" y2="${y + height / 2}" stroke="#60788E" stroke-width="4" marker-end="url(#arrow)"/>`;
  }).join("\n");
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="680" viewBox="0 0 1200 680" role="img" aria-labelledby="title desc">
  <title id="title">${esc(lesson.title)}</title>
  <desc id="desc">${esc(nodes.join("，然后"))}</desc>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#F8FBFF"/><stop offset="1" stop-color="#F7F4FF"/></linearGradient>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#60788E"/></marker>
  </defs>
  <rect width="1200" height="680" rx="34" fill="url(#bg)"/>
  <text x="70" y="62" font-family="sans-serif" font-size="18" font-weight="700" fill="#3B6A91">第 ${String(index).padStart(2, "0")} 节 · 概念图</text>
  ${textLines(lesson.title, 70, 112, { size: 34, color: "#102A43", weight: 760, anchor: "start", max: 28, gap: 42 })}
  ${arrows}
  ${cards}
  <rect x="70" y="512" width="1060" height="105" rx="20" fill="#102A43"/>
  ${textLines(lesson.plain, 600, 553, { size: 21, color: "#FFFFFF", weight: 500, max: 42, gap: 30 })}
  <text x="1130" y="650" text-anchor="end" font-family="sans-serif" font-size="16" fill="#6C8194">原创示意图</text>
</svg>`;
}

function makeMermaidFlow(items) {
  const labels = items.map((item, index) =>
    `    N${index}["${String(item).replaceAll('"', "'")}"]`
  );
  const edges = items.slice(0, -1).map(
    (_, index) => `    N${index} --> N${index + 1}`
  );
  return `\`\`\`mermaid
flowchart LR
${labels.join("\n")}
${edges.join("\n")}
\`\`\``;
}

function makeExtraVisual(index) {
  if (index === 1) {
    return `### 五类任务依赖图

\`\`\`mermaid
flowchart TB
    A["原始文本与标签"] --> B["基本方法<br/>分词 / POS / NER"]
    B --> C["数值表示<br/>ID / One-Hot / Embedding"]
    A --> D["语料分析<br/>标签 / 长度 / 词频"]
    C --> E["特征处理<br/>n-gram / 长度规范"]
    D --> E
    E --> F["数据增强与可训练 X、Y"]
\`\`\``;
  }
  if ([3, 4, 5].includes(index)) {
    return `### 三种分词模式对照图

\`\`\`mermaid
classDiagram
    class 精确模式 {
      主要切分路径
      候选通常不重叠
      一般文本分析
    }
    class 全模式 {
      枚举尽可能多候选
      允许重叠
      高召回候选发现
    }
    class 搜索模式 {
      精确切分再补短词
      长短词共同建索引
      搜索召回
    }
\`\`\``;
  }
  if (index === 8) {
    return `### 分词、POS 与 NER 的层级关系

\`\`\`mermaid
flowchart LR
    A["原始句子"] --> B["分词<br/>确定 token 边界"]
    B --> C["POS<br/>标注语法角色"]
    B --> D["NER<br/>识别人 / 地 / 机构"]
    C -. "可作为辅助信息" .-> D
\`\`\``;
  }
  if ([9, 10, 11, 12].includes(index)) {
    return `### 词表到模型输入的时序

\`\`\`mermaid
sequenceDiagram
    participant T as 训练语料
    participant V as 固定词表
    participant E as 编码器
    participant M as 模型
    T->>V: 建立 word → id
    V->>E: 保存映射
    E->>M: ID / One-Hot / Embedding
    Note over V,M: 预测时必须复用同一映射
\`\`\``;
  }
  if ([13, 14].includes(index)) {
    return `### CBOW 与 Skip-Gram 方向对照

\`\`\`mermaid
flowchart LR
    C1["左上下文 + 右上下文"] --> C2["CBOW"] --> C3["预测中心词"]
    S1["中心词"] --> S2["Skip-Gram"] --> S3["分别预测上下文词"]
\`\`\`

### 一次训练迭代时序

\`\`\`mermaid
sequenceDiagram
    participant W as 滑动窗口
    participant N as 浅层网络
    participant L as 损失函数
    participant P as 词向量参数
    W->>N: 构造输入与标签
    N->>L: 输出词表预测
    L->>P: 反向传播更新
    P-->>W: 移动到下一窗口
\`\`\``;
  }
  if ([15, 16, 17, 18].includes(index)) {
    return `### FastText 实验生命周期

\`\`\`mermaid
flowchart LR
    A["清洗、分词语料"] --> B["固定配置训练"]
    B --> C["保存 .bin"]
    C --> D["加载与近邻检查"]
    D --> E["下游任务评估"]
    E -->|"一次改一个参数"| B
\`\`\``;
  }
  if ([19, 20].includes(index)) {
    return `### 预训练与任务内 Embedding 关系

\`\`\`mermaid
flowchart TB
    A["大语料 Word2Vec / FastText"] --> B["预训练向量矩阵"]
    B --> C["初始化 nn.Embedding"]
    D["任务 token ID"] --> C
    C --> E["下游模型"]
    E --> F["任务损失"]
    F -. "可选择微调" .-> C
\`\`\``;
  }
  if ([22, 23, 24, 25].includes(index)) {
    return `### 从分析发现到处理决策

\`\`\`mermaid
flowchart LR
    A["精确计数 / 分布图"] --> B{"是否异常或偏斜？"}
    B -->|"否"| C["保留基线流程"]
    B -->|"是"| D["回看原始样本与采集来源"]
    D --> E["分层 / 权重 / 截断 / 清洗候选"]
    E --> F["在验证集做对照实验"]
\`\`\``;
  }
  if ([26, 27].includes(index)) {
    return `### 从嵌套语料到词汇统计的数据流

\`\`\`mermaid
flowchart LR
    A["DataFrame 的 sentence 列"] --> B["map(jieba.lcut)"]
    B --> C["列表的列表"]
    C --> D["chain.from_iterable"]
    D --> E["单一惰性词流"]
    E --> F["Counter：保留频率"]
    F --> G["总 token 数 / 词汇量 / OOV 分析"]
\`\`\``;
  }
  if (index === 28) {
    return `### 词云模块的职责关系

\`\`\`mermaid
classDiagram
    class 语料选择器 {
      选择训练集或测试集
      按 label 筛样本
    }
    class 形容词提取器 {
      jieba_posseg()
      filter_flag_a()
    }
    class 频率统计器 {
      Counter(words)
    }
    class 词云渲染器 {
      中文字体
      最大词数
      图像布局
    }
    语料选择器 --> 形容词提取器
    形容词提取器 --> 频率统计器
    频率统计器 --> 词云渲染器
\`\`\``;
  }
  if ([29, 30].includes(index)) {
    return `### zip 解包与 n-gram 滑窗

\`\`\`mermaid
sequenceDiagram
    participant T as token 列表
    participant S as 错位切片
    participant Z as zip(*slices)
    participant G as n-gram 结果
    T->>S: 生成 tokens[0:]、tokens[1:]…
    S->>Z: 星号拆成多个可迭代参数
    Z->>G: 按列组合，到最短输入停止
    Note over T,G: n=2 得 bigram，n=3 得 trigram
\`\`\``;
  }
  if (index === 31) {
    return `### 文本相似度的共同特征空间

\`\`\`mermaid
flowchart TB
    A["文本 A"] --> C["分词 / n-gram"]
    B["文本 B"] --> C
    C --> D["合并并固定特征表"]
    D --> E["A 的同维计数向量"]
    D --> F["B 的同维计数向量"]
    E --> G{"选择度量"}
    F --> G
    G --> H["欧氏 / 曼哈顿 / 余弦 / Jaccard"]
\`\`\``;
  }
  if (index === 32) {
    return `### 定长处理决策流程

\`\`\`mermaid
flowchart TB
    A["一条 token ID 序列"] --> B{"长度与 max_len 比较"}
    B -->|"大于"| C{"保留前端还是后端？"}
    C --> D["切片截断"]
    B -->|"小于"| E{"PAD 放前还是放后？"}
    E --> F["补专用 PAD ID"]
    B -->|"等于"| G["原样保留"]
    D --> H["固定长度矩阵"]
    F --> H
    G --> H
    H --> I["模型中用 mask 忽略 PAD"]
\`\`\``;
  }
  if (index === 33) {
    return `### 回译增强时序与质量闸门

\`\`\`mermaid
sequenceDiagram
    participant D as 原始样本+标签
    participant T1 as 中间语言模型
    participant T2 as 回译模型
    participant Q as 质量过滤器
    participant A as 增强数据集
    D->>T1: 原语言 → 中间语言
    T1->>T2: 中间文本 → 原语言
    T2->>Q: 候选改写
    Q->>Q: 去重、语义、否定、实体、标签检查
    alt 检查通过
      Q->>A: 加入并保留来源记录
    else 检查失败
      Q-->>D: 丢弃候选
    end
\`\`\``;
  }
  return "";
}

function makeNote(lesson, index) {
  const file = `${String(index).padStart(2, "0")}-${lesson.slug}.md`;
  const prev = index > 1 ? lessons[index - 2] : null;
  const next = index < lessons.length ? lessons[index] : null;
  const nav = [
    prev ? `[← 上一节：${String(index - 1).padStart(2, "0")} ${prev.title}](./${String(index - 1).padStart(2, "0")}-${prev.slug}.md)` : "← 已是第一节",
    "[返回总目录](./README.md)",
    next ? `[下一节：${String(index + 1).padStart(2, "0")} ${next.title} →](./${String(index + 1).padStart(2, "0")}-${next.slug}.md)` : "已是最后一节 →",
  ].join(" · ");
  const bullets = lesson.points.map(x => `- ${x}`).join("\n");
  return `# 第 ${index} 节：${lesson.title}

> 笔记编号 ${index}/33 · 对应原视频 P${lesson.p} · [打开这一集](https://www.bilibili.com/video/BV14mdfBDE4Q?p=${lesson.p})

${nav}

## 这节解决什么问题

${lesson.plain}

![第 ${index} 节原创概念图](./diagrams/${String(index).padStart(2, "0")}-concept.svg)

图要从左向右读。每个方框都是数据的一次变化，不是四个互不相关的名词。

## 辅助流程图

${makeMermaidFlow(lesson.diagram)}

${makeExtraVisual(index)}

## 老师原声整理稿（按讲解顺序）

${detailedArticles[lesson.p] ?? "本节正文正在依据完整音轨逐段校正；下方原声记录已可用于核查。"}

## 完整原声逐段记录

[查看本节按时间戳整理的完整音轨转写](./transcripts/p${String(lesson.p).padStart(3, "0")}.md)

这份记录用于核查老师讲过的内容是否遗漏；正文会纠正口误与语音识别中的技术术语。

## 零基础先记住

${bullets}

## 最小可运行代码

在项目根目录运行下面代码。课程原理的标准库版本集中在 [text_preprocessing_from_scratch](../../text_preprocessing_from_scratch/README.md)；需要 jieba、PyTorch、FastText 等的示例，请先按代码注释安装依赖。

\`\`\`python
${lesson.code}
\`\`\`

### 输入和输出怎么看

${lesson.output}

## 最容易踩的坑

${lesson.pit}

## 本节知识链

\`${lesson.diagram.join(" → ")}\`

如果中间任意一个箭头说不清楚，就回到图上，用代码中的一个具体值手算一遍；能预测输出，才算真正理解。

## 自测

**问题：${lesson.quiz}**

<details>
<summary>点开核对答案</summary>

${lesson.answer}

</details>

## 学完检查

- [ ] 我能不用术语，用自己的话解释“这节解决什么问题”
- [ ] 我能在运行前大致猜出代码输出
- [ ] 我知道本节方法不适用或容易出错的情况
- [ ] 我能回答自测题，而不只是记住答案

${nav}
`;
}

lessons.forEach((lesson, i) => {
  const index = i + 1;
  fs.writeFileSync(path.join(diagramDir, `${String(index).padStart(2, "0")}-concept.svg`), makeSvg(lesson, index));
  fs.writeFileSync(path.join(outDir, `${String(index).padStart(2, "0")}-${lesson.slug}.md`), makeNote(lesson, index));
});

const stages = [
  ["一、基础方法", 1, 8],
  ["二、文本数值表示", 9, 21],
  ["三、语料分析", 22, 28],
  ["四、特征处理", 29, 32],
  ["五、数据增强", 33, 33],
];

const indexLines = stages.map(([name, start, end]) => {
  const links = lessons.slice(start - 1, end).map((lesson, offset) => {
    const index = start + offset;
    return `${index}. [${lesson.title}](./${String(index).padStart(2, "0")}-${lesson.slug}.md)（原视频 P${lesson.p}）`;
  }).join("\n");
  return `## ${name}\n\n${links}`;
}).join("\n\n");

const readme = `# 文本预处理：33 节零基础学习笔记

这套笔记覆盖课程原视频 **P5–P37**，按学习顺序重新编号为 **第 1–33 节**。P38 已进入循环神经网络，不属于本专题。

每一节都包含：问题目标、零基础解释、原创 SVG 示意图、最小可运行代码、输入输出说明、易错点、知识链、自测与答案。图片均为本项目原创绘制，**没有使用课程截图**。

## 建议学习方法

1. 先看图并用自己的话复述箭头。
2. 不看输出，先手算代码会打印什么。
3. 运行代码，若结果不同，找出自己漏掉的规则。
4. 做自测并回答“什么时候会用错”。
5. 每学完一个阶段，再把该阶段的代码连起来做一次。

${indexLines}

## 配套代码

- [纯 Python 原理实现与测试](../../text_preprocessing_from_scratch/README.md)
- 运行测试：\`python3 -m unittest text_preprocessing_from_scratch.test_core -v\`

## 课程内容中的版本提醒

- 课程演示的部分 Keras/TensorFlow、Seaborn 和翻译接口会随时间变化；笔记保留原理，并标出更稳妥的现代写法。
- FastText 的重要能力之一是用字符 n-gram 组合未登录词（OOV）向量，不只是“找一个最相近的已知词”。
- TensorBoard Projector 是降维后的探索工具，图上的二维距离不能代替原空间的量化评估。
`;

fs.writeFileSync(path.join(outDir, "README.md"), readme);
console.log(`generated ${lessons.length} notes and ${lessons.length} SVG diagrams in ${outDir}`);

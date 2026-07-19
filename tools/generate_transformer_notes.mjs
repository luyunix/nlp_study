import fs from "node:fs";
import path from "node:path";

const root = "/Users/lyn/Documents/nlp_study";
const outDir = path.join(root, "notes", "transformer");
fs.mkdirSync(outDir, { recursive: true });

const lessons = [
  {
    p: 106, slug: "transformer-origin", title: "Transformer 的由来：为什么不再逐词递归",
    plain: "RNN 必须按时间步依次传递隐藏状态，训练难并行，远距离信息也要走很长的路。Transformer 用注意力让任意两个位置直接交互。",
    points: ["2017 年论文 Attention Is All You Need 提出 Transformer", "训练时多个位置可以并行计算", "长距离交互路径更短，但标准注意力仍有长度平方级成本"],
    code: `tokens = ["我", "喜欢", "自然", "语言", "处理"]\nfor i, token in enumerate(tokens):\n    visible = tokens if True else tokens[: i + 1]\n    print(token, "可以直接比较：", visible)`,
    output: "这不是神经网络实现，而是先体会差异：每个位置可以直接与整段序列比较，不必等前一个隐藏状态传来。",
    pit: "Transformer 不等于大语言模型；它是网络架构。GPT、BERT 是基于它构建并用不同目标训练的模型家族。",
    quiz: "Transformer 能建模长距离关系，是否意味着能无限处理长文本？", answer: "不能。上下文窗口、显存和标准注意力的 O(L²) 计算仍会限制长度。",
    chain: ["RNN 串行瓶颈", "全局注意力", "位置并行", "Transformer"]
  },
  {
    p: 107, slug: "transformer-architecture-text", title: "总体架构文字版：Encoder 理解，Decoder 生成",
    plain: "源句先由 Encoder 变成带上下文的 memory；Decoder 一边看已经给出的目标前缀，一边查询 memory，预测下一个词。",
    points: ["输入部分：Embedding + Positional Encoding", "Encoder 与 Decoder 都由重复层堆叠", "Generator 把 Decoder 隐藏状态映射为目标词表概率"],
    code: `shapes = {\n    "src_ids": "[B, Ls]",\n    "memory": "[B, Ls, D]",\n    "decoder": "[B, Lt, D]",\n    "log_probs": "[B, Lt, Vt]",\n}\nfor name, shape in shapes.items():\n    print(name, shape)`,
    output: "输出四个关键形状。后面每学一个组件，都要问它接收哪一种形状、返回哪一种形状。",
    pit: "训练 Decoder 时输入不是完整答案原封不动地预测自己，而是目标序列右移后预测下一位置。",
    quiz: "memory 的序列长度跟源句还是目标句一致？", answer: "跟源句一致，形状是 [B, Ls, D]；它是 Encoder 对源序列每个位置的表示。",
    chain: ["源词元", "Encoder", "memory", "Decoder", "词表概率"]
  },
  {
    p: 108, slug: "transformer-diagram-upper", title: "架构图上半部分：看懂 Encoder 数据流",
    plain: "源词 ID 经过词嵌入和位置编码，进入若干相同结构的 EncoderLayer；每层依次做自注意力和前馈网络，并用残差与归一化包住子层。",
    points: ["Self-Attention 中 Q=K=V，都来自当前源序列表示", "注意力混合不同位置的信息", "FFN 独立加工每个位置内部的特征"],
    code: `flow = ["src ids", "Embedding + PE", "Self-Attention", "FFN", "memory"]\nprint(" -> ".join(flow))`,
    output: "这条主线要能背着图口述。真实模型中 Attention 和 FFN 外面还各有残差、Dropout 与 LayerNorm。",
    pit: "Encoder 自注意力不是只看前文；非生成式编码通常允许一个位置同时看左右文，只屏蔽 PAD。",
    quiz: "EncoderLayer 中是谁负责跨位置混合，谁负责特征维变换？", answer: "Self-Attention 跨位置混合；Position-wise FFN 在每个位置独立变换特征。",
    chain: ["源 ID", "嵌入+位置", "自注意力", "FFN", "memory"]
  },
  {
    p: 109, slug: "transformer-diagram-lower", title: "架构图下半部分：Decoder 为什么有两种注意力",
    plain: "Decoder 先对目标前缀做带因果掩码的自注意力，再用目标状态作 Q、Encoder memory 作 K/V 做交叉注意力。",
    points: ["Masked Self-Attention 防止训练时偷看未来目标词", "Cross-Attention 让目标侧查询源句信息", "Decoder 比 Encoder 多一个注意力子层"],
    code: `roles = {\n    "decoder self-attn": "Q=tgt, K=tgt, V=tgt",\n    "cross-attn": "Q=tgt, K=memory, V=memory",\n}\nfor name, value in roles.items():\n    print(name, ":", value)`,
    output: "两行输出必须形成肌肉记忆。出错时首先检查 Q/K/V 来源和对应 mask。",
    pit: "交叉注意力的 Q 长度是 Lt，K/V 长度是 Ls，所以注意力分数是 [B,h,Lt,Ls]，两种长度不必相同。",
    quiz: "为什么 Cross-Attention 的 K 和 V 都来自 memory？", answer: "Decoder 用目标状态提出查询，去源序列表示中匹配位置并读取对应信息。",
    chain: ["目标前缀", "因果自注意力", "查询 memory", "FFN", "下一个词"]
  },
  {
    p: 110, slug: "transformer-learning-roadmap", title: "从零实现路线：先零件，再整机",
    plain: "Transformer 代码看起来大，是因为许多小组件嵌套。按输入组件、注意力、子层、层、堆叠、整模逐级实现，每次只验证一个接口。",
    points: ["每一步先写形状契约，再写计算", "组件测试通过后再组装，定位错误更容易", "不要一上来照抄整份模型代码"],
    code: `roadmap = ["Embedding/PE", "Mask/Attention", "MHA/FFN", "Norm/Residual",\n           "Encoder/Decoder", "Generator/Full model"]\nfor step, name in enumerate(roadmap, 1):\n    print(step, name)`,
    output: "输出六个实现阶段。配套代码正是按这个依赖顺序组织。",
    pit: "能跑通整模不等于理解。初学时必须能说出每层输入输出形状，以及哪个维度被改变。",
    quiz: "为什么应先测试 subsequent_mask，再测试完整 Decoder？", answer: "Decoder 依赖 mask；先隔离验证能避免最终出错时无法区分是掩码还是层连接问题。",
    chain: ["输入组件", "核心计算", "子层包装", "层与堆叠", "整模测试"]
  },
  {
    p: 111, slug: "token-embedding-code", title: "Token Embedding：把 ID 查成向量",
    plain: "Embedding 是一张 V×D 的可训练表。每个 token ID 是行号，查表后得到 D 维向量，再乘 √D 平衡与位置编码相加时的量级。",
    points: ["输入 [B,L]，输出 [B,L,D]", "ID 必须是整数且范围在 0 到 vocab_size-1", "缩放不改变形状，只改变数值大小"],
    code: `import torch\nfrom transformer_from_scratch.model import Embeddings\nlayer = Embeddings(vocab_size=20, d_model=8)\nids = torch.tensor([[1, 5, 3], [2, 4, 0]])\nprint(layer(ids).shape)`,
    output: "输出 torch.Size([2, 3, 8])：2 个样本、每个 3 个词、每词 8 维。",
    pit: "vocab_size=20 时 ID 20 已越界；最大合法 ID 是 19。",
    quiz: "输入形状 [32,50]、d_model=512，输出形状是什么？", answer: "[32,50,512]，Embedding 在末尾新增特征维。",
    chain: ["token ID", "查 Embedding 表", "乘 √D", "[B,L,D]"]
  },
  {
    p: 112, slug: "positional-encoding-principle", title: "正弦位置编码原理：让模型知道先后顺序",
    plain: "自注意力本身只比较内容，不天然知道第几个词。位置编码给每个位置一组不同频率的正弦和余弦值，再与词向量相加。",
    points: ["偶数特征维用 sin，奇数特征维用 cos", "低维变化较快，高维变化较慢，形成多尺度位置指纹", "固定编码不训练，也能外推到训练时未见的较长位置"],
    code: `import math\nfor pos in range(3):\n    sin0 = math.sin(pos / (10000 ** (0 / 8)))\n    cos1 = math.cos(pos / (10000 ** (0 / 8)))\n    print(pos, round(sin0, 3), round(cos1, 3))`,
    output: "位置 0 得到 sin=0、cos=1；后续位置值逐渐变化。完整编码会在多组频率上重复这种配对。",
    pit: "位置编码不是把位置编号直接加到每个维度；那会让所有特征同幅度漂移，难以表达多尺度关系。",
    quiz: "为什么没有位置编码时，注意力难区分“狗咬人”和“人咬狗”？", answer: "相同词集合仅改变排列，而纯注意力没有额外顺序信号来区分位置。",
    chain: ["位置 pos", "多频率 sin/cos", "位置向量", "加到词向量"]
  },
  {
    p: 113, slug: "positional-encoding-summary", title: "位置编码总结：内容回答“是什么”，位置回答“在哪儿”",
    plain: "词嵌入与位置编码形状相同，逐元素相加。相加后的表示同时携带词义和位置，并保持所有位置可并行处理。",
    points: ["两者都是 [B,L,D] 或可广播到该形状", "位置表常是 [1,max_len,D]，沿 batch 广播", "相加而非拼接，因此 d_model 不变"],
    code: `import torch\ncontent = torch.randn(2, 4, 8)\nposition = torch.randn(1, 4, 8)\ncombined = content + position\nprint(combined.shape)`,
    output: "输出 [2,4,8]。position 的 batch 维为 1，会自动广播给两个样本。",
    pit: "广播能运行不代表语义正确。必须确认长度维和特征维对齐，而不是把 [L,D] 误加到错误轴。",
    quiz: "为什么位置编码通常使用 [1,L,D] 而不是为每个 batch 复制一份？", answer: "同一位置规则对所有样本相同，保留 batch=1 可通过广播节省存储。",
    chain: ["内容向量", "位置向量", "广播相加", "顺序感知表示"]
  },
  {
    p: 114, slug: "positional-encoding-code", title: "PositionalEncoding 代码：预计算并注册 buffer",
    plain: "初始化时一次算好最大长度的位置表；前向传播只截取当前 L 行，加到输入上，再做 Dropout。",
    points: ["register_buffer 让 pe 随模型迁移设备并进入 state_dict", "buffer 不是可训练 Parameter，不接收梯度更新", "切片 self.pe[:,:L] 必须与输入长度一致"],
    code: `import torch\nfrom transformer_from_scratch.model import PositionalEncoding\nlayer = PositionalEncoding(d_model=8, dropout=0.0, max_len=16)\nx = torch.zeros(2, 5, 8)\ny = layer(x)\nprint(y.shape, "pe是否是参数：", "pe" in dict(layer.named_parameters()))`,
    output: "形状保持 [2,5,8]，pe是否是参数为 False；但它能在 named_buffers 中找到。",
    pit: "把 pe 写成普通属性张量，model.to('cuda') 时它不会自动迁移，前向会产生设备不一致错误。",
    quiz: "buffer 和 Parameter 最关键的训练差异是什么？", answer: "Parameter 参与梯度优化；buffer 保存模型状态并随设备迁移，但默认不被优化器更新。",
    chain: ["预计算位置表", "register_buffer", "按 L 切片", "相加+Dropout"]
  },
  {
    p: 115, slug: "positional-encoding-test", title: "位置编码测试：重点检查形状、广播和确定性",
    plain: "测试不仅是“没有报错”，还要证明输出形状不变、同位置编码跨 batch 一致，并在 dropout=0 时结果可重复。",
    points: ["输入 [B,L,D]，输出必须同形", "零输入能直接观察位置编码", "测试时关闭 Dropout，避免随机性干扰断言"],
    code: `import torch\nfrom transformer_from_scratch.model import PositionalEncoding\ntorch.manual_seed(0)\nlayer = PositionalEncoding(8, dropout=0.0, max_len=10)\ny = layer(torch.zeros(2, 4, 8))\nprint(y.shape)\nprint(torch.allclose(y[0], y[1]))`,
    output: "输出 [2,4,8] 和 True，说明同一位置表被广播给两个 batch。",
    pit: "训练模式下 dropout>0 时两个 batch 的随机丢弃可能不同，不能据此说位置编码计算错了。",
    quiz: "为什么用全零输入测试位置编码很直观？", answer: "输出就等于位置编码本身（若 dropout=0），不受词向量数值干扰。",
    chain: ["构造零输入", "关闭 Dropout", "前向", "检查形状与批次一致"]
  },
  {
    p: 116, slug: "upper-triangular-matrix", title: "上三角矩阵：标记当前位置之后的未来",
    plain: "因果生成中，第 i 个位置不能看 i 后面的词。矩阵第 i 行、第 j 列表示 Query i 是否能看 Key j；j>i 正好是对角线上方。",
    points: ["torch.triu(..., diagonal=1) 取严格上三角", "diagonal=0 会连当前位置也选中", "这一步通常先标出“要屏蔽”的未来区域"],
    code: `import torch\nfuture = torch.triu(torch.ones(4, 4, dtype=torch.int), diagonal=1)\nprint(future)`,
    output: "对角线上方为 1，其余为 0；这些 1 代表未来位置。",
    pit: "不要只背上三角或下三角。先明确行是 Query、列是 Key，否则坐标定义一换就会判断反。",
    quiz: "为什么 diagonal=1 而不是 0？", answer: "当前位置可以看自己；只应从右上方的下一个位置开始屏蔽。",
    chain: ["Query/Key 坐标", "j>i", "严格上三角", "未来区域"]
  },
  {
    p: 117, slug: "lower-triangular-matrix", title: "下三角可见区：只看自己和过去",
    plain: "把未来区域逻辑取反，就得到允许注意的位置：第 i 行只有第 0 到 i 列为 True，形状像下三角。",
    points: ["项目约定 True=允许、False=屏蔽", "subsequent_mask 返回 [1,L,L] 便于 batch 广播", "不同库可能采用相反布尔语义，必须看接口约定"],
    code: `from transformer_from_scratch.model import subsequent_mask\nprint(subsequent_mask(4)[0].int())`,
    output: "第一行只有第 0 列为 1，最后一行四列全为 1。",
    pit: "Mask 语义最常见的坑是 1/0 反了。本项目在 attention 中用 mask==0 的位置填极小值。",
    quiz: "长度 4 时，第 2 个位置（从 0 开始）能看哪些列？", answer: "第 0、1、2 列；不能看第 3 列。",
    chain: ["未来上三角", "逻辑取反", "下三角 True", "允许自己+过去"]
  },
  {
    p: 118, slug: "mask-visualization", title: "因果 Mask 可视化：学会读横纵轴",
    plain: "热力图的每个格子都是一个 Query-Key 配对。逐行读：某行亮起的列，就是该目标位置能读取的信息。",
    points: ["行对应输出/Query 位置", "列对应被查询的 Key 位置", "理想因果 mask 的可见区域逐行增加一个"],
    code: `from transformer_from_scratch.model import subsequent_mask\nmask = subsequent_mask(5)[0]\nfor row in mask:\n    print(" ".join("■" if x else "·" for x in row.tolist()))`,
    output: "终端会画出一个由 ■ 组成的下三角；· 是不可见的未来。",
    pit: "只画 mask 不验证 attention 权重仍不够；还应确认被屏蔽位置经过 softmax 后概率确实为 0。",
    quiz: "热力图第一行为什么通常只有一个可见格？", answer: "生成第一个目标位置时只能看当前位置/起始标记，不能读取任何未来目标词。",
    chain: ["布尔矩阵", "行=Query", "列=Key", "下三角可见区"]
  },
  {
    p: 119, slug: "masked-fill", title: "masked_fill：在 Softmax 前把未来分数压到极小",
    plain: "注意力先得到匹配分数。对不允许的位置填入负无穷附近的极小值，Softmax 后这些位置的概率就趋近 0。",
    points: ["屏蔽必须发生在 Softmax 之前", "浮点实现常用 torch.finfo(dtype).min", "最终每行未屏蔽位置的概率仍应和为 1"],
    code: `import torch\nscores = torch.tensor([[1.0, 2.0, 3.0]])\nmask = torch.tensor([[True, True, False]])\nmasked = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)\nprint(masked)\nprint(torch.softmax(masked, dim=-1))`,
    output: "第三个分数变为极小值，对应概率为 0；前两个位置重新归一化。",
    pit: "Softmax 后再把概率乘 0 会让整行概率和小于 1，除非再次归一化；标准做法是先改分数。",
    quiz: "为什么不能把被屏蔽分数简单填成 0？", answer: "0 经过指数仍是 1，可能获得非零概率；必须填足够小的负值。",
    chain: ["原始 scores", "mask==False", "填极小值", "Softmax 概率为 0"]
  },
  {
    p: 120, slug: "scaled-dot-product-attention", title: "缩放点积注意力：Q 找谁，V 提供什么",
    plain: "Q 与 K 点积得到匹配程度，除以 √dₖ 稳定数值，经过 mask 和 Softmax 得到权重，最后对 V 加权求和。",
    points: ["scores=QKᵀ/√dₖ，形状 [...,Lq,Lk]", "Softmax 沿 Key 维进行", "输出长度跟 Q 一致，特征来自 V"],
    code: `import torch\nfrom transformer_from_scratch.model import attention\nq = k = v = torch.randn(1, 3, 4)\nout, weights = attention(q, k, v)\nprint(out.shape, weights.shape)\nprint(weights.sum(dim=-1))`,
    output: "输出 [1,3,4]，权重 [1,3,3]；每个 Query 对所有 Key 的权重和约为 1。",
    pit: "忘记除 √dₖ 时，高维点积容易变大，让 Softmax 极端饱和、梯度变小。",
    quiz: "若 Q 长度 5、K/V 长度 7，权重矩阵最后两维是什么？", answer: "[5,7]，每个 Query 位置对 7 个 Key 分配权重。",
    chain: ["QKᵀ", "缩放", "Mask+Softmax", "权重×V"]
  },
  {
    p: 121, slug: "multi-head-attention-principle-upper", title: "多头注意力原理上：把特征空间分给多个头",
    plain: "一个注意力头只能形成一套匹配模式。多头先用不同线性投影把表示映射到多个子空间，让各头并行学习不同关系。",
    points: ["d_model 必须能被头数 h 整除", "每头维度 dₖ=d_model/h", "拆头不是把同一个向量机械复制 h 份"],
    code: `d_model, heads = 512, 8\nassert d_model % heads == 0\nd_k = d_model // heads\nprint("每头维度：", d_k, "合计：", heads * d_k)`,
    output: "每头 64 维，8 个头合计仍为 512 维。",
    pit: "头数越多不保证越好；每头太窄会限制表示能力，并增加调参和运行开销。",
    quiz: "d_model=510、h=8 能直接均匀拆头吗？", answer: "不能，510 不能被 8 整除；需要修改维度或头数。",
    chain: ["Q/K/V 投影", "拆成 h 头", "不同子空间", "并行注意力"]
  },
  {
    p: 122, slug: "multi-head-attention-principle-lower", title: "多头注意力原理下：拆头、计算、再合头",
    plain: "多头完整过程有五步：投影 Q/K/V，重排成头维，每头独立注意力，拼回所有头，再做一次输出投影。",
    points: ["[B,L,D] view 为 [B,L,h,dₖ]", "transpose 得到 [B,h,L,dₖ] 便于并行", "合头后回到 [B,L,D]，模型其他层无需知道头的细节"],
    code: `import torch\nx = torch.randn(2, 5, 16)\nh = 4\nsplit = x.view(2, 5, h, 4).transpose(1, 2)\nmerged = split.transpose(1, 2).contiguous().view(2, 5, 16)\nprint(split.shape, merged.shape, torch.equal(x, merged))`,
    output: "拆头形状 [2,4,5,4]，合回 [2,5,16]，在没有中间计算时数值完全恢复。",
    pit: "transpose 后内存通常不连续；用 view 合并前要 contiguous()，或谨慎使用 reshape。",
    quiz: "为什么 Attention 计算时把 h 放在长度 L 前面？", answer: "这样张量是 [B,h,L,dₖ]，矩阵乘法可同时对 batch 和各头并行。",
    chain: ["线性投影", "view+transpose", "各头 Attention", "concat+输出投影"]
  },
  {
    p: 123, slug: "multi-head-attention-code", title: "MultiHeadedAttention 代码：四个线性层和形状重排",
    plain: "前三个线性层分别产生 Q、K、V，第四个在线性拼接后融合各头。mask 增加 head 维后可广播给所有头。",
    points: ["使用深拷贝保证四个 Linear 参数独立", "保存 self.attn 方便调试和可视化", "常见 mask 扩为 [B,1,Lq,Lk]"],
    code: `import torch\nfrom transformer_from_scratch.model import MultiHeadedAttention, subsequent_mask\nlayer = MultiHeadedAttention(h=4, d_model=16, dropout=0.0)\nx = torch.randn(2, 5, 16)\ny = layer(x, x, x, subsequent_mask(5))\nprint(y.shape, layer.attn.shape)`,
    output: "输出 [2,5,16]；注意力权重为 [2,4,5,5]，包含 2 个 batch、4 个头。",
    pit: "把同一个 nn.Linear 对象重复放进列表会共享权重；需要 deepcopy，而不是只复制引用。",
    quiz: "为什么代码需要 4 个 d_model→d_model 的线性层？", answer: "三个分别投影 Q/K/V，最后一个融合拼接后的多头输出。",
    chain: ["3 个投影", "拆头", "Attention", "合头", "第 4 个投影"]
  },
  {
    p: 124, slug: "multi-head-attention-test", title: "多头注意力测试：不能只检查“能运行”",
    plain: "好的测试要同时检查输出形状、每头权重形状、概率和，以及 mask 后未来位置是否真的为 0。",
    points: ["使用带 0 和 1 的非平凡 mask", "检查 attention.sum(-1)≈1", "检查严格上三角权重全为 0"],
    code: `import torch\nfrom transformer_from_scratch.model import MultiHeadedAttention, subsequent_mask\nlayer = MultiHeadedAttention(4, 16, dropout=0.0)\ny = layer(*(torch.randn(2, 5, 16),) * 3, mask=subsequent_mask(5))\nprint(y.shape)\nprint(torch.allclose(layer.attn.sum(-1), torch.ones(2, 4, 5), atol=1e-6))`,
    output: "输出形状正确，第二行应为 True，说明每个 Query、每个头的权重已归一化。",
    pit: "全 1 mask 只能证明广播没有报错，无法证明屏蔽逻辑正确；全 0 mask 又会制造没有合法位置的异常情形。",
    quiz: "为什么测试时把 dropout 设为 0？", answer: "Dropout 会随机改变注意力权重和，使确定性断言复杂；组件性质测试应先关闭随机性。",
    chain: ["构造非平凡 mask", "执行多头", "查形状", "查概率与屏蔽"]
  },
  {
    p: 125, slug: "positionwise-feed-forward", title: "Position-wise FFN：每个位置独立加工特征",
    plain: "注意力负责在词之间交换信息，FFN 负责在每个词内部对特征做非线性变换。所有位置共享同一组两层网络参数。",
    points: ["形状 D→d_ff→D，通常 d_ff 大于 D", "对每个位置独立，因此不混合 L 维", "中间使用 ReLU/GELU 和 Dropout 提供非线性与正则化"],
    code: `import torch\nfrom transformer_from_scratch.model import PositionwiseFeedForward\nlayer = PositionwiseFeedForward(d_model=16, d_ff=64, dropout=0.0)\nx = torch.randn(2, 5, 16)\nprint(layer(x).shape)`,
    output: "输出仍是 [2,5,16]；中间曾扩展到 64 维，但第二个线性层又投回 16 维。",
    pit: "Position-wise 不代表每个位置有一套独立参数；恰恰是所有位置复用同一 FFN。",
    quiz: "FFN 会把第 2 个词的信息传到第 5 个词吗？", answer: "不会。跨位置交流由注意力完成，FFN 只处理各位置自身的特征向量。",
    chain: ["每个 token 的 D 维", "Linear 扩维", "激活+Dropout", "Linear 还原 D"]
  },
  {
    p: 126, slug: "layer-normalization-code", title: "LayerNorm 代码：在每个 token 内标准化特征",
    plain: "对每个 token 的 D 个特征计算均值和方差，标准化后再用可学习 scale 与 bias 调整。它不依赖同批次的其他样本。",
    points: ["归一化轴是最后一维 D", "unbiased=False 才与框架 LayerNorm 的总体方差定义一致", "scale 初始为 1，bias 初始为 0"],
    code: `import torch\nfrom transformer_from_scratch.model import LayerNorm\nx = torch.randn(2, 4, 8)\ny = LayerNorm(8)(x)\nprint(y.shape)\nprint(y.mean(-1).round(decimals=5))`,
    output: "形状保持 [2,4,8]；每个 token 沿最后一维的均值约为 0。",
    pit: "课程手写方差若使用默认无偏估计，会与 nn.LayerNorm 有小差异；本项目明确使用 unbiased=False。",
    quiz: "输入 [B,L,D] 时，LayerNorm 会把不同句子的同一特征混在一起统计吗？", answer: "不会。它对每个 [D] 向量独立统计。",
    chain: ["token 特征", "求 mean/var", "标准化", "scale+bias"]
  },
  {
    p: 127, slug: "layer-normalization-test", title: "LayerNorm 测试：均值约 0、方差约 1",
    plain: "测试要沿正确轴计算统计量，并允许浮点误差。还要检查输出形状和参数是否可学习。",
    points: ["使用 atol 容忍数值误差", "方差断言应与实现的 eps 和定义一致", "极端常量输入不会除零，因为有 eps"],
    code: `import torch\nfrom transformer_from_scratch.model import LayerNorm\nx = torch.randn(3, 5, 16)\ny = LayerNorm(16)(x)\nprint(torch.allclose(y.mean(-1), torch.zeros(3, 5), atol=1e-5))\nprint(torch.allclose(y.var(-1, unbiased=False), torch.ones(3, 5), atol=2e-5))`,
    output: "通常输出两个 True，分别验证最后一维均值和方差。",
    pit: "直接用 == 比较浮点数很脆弱；应使用 torch.allclose 并给合理容差。",
    quiz: "为什么常量向量标准化后不会产生 NaN？", answer: "分母加入 eps；方差为 0 时仍有正数可除，结果接近全 0 再经过 scale/bias。",
    chain: ["随机输入", "LayerNorm", "按 D 求统计", "allclose 断言"]
  },
  {
    p: 128, slug: "batchnorm-vs-layernorm", title: "BatchNorm 与 LayerNorm：统计方向不同",
    plain: "BatchNorm 通常跨 batch 统计同一特征，LayerNorm 在一个样本/词元内部跨特征统计。序列长度和批量变化时，LayerNorm 更稳定。",
    points: ["BatchNorm 依赖批次统计及训练/推理状态", "LayerNorm 对每个 token 独立，不依赖 batch 大小", "Transformer 通常使用 LayerNorm"],
    code: `import torch\nx = torch.randn(2, 3, 4)  # [B,L,D]\nprint("LayerNorm统计后形状：", x.mean(dim=-1).shape)\nprint("若跨batch统计形状：", x.mean(dim=0).shape)`,
    output: "LayerNorm 风格统计得到 [B,L]；跨 batch 统计得到 [L,D]，两者混合数据的方向完全不同。",
    pit: "BN 和 LN 不是谁绝对更高级；卷积视觉任务常用 BN，选择取决于结构、数据和批次特性。",
    quiz: "batch size=1 时，哪一种通常更自然适合 Transformer？", answer: "LayerNorm，因为它不需要依赖其他样本估计统计量。",
    chain: ["输入 [B,L,D]", "选择统计轴", "BN 跨样本", "LN 跨特征"]
  },
  {
    p: 129, slug: "sublayer-connection-code", title: "SublayerConnection：残差、Dropout 与归一化",
    plain: "残差连接保留原始 x，再加上子层学到的改变量。Pre-LN 先规范化再进入子层，公式是 x + Dropout(Sublayer(LN(x)))。",
    points: ["本项目采用 Pre-LN，深层训练通常更稳定", "残差相加要求子层输出与 x 形状一致", "课程中某次现场书写的顺序需与最终架构约定区分"],
    code: `import torch\nfrom transformer_from_scratch.model import SublayerConnection\nlayer = SublayerConnection(size=8, dropout=0.0)\nx = torch.randn(2, 3, 8)\ny = layer(x, lambda z: z * 0.5)\nprint(y.shape)`,
    output: "输出仍为 [2,3,8]。lambda 模拟一个子层，SublayerConnection 负责通用包装。",
    pit: "Pre-LN 与 Post-LN 都存在，不能混着写。读代码时以实际公式为准，而不是只看“Add & Norm”标签。",
    quiz: "残差连接为什么要求输入输出形状相同？", answer: "因为要逐元素执行 x + sublayer_output；形状不一致无法直接相加。",
    chain: ["原始 x", "LayerNorm", "子层+Dropout", "与 x 残差相加"]
  },
  {
    p: 130, slug: "sublayer-connection-test", title: "子层连接测试：用 lambda 注入不同组件",
    plain: "Attention 需要 mask 等额外参数，而 FFN 只接收 x。用 lambda 把它们统一包装成“接收一个 x 的可调用对象”。",
    points: ["SublayerConnection 不关心内部是 Attention 还是 FFN", "lambda 闭包可以捕获 mask、memory 等额外参数", "测试要检查残差形状和调用顺序"],
    code: `import torch\nfrom transformer_from_scratch.model import SublayerConnection\nx = torch.ones(1, 2, 4)\nlayer = SublayerConnection(4, dropout=0.0)\ny = layer(x, lambda normalized: torch.zeros_like(normalized))\nprint(torch.equal(x, y))`,
    output: "输出 True：当子层返回全零时，残差结果应精确等于原始 x。",
    pit: "lambda 中若误用外层旧 x，而不是传入的 normalized x，会绕过 LayerNorm，让 Pre-LN 失效。",
    quiz: "当子层输出全 0 且 dropout=0 时，残差输出是什么？", answer: "原始 x，因为 x+0=x。",
    chain: ["准备 x", "注入 callable", "子层返回改变量", "验证残差"]
  },
  {
    p: 131, slug: "encoder-layer-code", title: "EncoderLayer 代码：两个子层串起来",
    plain: "一个 EncoderLayer 先做自注意力，再做位置前馈网络；两者分别由独立 SublayerConnection 包住。",
    points: ["自注意力调用 self_attn(x,x,x,mask)", "FFN 接收上一个子层输出", "两个 SublayerConnection 必须是独立深拷贝"],
    code: `import torch\nfrom transformer_from_scratch.model import EncoderLayer, MultiHeadedAttention, PositionwiseFeedForward\nlayer = EncoderLayer(16, MultiHeadedAttention(4,16,0.0), PositionwiseFeedForward(16,32,0.0), 0.0)\nx = torch.randn(2,5,16)\nprint(layer(x, None).shape)`,
    output: "输出仍为 [2,5,16]，所以多个 EncoderLayer 才能首尾相接。",
    pit: "Encoder 自注意力的 Q/K/V 都是当前 x；若误传三份独立随机张量，就不是 Self-Attention。",
    quiz: "EncoderLayer 为什么只有两个子层？", answer: "它包含 Self-Attention 和 FFN；交叉注意力只存在于 Encoder-Decoder 架构的 Decoder 中。",
    chain: ["x", "自注意力残差子层", "FFN 残差子层", "x′"]
  },
  {
    p: 132, slug: "encoder-layer-test", title: "EncoderLayer 测试：看模块树，也看 mask",
    plain: "打印模块树能确认 self_attn、feed_forward 和两个残差包装都被注册；前向测试再确认 mask 能正确传入注意力。",
    points: ["nn.ModuleList 才能正确注册重复子层参数", "测试输出形状与输入一致", "检查不同 mask 是否会改变注意力权重"],
    code: `from transformer_from_scratch.model import EncoderLayer, MultiHeadedAttention, PositionwiseFeedForward\nlayer = EncoderLayer(8, MultiHeadedAttention(2,8,0.0), PositionwiseFeedForward(8,16,0.0), 0.0)\nprint(layer)\nprint("参数量：", sum(p.numel() for p in layer.parameters()))`,
    output: "模型树会列出注意力、FFN 和两个 SublayerConnection；参数量应大于 0。",
    pit: "普通 Python list 中的层不会自动注册到 Module 参数树；重复网络层应使用 ModuleList。",
    quiz: "为什么只看输出形状不能发现所有错误？", answer: "即使 mask 未生效或两个层错误共享参数，形状仍可能完全正确。",
    chain: ["检查模块树", "检查参数注册", "执行前向", "检查 mask 效果"]
  },
  {
    p: 133, slug: "encoder-code-and-test", title: "Encoder 堆叠：N 层独立参数逐层提炼",
    plain: "Encoder 深拷贝同一个层结构 N 次，按顺序传递 x，最后再做一次 LayerNorm，输出供 Decoder 使用的 memory。",
    points: ["层结构相同但参数对象独立", "每层都使用同一份源 mask", "最终 memory 形状仍为 [B,Ls,D]"],
    code: `import torch\nfrom transformer_from_scratch.model import Encoder, EncoderLayer, MultiHeadedAttention, PositionwiseFeedForward\nbase = EncoderLayer(8, MultiHeadedAttention(2,8,0.0), PositionwiseFeedForward(8,16,0.0), 0.0)\nencoder = Encoder(base, n=3)\nprint(encoder(torch.randn(2,6,8), None).shape)`,
    output: "输出 [2,6,8]；层数增加改变的是计算深度，不改变对外形状。",
    pit: "写 layers=[base]*N 会让每一层引用同一个对象、共享权重；需要 deepcopy。",
    quiz: "三个 EncoderLayer 是否读取三份不同源 mask？", answer: "通常读取同一份源 mask，因为 PAD 位置在所有层都相同。",
    chain: ["嵌入表示", "Layer 1", "…", "Layer N", "最终 Norm→memory"]
  },
  {
    p: 134, slug: "decoder-layer-code", title: "DecoderLayer 代码：三个子层与两种长度",
    plain: "DecoderLayer 按顺序执行带因果 mask 的目标自注意力、读取 memory 的交叉注意力，以及位置前馈网络。",
    points: ["自注意力：Q=K=V=x，使用 tgt_mask", "交叉注意力：Q=x，K=V=memory，使用 src_mask", "三个子层都用残差包装，输出长度始终是 Lt"],
    code: `import torch\nfrom transformer_from_scratch.model import DecoderLayer, MultiHeadedAttention, PositionwiseFeedForward\nattn = MultiHeadedAttention(2,8,0.0)\nlayer = DecoderLayer(8, attn, MultiHeadedAttention(2,8,0.0), PositionwiseFeedForward(8,16,0.0), 0.0)\ny = layer(torch.randn(2,4,8), torch.randn(2,6,8), None, None)\nprint(y.shape)`,
    output: "目标长度 4、源长度 6，输出仍为 [2,4,8]，因为 Query 来自目标侧。",
    pit: "交叉注意力若把 Q/K/V 都传成 x，代码仍能跑，却完全读不到 Encoder memory。",
    quiz: "Cross-Attention 输出为什么是 Lt 长而不是 Ls 长？", answer: "输出每个位置对应一个 Query；Q 来自目标侧，所以长度是 Lt。",
    chain: ["目标 x", "因果自注意力", "交叉注意力读 memory", "FFN", "目标 x′"]
  },
  {
    p: 135, slug: "decoder-layer-test", title: "DecoderLayer 测试：故意让源长和目标长不同",
    plain: "若测试中 Ls=Lt，许多错误维度会被巧合掩盖。让源长 6、目标长 4，更容易发现 mask 或 Q/K/V 传错。",
    points: ["x [B,Lt,D]，memory [B,Ls,D]", "tgt_mask 最后两维 Lt×Lt", "src_mask 的 Key 长度必须是 Ls，并能广播到 [B,h,Lt,Ls]"],
    code: `import torch\nfrom transformer_from_scratch.model import subsequent_mask\nB, Ls, Lt = 2, 6, 4\nsrc_mask = torch.ones(B, 1, 1, Ls, dtype=torch.bool)\ntgt_mask = subsequent_mask(Lt)\nprint(src_mask.shape, tgt_mask.shape)`,
    output: "源 mask 为 [2,1,1,6]，目标 mask 为 [1,4,4]；它们服务于不同注意力。",
    pit: "把 tgt_mask 错传给 cross-attention，在 Ls=Lt 时可能不报错；长度不同后会立即暴露。",
    quiz: "Decoder 自注意力分数最后两维是什么？交叉注意力呢？", answer: "自注意力是 [Lt,Lt]；交叉注意力是 [Lt,Ls]。",
    chain: ["设置 Ls≠Lt", "构造两种 mask", "运行三子层", "检查输出/权重形状"]
  },
  {
    p: 136, slug: "decoder-code-and-test", title: "Decoder 堆叠：每层共享 memory，目标状态逐层变化",
    plain: "Decoder 把基础 DecoderLayer 深拷贝 N 份。每一层都接收上一层目标状态，但读取同一份 Encoder memory。",
    points: ["memory 在 Decoder 层间不被改写", "src_mask 与 tgt_mask 每层重复使用", "最后 LayerNorm 后输出 [B,Lt,D]"],
    code: `import torch\nfrom transformer_from_scratch.model import Decoder, DecoderLayer, MultiHeadedAttention, PositionwiseFeedForward\nbase = DecoderLayer(8, MultiHeadedAttention(2,8,0.0), MultiHeadedAttention(2,8,0.0), PositionwiseFeedForward(8,16,0.0), 0.0)\ndecoder = Decoder(base, 2)\nprint(decoder(torch.randn(2,4,8), torch.randn(2,6,8), None, None).shape)`,
    output: "输出 [2,4,8]。两层都能访问 [2,6,8] 的同一份 memory。",
    pit: "共享 memory 不等于各层交叉注意力共享参数；层是深拷贝的，参数应独立。",
    quiz: "DecoderLayer 之间变化的主要张量是什么？", answer: "目标侧隐藏状态 x；Encoder memory 通常保持不变供每层读取。",
    chain: ["目标嵌入", "DecoderLayer 1", "共享 memory", "Layer N", "最终 Norm"]
  },
  {
    p: 137, slug: "generator-code", title: "Generator 代码：从隐藏维映射到目标词表",
    plain: "Decoder 每个位置只输出 D 维隐藏状态。Generator 用 Linear(D,Vt) 为目标词表中每个词生成一个分数，再转成对数概率。",
    points: ["输入 [B,Lt,D]，输出 [B,Lt,Vt]", "Vt 是目标词表大小，不一定等于源词表", "log_softmax 数值稳定，常与 NLLLoss 配合"],
    code: `import torch\nfrom transformer_from_scratch.model import Generator\nlayer = Generator(d_model=16, vocab_size=30)\nx = torch.randn(2,5,16)\ny = layer(x)\nprint(y.shape)`,
    output: "输出 [2,5,30]：每个样本、每个目标位置都有 30 个词的对数概率。",
    pit: "Generator 输出不是词 ID。推理时还要根据概率选择 token，训练时则与正确目标计算损失。",
    quiz: "为什么 Generator 的词表大小用 tgt_vocab？", answer: "它预测的是目标语言/目标序列中的词，两侧词表可能完全不同。",
    chain: ["Decoder 隐藏状态", "Linear D→Vt", "log_softmax", "词表对数概率"]
  },
  {
    p: 138, slug: "generator-test", title: "Generator 测试：指数还原后概率和应为 1",
    plain: "除了检查最后一维等于词表大小，还要把 log_probs 取 exp，验证每个位置沿词表维求和约为 1。",
    points: ["log_probs.exp() 还原普通概率", "softmax 归一化轴必须是最后的词表维", "使用 allclose 处理浮点误差"],
    code: `import torch\nfrom transformer_from_scratch.model import Generator\ny = Generator(8, 11)(torch.randn(2,4,8))\nprint(y.shape)\nprint(torch.allclose(y.exp().sum(-1), torch.ones(2,4), atol=1e-6))`,
    output: "形状 [2,4,11]，概率和检查通常为 True。",
    pit: "若误在长度维做 softmax，每个位置的词表概率不会归一化，但输出形状仍完全正确。",
    quiz: "为什么 log_probs 本身求和不等于 1？", answer: "它们是概率的对数；需要先 exp 还原概率再求和。",
    chain: ["随机隐藏状态", "Generator", "exp", "沿 V 求和≈1"]
  },
  {
    p: 139, slug: "full-transformer-upper", title: "完整模型上：forward 如何组织编码和解码",
    plain: "完整 forward 先 encode(src) 得到 memory，再 decode(tgt,memory)，最后交给 Generator 得到词表对数概率。",
    points: ["src_embed 与 tgt_embed 独立处理两侧词 ID", "Encoder 只运行一次，memory 供 Decoder 使用", "项目 forward 直接返回 Generator 结果"],
    code: `flow = ["src", "src_embed", "encoder -> memory",\n        "tgt + memory -> decoder", "generator -> log_probs"]\nprint("\\n".join(f"{i+1}. {x}" for i, x in enumerate(flow)))`,
    output: "打印整模五步。调试时应在每一步打印 shape，而不是只看最终异常。",
    pit: "encode 是封装方法，encoder 是模块对象；命名混淆会导致错误递归或参数传递。",
    quiz: "为什么训练一个 batch 时 Encoder memory 不需要为每个目标位置重新计算？", answer: "同一源句的编码结果不变，可以由整段目标序列共同查询。",
    chain: ["src_embed", "Encoder", "memory", "tgt_embed+Decoder", "Generator"]
  },
  {
    p: 140, slug: "full-transformer-lower", title: "完整模型下：encode/decode 接口要分清",
    plain: "encode 只接收源 ID 和源 mask；decode 接收目标 ID、memory、源 mask 和目标 mask。清晰接口能避免调用时参数顺序混乱。",
    points: ["encode 输出 [B,Ls,D] memory", "decode 输出 [B,Lt,D] 隐藏状态", "Generator 再把隐藏状态变成 [B,Lt,Vt]"],
    code: `import torch\nfrom transformer_from_scratch.model import make_model, subsequent_mask\nmodel = make_model(20, 25, n=1, d_model=8, d_ff=16, h=2, dropout=0.0)\nsrc, tgt = torch.randint(0,20,(2,6)), torch.randint(0,25,(2,4))\nmemory = model.encode(src, None)\nhidden = model.decode(tgt, memory, None, subsequent_mask(4))\nprint(memory.shape, hidden.shape)`,
    output: "memory 为 [2,6,8]，hidden 为 [2,4,8]，清楚展示源长和目标长各自保留在哪里。",
    pit: "源 ID 必须小于 src_vocab，目标 ID 必须小于 tgt_vocab，不能用同一随机上界偷懒。",
    quiz: "decode 为什么同时需要 src_mask 和 tgt_mask？", answer: "src_mask 用于交叉注意力屏蔽源 PAD；tgt_mask 用于目标自注意力屏蔽 PAD 和未来。",
    chain: ["encode(src)", "memory", "decode(tgt,memory)", "hidden", "generator"]
  },
  {
    p: 141, slug: "transformer-test-upper", title: "完整模型组装：make_model 把所有组件接起来",
    plain: "make_model 创建 Attention、FFN、Encoder、Decoder、两侧输入组件和 Generator，再对矩阵参数做 Xavier 初始化。",
    points: ["源/目标 Embedding 不应无条件共享，因为词表可能不同", "重复层和注意力模块用 deepcopy 保证参数独立", "Xavier 初始化帮助信号在深层网络中保持合理尺度"],
    code: `from transformer_from_scratch.model import make_model\nmodel = make_model(src_vocab=31, tgt_vocab=37, n=2,\n                   d_model=16, d_ff=32, h=4, dropout=0.0)\nprint(type(model).__name__)\nprint(sum(p.numel() for p in model.parameters()))`,
    output: "输出 EncoderDecoder 和总参数量，说明所有子模块已注册进模型树。",
    pit: "参数量大于 0 不代表连接正确；仍需端到端前向和组件性质测试。",
    quiz: "为什么 src_vocab=31、tgt_vocab=37 时不能把源 Embedding 深拷贝给目标侧？", answer: "两张表行数不同，目标 ID 最高可到 36，31 行的源表会越界。",
    chain: ["创建基础组件", "深拷贝层", "连接整模", "Xavier 初始化"]
  },
  {
    p: 142, slug: "transformer-components-review", title: "组件总复盘：从模型树反向读出数据流",
    plain: "看到模型树时，不要被缩进吓住。沿 src_embed→encoder→memory→tgt_embed→decoder→generator 追踪，就能把嵌套结构还原成一条主线。",
    points: ["模块树回答“有哪些参数化组件”", "shape 主线回答“数据怎么走”", "Q/K/V 表回答“三种注意力各读哪里”"],
    code: `qkv = [\n    ("Encoder self", "src", "src", "src"),\n    ("Decoder self", "tgt", "tgt", "tgt"),\n    ("Cross", "tgt", "memory", "memory"),\n]\nfor name, q, k, v in qkv:\n    print(f"{name:12} Q={q:6} K={k:6} V={v}")`,
    output: "输出三组 Q/K/V 来源。这张表是排查 Transformer 连接错误的核心速查表。",
    pit: "只背类名不追踪张量，遇到变长序列或 mask 广播就容易断层。",
    quiz: "完整模型中哪三个地方会改变最后一维大小？", answer: "Embedding 把 ID 变为 D，FFN 中间暂时 D→d_ff→D，Generator 最终 D→Vt。",
    chain: ["模型树", "shape 主线", "三组 Q/K/V", "逐层定位"]
  },
  {
    p: 143, slug: "transformer-test-lower", title: "完整模型端到端测试：最终输出不等于训练完成",
    plain: "端到端测试构造合法源/目标 ID 和两种 mask，执行整模前向，检查输出 [B,Lt,Vt] 与概率归一化。",
    points: ["测试同时覆盖嵌入范围、mask 广播和编码解码连接", "model.eval()+torch.no_grad() 让测试确定且省显存", "前向通过只证明结构可运行，训练还需数据、目标右移、损失和解码"],
    code: `import torch\nfrom transformer_from_scratch.model import make_model, subsequent_mask\nmodel = make_model(31,37,n=2,d_model=16,d_ff=32,h=4,dropout=0.0).eval()\nsrc, tgt = torch.randint(0,31,(2,6)), torch.randint(0,37,(2,5))\nwith torch.no_grad():\n    y = model(src, tgt, torch.ones(2,1,6,dtype=torch.bool), subsequent_mask(5))\nprint(y.shape, torch.allclose(y.exp().sum(-1), torch.ones(2,5), atol=1e-5))`,
    output: "输出 [2,5,37] 和 True，说明每个目标位置都有合法的目标词表概率分布。",
    pit: "这不是完整翻译系统。还缺 padding mask 合并、目标右移、损失、优化器、学习率调度与自回归解码。",
    quiz: "为什么模型输入 tgt 不能包含正在预测位置之后的真实词信息？", answer: "否则训练时会偷看答案；必须右移目标，并用因果 mask 限制可见范围。",
    chain: ["合法 ID+mask", "整模前向", "[B,Lt,Vt]", "概率检查", "再进入训练"]
  }
];

const articles = [
  `### 0:00–1:27　老师先交代：这一大章究竟要学什么

老师开头说，接下来整章叫“Transformer 架构和实现”。这里的“架构”，你可以先理解为一张建筑设计图：模型由哪些房间组成、数据从哪扇门进去、经过哪些房间、最后从哪里出来。“实现”则是把设计图里的每一个方框写成真正可以运行的 PyTorch 代码。

老师先把完整 Transformer 分成四块：

1. **输入部分**：把文字变成模型能够计算的数字向量。
2. **编码器 Encoder**：阅读并理解输入句子，得到包含上下文的信息。
3. **解码器 Decoder**：一边参考编码器的理解结果，一边逐步生成目标句子。
4. **输出部分**：把解码器内部的数字表示变成“词表中每个词的概率”。

老师还提前提醒，后面编码器代码讲得会比较慢、比较细，解码器反而会快一些。不是因为解码器不重要，而是两者复用了很多相同零件：注意力、前馈网络、残差连接和归一化等。也就是说，**编码器部分不是可以跳过的前置知识；它是后面读懂解码器的地基。**

### 1:27–3:10　Transformer 出现以前，大家主要用什么

老师接着回顾 RNN、LSTM 和 GRU。先不要急着记三个缩写的结构，只要抓住它们共同的工作方式：处理句子时通常按顺序一步一步往后走。

例如句子是：

> 我 / 今天 / 学习 / Transformer

循环网络先处理“我”，得到一个隐藏状态；再把这个隐藏状态和“今天”一起处理；然后继续传给“学习”，最后才轮到“Transformer”。这种方式像接力跑，后一个人必须等前一个人把接力棒传过来。

老师由此指出几个问题：

- **难以在一个序列内部充分并行。** 第 4 步依赖第 3 步，第 3 步又依赖第 2 步，不能把所有位置同时算完。
- **长文本中的早期信息可能逐渐变弱。** 第一个词的信息要经过很多次传递才能到达最后一个词。
- **训练可能遇到梯度消失或梯度爆炸。** 你现在不用推导梯度，只需先理解：模型根据错误反向调整参数时，信号可能变得过小，学不动；也可能变得过大，训练不稳定。

LSTM 和 GRU 对长期信息问题做了改进，但它们仍保留“按时间步递归”的基本路线，所以并行限制并没有彻底消失。

### 3:10–4:58　注意力为什么先在机器翻译中受到重视

老师回忆前面讲过的机器翻译：早期 Encoder–Decoder 往往试图把整个源句压进一个固定向量，再让 Decoder 依靠这个向量生成译文。句子变长后，一个固定向量很难完整保存所有细节。

注意力机制提供了另一种方法。Decoder 每生成一个词，都可以重新查看源句中的所有位置，并给当前最相关的位置更高权重。例如翻译法语中的一个名词时，模型可以重点读取英语源句中对应的名词，而不是要求一个固定向量始终记住整句的每个细节。

因此老师强调，注意力早期的重要应用场景就是机器翻译。Transformer 再向前走了一步：它不只是给旧式循环模型外挂一个注意力模块，而是让注意力成为整个架构的核心计算。

### 4:58–6:28　2017 年论文、BERT，以及 GPT 名字中的 T

老师把时间线串了起来：

- **2017 年**，Google 团队发表 *Attention Is All You Need*，提出 Transformer。
- **2018 年**，BERT 使用 Transformer 编码器路线，在多项自然语言理解任务上取得很强效果，让整个架构获得更广泛关注。
- 此后许多模型即使训练任务和细节不同，核心骨架仍继续使用 Transformer。

老师还用 GPT 的名字帮助记忆：GPT 中的 **T 就是 Transformer**。但这里必须分清层级：

- Transformer 是一种网络架构，好比建筑结构。
- BERT、GPT 是采用这种结构并配合不同训练目标得到的模型家族。
- ChatGPT 是在 GPT 路线上继续训练、对齐并做成的对话系统，不等于 Transformer 本身。

课堂上老师打开论文页面，是想让大家知道后续课件中的一些 BERT 架构图、预训练和微调图并不是凭空画出来的，后面的预训练模型专题还会重新解释。

### 6:28–9:26　Transformer 的第一个核心优势：并行

老师把循环网络的“串行”与 Transformer 的“并行”放在一起比较。

循环网络像：

> 第 1 个词算完 → 第 2 个词才能算 → 第 3 个词才能算

Transformer 在训练时可以把整段序列同时组成矩阵，计算各位置之间的相关程度。这样 GPU 尤其擅长的大规模矩阵乘法就能充分发挥作用，也更容易使用多块 GPU 做分布式训练。

注意，“并行”主要说的是**训练时，同一层中的多个位置可以一起计算**。真正生成文本时，Decoder 仍然常常要一个 token 接一个 token 地生成，因为下一个 token 依赖已经生成的前缀。不能把“训练可并行”误解成“任何情况下都一次生成完整答案”。

### 9:26–11:26　第二个核心优势：更直接地处理长距离关系

为了让大家理解注意力，老师用了人的记忆作类比。假如让你回忆从初中到现在发生的每一天，你不可能把所有细节都同样清楚地说出来；但对你很重要的几件事，可能多年后仍然记得。

老师想表达的是：

> 与当前任务关系小的信息，权重可以低一些；关系大的信息，权重可以高一些。

在循环网络中，句首和句尾的信息要经过许多隐藏状态传递才能相遇。自注意力则允许句尾位置直接和句首位置计算关系，信息交互路径更短。

这里也要校准老师的生活类比：模型不是像人一样“真正理解重要性”，也不是把低权重词直接删掉；它是通过训练学出一组连续数值权重，再用这些权重对信息做加权组合。

课件还展示了随着句子变长，不同模型效果变化的实验曲线。老师希望大家观察的是趋势：没有注意力的循环模型在长序列上下降更明显，加入注意力后相对稳定。图中的 20、30、40 等长度来自特定实验，**不是所有任务通用的硬分界线**。

### 11:26–12:53　SOTA 和“市场占用”到底是什么意思

老师展示机器翻译排行榜，指出当时排名靠前的许多系统都使用 Transformer。他想说明学习它不是只为了考试，而是因为后续工业系统和招聘要求中经常出现。

课上提到的 **SOTA** 是 *State of the Art* 的缩写。它不是某个固定模型的名字，而表示：在某个时间、某个数据集、某项任务上达到的先进水平。

所以“SOTA 模型”必须带着条件理解：

- 换一个任务，最佳模型可能不同。
- 换一个数据集，排名可能变化。
- 随着新研究出现，今天的 SOTA 也可能被超过。

### 12:53–15:23　老师最后复习的两组概念

第一组是 Transformer 的来源与优势：

- 来自 Google 团队在 2017 年发表的论文。
- 关键创新是以 Self-Attention 为核心，而不是继续依赖循环结构。
- 相比传统循环网络，更容易并行，也更容易让远距离位置直接交互。

第二组是 NLP 任务的粗略划分：

- **NLU，Natural Language Understanding，自然语言理解。** 重点是从现有文本中判断、分类或提取信息，例如分词、词性标注、命名实体识别、情感分类。
- **NLG，Natural Language Generation，自然语言生成。** 重点是产生新的文本，例如机器翻译、摘要、对话生成。

老师为了入门记忆，把 Encoder 与“理解”、Decoder 与“生成”对应起来。这个方向大体有帮助：BERT 更偏 Encoder 路线，GPT 更偏 Decoder 路线。但它不是绝对禁令，真实系统可能同时包含理解和生成，一个架构也可能通过不同训练方式服务多种任务。

读完整节，你目前只需要牢牢记住一条主线：

> 循环网络按顺序传递信息，难并行且远距离路径长；Transformer 用注意力让位置直接交互，因此成为后来许多 NLP 模型的基础架构。`,

  `### 0:00–2:10　先明确这一节的目标

老师开头说，这一节只要求先看懂 Transformer 的“总架构”，还不要求马上知道每个公式怎么计算。学习目标有两个：第一，能说出模型由哪四部分组成；第二，能继续说出每一部分里面有哪些层、这些层大致负责什么。

老师把任务背景放在机器翻译上：输入是一句已知英语，目标是一句已知法语，模型学习从一种语言到另一种语言的映射。他还顺带提到预训练模型和迁移学习：别人已经在大量数据上训练好模型，我们可以加载它，再针对自己的任务继续使用。这里先记概念，后面有专门专题。

### 2:10–4:30　完整模型不是两个框，而是四个部分

老师指着架构图从下往上辨认：

1. **输入部分**：接收源文本和训练时的目标文本。
2. **编码器 Encoder**：左侧大模块，处理源句。
3. **解码器 Decoder**：右侧大模块，处理目标前缀并读取编码结果。
4. **输出部分**：最上方的 Linear 与 Softmax，把隐藏表示变成预测。

只回答“四部分是输入、编码、解码、输出”还不够。老师用“别人问你学了什么，你只回答小学、初中、高中、大学”作类比：分类名称对了，却没说明具体内容。面试或自测时必须继续展开内部层级。

图中每个 Attention 方框都离不开 Q、K、V。现在不需要计算，但要形成习惯：看到 Attention，就问“谁提供 Query，谁提供 Key，谁提供 Value”。

### 4:30–8:20　输入部分：词义和位置缺一不可

Embedding 把 token ID 查成稠密向量，让模型能用数字表示词义。随后加入 Positional Encoding。

老师用“我爱你”和“你爱我”说明位置的重要性。两句话拥有相同的词，顺序不同，角色和语义就不同。单纯注意力不天然包含先后；位置编码相当于给每个词加坐标。

训练翻译模型时有两路输入：左边是已知源语言句子，右边是右移后的目标语言句子。最上方才是模型预测的目标分布。不要把“训练时输入的目标句”和“模型最终预测”混成一个东西。

### 8:20–13:40　四级层级：组件 → 子层 → 编码器层 → 编码器

这一段是老师反复强调的重点：

- **组件**：Attention、Feed Forward、LayerNorm、残差连接等零件。
- **子层 Sublayer**：把某个组件放进带残差和归一化的通用外壳。
- **编码器层 EncoderLayer**：由自注意力子层和 FFN 子层组成。
- **编码器 Encoder**：由 N 个 EncoderLayer 堆叠组成，原论文典型配置 N=6。

所以“一个编码器有六个编码器”是错误说法，正确的是“一个 Encoder 由六个 EncoderLayer 组成”。N 也不是只能等于 6；增加层数可能提高容量，同时增加计算量。

### 13:40–18:30　Decoder 为什么有三个子层

Decoder 的层级结构相似，但每个 DecoderLayer 有三个子层：

1. 带因果 mask 的目标自注意力。
2. 读取 Encoder memory 的交叉注意力。
3. Position-wise FFN。

这就是 Decoder 后面讲得更快的原因：残差、归一化、FFN 和普通多头注意力已经实现，可以复用。图中从 Encoder 连到 Decoder 的线表示各 DecoderLayer 都能读取同一份 memory；DecoderLayer 之间还会逐层传递目标隐藏状态。

### 18:30–21:35　输出部分：先得到分数，再得到概率

Decoder 输出仍是 d_model 维隐藏向量。Linear 把它投影到目标词表大小 V。若词表有 10,000 个词，就为每个位置产生 10,000 个分数；随后 Softmax 把这些分数转成和为 1 的概率。

Linear 输出的是 logits，不是概率。真正生成时也不一定永远选最大概率词，还可能采样或使用 beam search；这一节先理解“隐藏状态 → 词表分布”。

### 21:35–23:19　老师最后检查什么

老师用选择题复习：输入负责词嵌入与位置编码；Encoder 把源信息加工后交给 Decoder；Decoder 自注意力会对目标序列位置分配权重。

最后要求手绘架构图，目的不是练美术，而是强迫自己重建层级和连线。若能不看原图画出四大部分、Encoder 的两个子层、Decoder 的三个子层和 Encoder→Decoder 连接，才算真正读懂总架构。`,

  `### 0:00–1:40　只看左半边，把 Encoder 单独拆出来

上一节先认了完整架构，这一节老师把图的左半边放大，要求沿着一条数据路线讲清楚 Encoder。源语言句子从底部进入，经过输入处理，再进入一层层 EncoderLayer，顶部得到供 Decoder 使用的 memory。

这里不要急着背类名。先抓住方向：**原始 token → 带词义和位置的向量 → 上下文表示 → 更深的上下文表示**。

### 1:40–4:20　Embedding 和位置编码为什么必须相加

Embedding 把离散词编号变成连续向量；它回答“这是什么词”。但仅有词向量不能可靠表达顺序。老师再次用词序改变句义的例子说明，“我打你”和“你打我”用到相同的三个字，施事与受事却完全交换了。

位置编码回答“这个词站在哪里”。二者形状都是 [B,L,D]，逐元素相加后仍保持统一形状，才能送给下一层。这里的“相加”不是把词义覆盖掉，而是让每个位置同时携带内容信号和坐标信号。

### 4:20–8:30　多头自注意力：让每个词从整句读取信息

EncoderLayer 的第一个核心计算是 Multi-Head Self-Attention。“Self”表示 Q、K、V 都来自当前这份源序列表示。每个词作为 Query，去和全句各位置的 Key 比较，再按权重汇总 Value。

老师用“从多个角度观察”来解释多头：同一句话不能只用一种关系分析。一个头可能关注主谓关系，一个头可能关注动宾关系，另一个头可能关注远距离修饰。类似几名士兵或几组观察员各自侦察，最后把报告合并。以“猫追老鼠”为例，模型不仅要看到三个孤立词，还要学会谁执行动作、动作指向谁。

多头不等于简单复制相同结果。每个头有自己的线性投影参数，因此能进入不同表示子空间。

### 8:30–11:10　Add 与 Norm：旧信息要保留，训练也要稳定

注意力输出后不是直接丢给下一模块，而要经过 Add & Norm：

1. Add 是残差连接，把子层输入与子层输出相加。即使新计算暂时学得不好，旧信息仍有直接通路；深层网络的梯度也更容易传播。
2. Norm 是 Layer Normalization，把每个位置的特征尺度整理得更稳定，减轻层层堆叠时数值漂移。

所以图上的 Add & Norm 不是一个神秘算法，而是“残差相加 + 归一化”的组合外壳。

### 11:10–13:40　FFN：交流之后，每个位置自己加工

注意力解决不同位置之间的信息交换，Position-wise Feed Forward Network 则对每个位置独立做非线性变换。可以理解为先开会交流，再让每个人拿着交流结果独立思考、整理。

FFN 通常先把 d_model 扩大到 d_ff，加入非线性，再投回 d_model。它不直接混合不同位置；所有位置共享同一套参数。FFN 后面再次使用残差连接与归一化，因此一个 EncoderLayer 共有两个子层。

### 13:40–16:20　N× 的真正含义

图中的 N× 表示把整个 EncoderLayer 堆叠 N 次，原论文典型设置为 6。第 1 层输出成为第 2 层输入，如此逐层提炼。老师用“领导分析问题要多想几层”作类比：第一层捕获直接关系，后续层可组合出更抽象的关系。

必须区分：Encoder 是完整编码器；EncoderLayer 是其中可重复的一层；Attention 和 FFN 又是层中的组件。说清这三级，后面写类时才不会把容器和零件混成一件事。

### 本节主线

源 token 先获得词义与位置，经过“自注意力交流 → 残差归一化 → FFN 加工 → 残差归一化”，再把这一整层重复 N 次。Encoder 顶部输出的不是翻译结果，而是携带整句上下文的 memory。`,

  `### 0:00–1:50　沿右半边追踪 Decoder

老师接着画完整架构的右半边。Decoder 的底部接收目标侧序列，先经过目标词 Embedding 与位置编码；中间由多层 DecoderLayer 处理；顶部再交给 Linear 和 Softmax 产生词表预测。

训练时目标句会右移一位作为输入。例如希望模型输出“我爱你”，输入可能从起始符开始，让模型依次预测“我”“爱”“你”。因此底部的目标输入和顶部的目标预测不是同一个概念。

### 1:50–5:20　第一个子层必须 Mask：不许偷看答案

DecoderLayer 的第一种注意力是 Masked Multi-Head Self-Attention。它仍是目标序列内部的自注意力，但每个位置只能读取自己和过去，不能读取未来。

老师用“预测第三个词”解释：训练数据中完整目标句已经装进张量，如果不加 mask，当前位置会直接看到第三个词甚至后面的词，相当于考试时翻答案。训练损失可能看起来很低，真正生成时却没有未来答案可看，模型就会失效。

因果 mask 通常是下三角允许矩阵。第 i 行只开放第 0 到第 i 列。这里要一直记住矩阵约定：行对应 Query 位置，列对应 Key 位置。

### 5:20–9:10　第二个子层是交叉注意力：去源句里找依据

Decoder 的第二种注意力是 Encoder–Decoder Attention，也叫 Cross-Attention。它连接左右两半：

- Query 来自 Decoder 当前目标状态；
- Key 和 Value 来自 Encoder 顶部输出的 memory。

可以把它想成目标侧带着问题去英文源句中检索。例如准备生成法语中的某个词时，Query 表达“我现在缺什么信息”，Key 帮助判断源句哪些位置相关，Value 提供实际读取的源语义。这样 Decoder 才能建立源词与目标词之间并不总是一一对应的对齐关系。

### 9:10–11:50　FFN、残差和归一化继续复用

交叉注意力后还有 Position-wise FFN；三个核心计算外面都配有残差连接和归一化。一个 DecoderLayer 因此包含三个子层：

1. masked target self-attention；
2. source–target cross-attention；
3. position-wise FFN。

完整 Decoder 再把 DecoderLayer 堆叠 N 次。这里大量结构都能复用 Encoder 阶段已经实现的组件，新的关键差异主要是因果 mask 与 Q/K/V 来源。

### 11:50–14:40　Linear 与 Softmax 把隐藏向量变成词

Decoder 顶部输出仍是 [B,Lt,d_model] 隐藏表示。Linear 将最后一维从 d_model 投影到目标词表大小 V，得到每个候选词的 logits；Softmax 再把 logits 变成概率分布。

若目标词表有 10,000 个 token，每个目标位置就会得到 10,000 个分数。最大概率项可作为简单预测，实际生成还可能使用采样或 beam search。

### 14:40–结束　两类注意力的形状不要混

目标自注意力的 Query 和 Key 都有 Lt 个位置，权重矩阵最后两维为 Lt×Lt；交叉注意力中 Query 长度是 Lt，Key 长度是源句 Ls，因此为 Lt×Ls。源句和目标句长度不同完全正常。

读 Decoder 图时，先问“这一块 Q、K、V 分别从哪来”，再问“是否需要 mask”。能回答这两个问题，就已经抓住右半边最关键的逻辑。`,

  `### 0:00–2:30　为什么不直接从完整模型代码开始

老师这一节先安排实现顺序。完整 Transformer 包含许多类，如果一上来复制几百行代码，报错时很难知道是哪一层出了问题。正确方法是按依赖关系，从不会再拆的小零件开始，每完成一个组件就做最小测试。

这也对应前面学过的层级：组件先正确，才能包装成子层；子层先正确，才能组合成 EncoderLayer 与 DecoderLayer；Layer 正确后才能堆叠为 Encoder 和 Decoder。

### 2:30–5:10　第一组：输入组件

首先实现 Embeddings 和 PositionalEncoding。需要验证：

- token ID 的 [B,L] 是否变成 [B,L,D]；
- 位置表切片能否广播到 batch；
- 两者相加后形状是否不变；
- dropout 在训练与测试模式下行为是否符合预期。

输入端错误会污染后面所有计算，所以先用很小的张量把形状和数值打印清楚。

### 5:10–8:50　第二组：Attention 与 FFN 基础件

接着实现 subsequent_mask、scaled dot-product attention、多头注意力和 position-wise FFN。顺序不能颠倒：多头注意力内部依赖单头 attention，而 Decoder 的自注意力又依赖 mask。

老师强调每一步都要验证性质，而不只是“代码能运行”：mask 上三角是否真的禁止；注意力沿 Key 维的权重和是否为 1；被屏蔽位置权重是否为 0；拆头和合头前后元素总量是否一致。

### 8:50–11:40　第三组：通用包装件与层

然后实现 LayerNorm、SublayerConnection 和 clones。SublayerConnection 把 LayerNorm、具体子层、Dropout、残差相加组织成统一流程；EncoderLayer 和 DecoderLayer 都会复用它。

有了这些基础件，再组合 EncoderLayer，堆叠为 Encoder；组合 DecoderLayer，堆叠为 Decoder。代码结构应与架构图一一对应，类名也应表达语义。

### 11:40–14:10　文件名和模块组织也属于学习内容

老师提醒不要把 Python 文件只命名为 1.py、2.py。数字文件名既不表达用途，也不便于正常 import。更合适的是 embedding.py、attention.py、encoder.py、decoder.py 等语义化名称。

为了教学可以按组件拆文件；真正项目也可按模块组织，但原则一样：名字让人一眼知道职责，依赖方向清楚，底层模块不应反过来导入整模。

### 14:10–结束　最后再组装输出与整模

Encoder 和 Decoder 都通过测试后，增加 Generator：Linear 把 d_model 投影到目标词表，log_softmax 给出对数概率。最后用 make_model 创建所需组件、深拷贝独立层、统一初始化参数并返回完整 EncoderDecoder。

整条实现路线可以口述为：

> 输入组件 → mask/attention/FFN → norm/残差外壳 → EncoderLayer/DecoderLayer → Encoder/Decoder → Generator → 完整模型。

每走一步都先检查形状，再检查数值性质，最后才做整模前向。这是老师安排代码顺序的真正目的：让错误被限制在刚写完的最小范围内。`,

  `### 0:00–1:47　输入部分有源端和目标端两路

老师先回到完整架构：左下方 Input Embedding 接收源语言，右下方 Output Embedding 接收训练时右移后的目标语言。以英法翻译为例，一行数据同时包含已知英语句和已知法语句；英语走 Encoder 输入端，法语前缀走 Decoder 输入端。

两路都由“词嵌入 + 位置编码”组成。词嵌入负责把 token ID 变成向量，位置编码再补充顺序。模型不能直接对字符串做矩阵运算，所以必须先数值化。

### 1:47–5:28　先整理文件和命名，后面才能被正常复用

课程接下来要分多个 Python 文件实现 Transformer。老师反复提醒类名、参数名和文件名保持一致，因为后面的 Encoder、Decoder 和整模文件都会导入这些组件。

Python 模块最好不要用纯数字开头的 01.py、02.py；语义化文件名更容易 import，也让人知道职责。教学代码用 dm01_input 一类名称区分顺序，实际项目可直接使用 embedding.py、position.py 等。类名也不要故意与 nn.Embedding 完全相同，以免阅读时分不清自定义包装和框架类。

### 5:28–9:09　自定义 Embeddings 类需要保存什么

类继承 nn.Module，构造函数接收两个参数：

- vocab_size：词表行数，也就是可用 token 的数量；
- d_model：每个 token 用多少维向量表示。

构造函数先调用 super().__init__()，再创建 nn.Embedding(vocab_size, d_model)。老师用“挂羊头卖狗肉”作口语类比：表面上调用自己的 Embeddings 类，内部真正完成查表的是 PyTorch 提供的 nn.Embedding；自定义类负责统一接口和额外缩放。

### 9:09–13:49　从文字到 ID，再从 ID 查向量

老师再次串起词表流程。假设句子中有“欢迎、来、武汉”，先建立 word_to_index，例如欢迎→0、来→1、武汉→2；送入 Embedding 的是 [0,1,2]，不是三个字符串。

nn.Embedding 本质是一张可训练矩阵，形状 [vocab_size,d_model]。ID 是行号，查到的那一行就是该 token 当前的向量。训练反向传播时，相关行会随损失更新。

课程回顾了 One-Hot、Word2Vec 与 Embedding。One-Hot 直接表示词表位置但十分稀疏；预训练 Word2Vec 可提供已有稠密向量；nn.Embedding 则通常在当前任务中把 ID 查成可学习参数。它们都离不开“词 ↔ ID”的映射。

### 13:49–17:44　前向为什么乘 √d_model

forward 只有一条主线：

\`\`\`python
return self.lut(x) * math.sqrt(self.d_model)
\`\`\`

x 的形状是 [B,L]，查表后成为 [B,L,D]。原始 Transformer 将 embedding 乘以 √d_model，再与位置编码相加。更准确的解释是调整词嵌入与位置编码的相对数值尺度；它不是一个通用的“防止梯度爆炸/消失”公式。缩放让高维模型中的 token 表示在相加时保持合适量级。

### 17:44–21:22　用小张量预测输出形状

老师设置词表大小 1000、d_model=512，创建两个句子，每句四个 token ID，所以输入是 [2,4]。每个 ID 被替换为 512 维向量，输出自然是 [2,4,512]。

读形状时逐维翻译：

- 2：batch 中有两个句子；
- 4：每句四个 token；
- 512：每个 token 的特征维。

数值会很多，不需要逐个理解；先确认形状与 dtype。ID 必须是整型张量，Embedding 输出则是浮点张量。

### 21:22–24:00　课堂报错揭示 vocab_size 的边界

测试中曾放入 3060、5090 一类编号，而 Embedding 只有 1000 行，于是出现 index out of range。若 vocab_size=1000，合法 ID 是 0–999，1000 本身也已经越界。

老师随后把编号改回范围内，模型得到 [2,4,512] 输出。这个现场修正很重要：遇到 Embedding 越界，先检查词表映射、UNK/PAD 编号和输入最大值，不要去改网络后半部分。

最后使用 \`if __name__ == "__main__":\` 包住测试。这样直接运行文件时会执行测试，被其他模块 import 时不会自动打印和构造示例数据。`,

  `### 0:00–2:50　为什么注意力看得到所有词，却不知道先后

老师用多组语序例子引出问题：“我爱你”和“你爱我”、“你欠我 100 元”和“我欠你 100 元”拥有高度相似的词，顺序一换，角色和意义就改变。中文绕口句还会让同一个字在不同位置承担不同词性和含义。

Encoder 自注意力通常允许每个非 PAD 位置读取整句，并不像 RNN 那样按时间步逐个前进。若只把词向量交给它，注意力运算本身对位置置换没有天然的先后概念：处理“爱”时可以同时看到“我”和“你”，却缺少谁在前、谁在后的坐标。

老师用军训排队作类比：你不一定记得所有人的完整信息，但需要知道自己站在哪、前后是谁。位置编码就像给每个位置发一个不会混淆的号，再把这个号所表达的信息加入词向量。

### 2:50–5:44　公式先只认四个量

原始 Transformer 使用固定的正弦/余弦位置编码：

\`\`\`text
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
\`\`\`

先认参数，不必整式死背：

- PE：位置编码表中的一个数；
- pos：token 在序列中的位置，代码通常从 0 开始；
- 2i 与 2i+1：成对的偶数、奇数特征列；
- d_model：词向量和位置向量共同的维度。

每个 pos 最终产生 d_model 个数，才能与同位置的词向量逐元素相加。

### 5:44–9:24　为什么一半用 sin、一半用 cos

偶数列使用 sin，奇数列使用 cos。sin/cos 成对提供相位不同的周期信号；不同 i 又带来不同波长。低维部分变化较快，适合区分邻近位置；高维部分变化较慢，给出更长尺度的坐标。多种频率合起来像一组“位置指纹”。

老师还用三角恒等式说明相对位移关系。对于固定偏移 k，sin(pos+k) 和 cos(pos+k) 可以由 sin(pos)、cos(pos) 的线性组合表达。这使模型有机会通过线性运算学习相对位置模式。

需要校正一个容易误解的说法：标准实现通常直接预先计算或按公式生成位置表，不会在推理时把“位置 5”写成“位置 2 + 位置 3”来节省一次计算。恒等式的主要意义是展示相对位移可线性表达，而预计算表才是避免重复三角函数计算的工程手段。

### 9:24–12:20　大白话：给词向量贴顺序标签

老师用饭店排号继续解释：服务员不必认识每个客人，只需根据号码知道先来后到。位置编码给每个序列位置一个独特标签；Embedding 表示词的内容，PE 表示它所处的坐标，两者相加后成为同时含“谁”和“在哪”的新表示。

位置编码不是在向量末尾追加一个普通整数。若直接追加 pos，数值范围会随长度扩大，结构也很单一；正弦/余弦在所有维度提供连续、多尺度信号，并保持最终接口仍是 [B,L,D]。

### 12:20–17:56　pos、i、d_model 怎样对应下标

以“我爱吃西瓜”为例，从 0 计数时“爱”的 pos=1。假设 d_model=512，特征列下标是 0–511。i 取 0–255：每个 i 同时控制偶数列 2i 和奇数列 2i+1，所以刚好覆盖 512 列。

i=0 对应列 0、1；i=1 对应列 2、3；……；i=255 对应列 510、511。代码中常先构造只含偶数指数的 div_term，再分别填入两组切片。

### 17:56–20:56　10000 指数项控制变化尺度

分母 10000^(2i/d_model) 是频率缩放因子。随着 i 增大，分母变大，pos/分母变化更慢，因此对应 sin/cos 的波长更长。老师把它概括为“i 越大，周期越长”。

数字 10000 是原论文选定的超参数，不是词表大小，也不是最大句长。真正限制可用长度的是实现中准备的位置表 max_len，或动态生成策略。

### 20:56–30:04　用 d_model=4 手算一个位置

为了手算，老师把真实常见的 512 维缩成 d_model=4，并取 pos=2。此时 i 只能取 0、1：

- i=0 生成第 0、1 列：sin(2) 与 cos(2)；
- i=1 生成第 2、3 列：sin(2/100) 与 cos(2/100)。

于是 pos=2 得到一个四维位置向量，大致为 [0.9093,-0.4161,0.0200,0.9998]。再把它与该 token 的四维词向量逐元素相加，得到送入下一层的新向量。

手算的目标不是记住小数，而是看清：一个位置生成 D 个数；偶数列填 sin，奇数列填 cos；不同列使用不同频率；最终与 Embedding 同形相加。`,

  `### 0:00–2:51　老师先总结“为什么好用”

老师把位置编码的价值归纳为两类。第一，给每个位置提供独特信号，让模型区分词序；第二，正弦/余弦具有可表达相对位移的结构，而且位置表可以一次预计算后反复切片使用。

课堂用 pos=5 可以分解成其他位置之和来讲三角函数关系。严格来说，模型前向并不会现场用 pos2 和 pos3 拼出 pos5；代码通常直接读取已经计算好的第 5 行。三角恒等式说明的是固定偏移可用线性关系表达，帮助注意力层学习相对距离。

### 2:51–4:49　面试回答应分四点，但每点要解释

老师给出四个关键词：

1. **记住词序**：不同位置得到不同的多频率标签，“我爱你”与“你爱我”不再只是同一组词向量。
2. **计算高效**：整张 [max_len,D] 表可以在初始化时向量化生成，训练时按当前 L 切片并沿 batch 广播。
3. **可处理不同长度**：只要 L≤max_len，同一套函数能产生任意已准备位置的编码；超过 max_len 则要扩表或动态计算。
4. **提供相对结构**：sin/cos 的相位关系让模型更容易学习“相隔 k 个位置”这样的模式。

最后一句不要说成“位置编码让模型天然完全理解语言”。它只补充顺序信息，语义关系仍需要注意力、FFN、训练数据和目标共同学习。

### 4:49–6:42　四维手算只是教学缩小版

老师提醒，上一节用 d_model=4 是为了让人能把四个数写在纸上；课程代码和原论文示例常使用 512 维。公式不随维度改变，只是 i 的取值范围和位置表列数变多。

学习时应能回答三个层次：

- 大白话：Embedding 说“是什么”，PE 说“在哪里”；
- 形状：两者都是 [B,L,D]，相加后不变；
- 实现：预计算 [1,max_len,D]，前向截取 [:,:L]，随设备迁移但不参与梯度更新。

老师还把这一知识点列入模拟面试范围，并回顾 NER、POS 等早期知识。这提醒我们：Transformer 组件不是新的孤岛，输入仍然建立在前面学过的 token、词表和表示基础上。`,

  `### 0:00–2:49　PositionEncoding 类的三个参数

位置编码继续写在输入模块文件中，因为它和 Embeddings 一起构成 Transformer 输入端。类继承 nn.Module，初始化接收：

- d_model：每个 token 的特征维；
- dropout：相加后的随机失活概率；
- max_len：预先准备多少个位置。

老师示例使用 d_model=512、max_len=60。max_len 不是当前每句话都必须有 60 个词，而是位置表容量上限；当前句长 L 只取前 L 行。

### 2:49–5:45　先准备 Dropout、全零 PE 表和位置列

构造函数先创建 nn.Dropout(p=dropout)，再建立全零矩阵：

\`\`\`python
pe = torch.zeros(max_len, d_model)       # [60, 512]
position = torch.arange(0, max_len).unsqueeze(1)  # [60, 1]
\`\`\`

position 的每一行保存 0、1、2……的序列位置。unsqueeze(1) 把一维 [60] 变成列向量 [60,1]，这样后面与多种频率 [256] 相乘时能广播成 [60,256]。

### 5:45–12:41　div_term 是公式分母的向量化写法

老师把论文中的 10000 指数项改写成便于 PyTorch 向量计算的形式：

\`\`\`python
div_term = torch.exp(
    torch.arange(0, d_model, 2) *
    -(math.log(10000.0) / d_model)
)
\`\`\`

当 d_model=512，arange(0,512,2) 产生 0,2,…,510，共 256 个数，所以 div_term 形状是 [256]。它等价于 1 / 10000^(2i/d_model)。使用 exp 与 log 只是代数变形，不改变公式。

这里最容易看晕的是负号。因为目标是分母的倒数，所以指数必须是负的；若漏掉负号，高维频率会向相反方向变化。

### 12:41–18:37　一次算出所有位置、所有偶奇列

position [max_len,1] 与 div_term [d_model/2] 广播相乘，得到 [max_len,d_model/2]。然后分别填表：

\`\`\`python
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
\`\`\`

0::2 表示第 0、2、4…列；1::2 表示第 1、3、5…列。相同的 [max_len,256] 角度矩阵，一份取 sin 写入偶数列，一份取 cos 写入奇数列，最终 pe 仍是 [max_len,512]。

随后 \`pe.unsqueeze(0)\` 增加 batch 广播维，成为 [1,max_len,d_model]。这里的 1 不是 batch 真的固定为 1，而是表示所有样本共享同一套位置规则。

### 18:37–20:35　为什么使用 register_buffer

位置表不是通过梯度学习的权重，因此不应设成 nn.Parameter；但它又属于模型状态，需要保存到 state_dict，并在 model.to("cuda") 时一起迁移设备。老师用：

\`\`\`python
self.register_buffer("pe", pe)
\`\`\`

注册缓冲区同时满足这两点。若只把 pe 保存成普通属性，某些设备迁移或序列化流程中就需要手工处理。

### 20:35–23:35　forward 只做切片、相加和 Dropout

输入 x 已是词向量 [B,L,D]。前向根据 L 截取：

\`\`\`python
x = x + self.pe[:, :x.size(1)]
return self.dropout(x)
\`\`\`

self.pe 切片是 [1,L,D]，batch 维自动广播到 B，因此输出仍为 [B,L,D]。它融合了内容和位置，但不改变模型接口。

部分旧代码会写 Variable(..., requires_grad=False)。现代 PyTorch 不再需要显式 Variable；register_buffer 已明确位置表不参与梯度。

### 23:35–24:44　本节结束时输入端已经完整

到这里，token ID 先经过 Embeddings 变成缩放词向量，再经过 PositionalEncoding 加入坐标并 Dropout。这个 [B,L,D] 结果才会送入 Encoder 或 Decoder 的第一层。

代码虽短，必须能写出每个中间形状：[max_len,D]、[max_len,1]、[D/2]、[max_len,D/2]、[1,max_len,D]。形状路线说清楚，公式实现就不再神秘。`,

  `### 0:00–3:54　搭建完整测试路线

老师沿用上一节的词表大小 1000 和 d_model=512，创建自定义 Embeddings，并构造两个句子、每句四个 token ID，输入形状 [2,4]。

测试步骤是：

1. ID 经 Embeddings 得到词向量 [2,4,512]；
2. 创建 PositionalEncoding(d_model=512, dropout=...)；
3. 把词向量送入位置编码层；
4. 打印结果与 shape。

最终仍是 [2,4,512]。形状不变不代表什么都没做；数值已经加入不同位置的 PE，并可能经过 Dropout。

### 3:54–7:51　max_len=60 与实际 L=4 不冲突

老师专门追问：明明句子只有四个词，为什么位置表准备了 60 行？因为 60 是“最多支持的长度”，本批只需前四行。PE 缓冲区形状是 [1,60,512]，实际切片为 [1,4,512]。

位置编号从 0 到 59；当前四个 token 使用 0、1、2、3。若将来输入 61 个位置，当前表就不够，代码应报错、扩表或动态生成，而不是静默重复位置。

### 7:51–11:51　逐个打印中间形状来反推

老师在源码中临时加入打印，追踪：

- position：[60,1]；
- div_term：[256]；
- position * div_term：[60,256]；
- pe 填完偶奇列：[60,512]；
- unsqueeze 后：[1,60,512]。

这是一种很实用的调试办法：不懂广播时不要盯着整段公式，先打印参与相乘的两个 shape，再按“60×1 与 256 → 60×256”解释。看懂后应删掉库组件中的临时打印，避免训练时每个 batch 都刷屏。

### 11:51–14:51　数值很长，测试应关注结构和性质

老师尝试打印完整 PE，发现大量小数不适合逐个阅读。更有意义的是检查：

- PE 是否是浮点张量；
- 同一位置的 D 个数是否成对来自 sin/cos；
- pos=0 时偶数列 sin(0)=0，奇数列 cos(0)=1；
- 没有 Dropout 时，同位置在不同 batch 中编码相同。

只打印最终大矩阵容易“看到了很多数，却不知道对不对”。形状断言和上述数值性质更可靠。

### 14:51–17:50　广播相加的完整解释

Embedding 结果是 [2,4,512]；从 PE [1,60,512] 截出 [1,4,512]。相加时第一维 1 广播成 2，相同的位置表被加到 batch 中两句话的对应位置，结果仍为 [2,4,512]。

这里共享的是位置规则，不是词向量。第一个样本位置 2 和第二个样本位置 2 加上相同 PE，但二者原始 token embedding 可以完全不同。

### 17:50–18:49　测试完成后数据将进入 Encoder

老师最后重新口述：先建立 PE 全零矩阵，计算偶数和奇数列，注册缓冲区；前向按句长切片，与词向量相加并 Dropout。融合后的张量就可传给 Encoder。

为了可复现地检查“相加是否正确”，测试时应将 dropout=0 或调用 eval()；否则随机失活会让重复前向的具体数值不同。`,

  `### 0:00–1:47　写 Encoder 前先补矩阵基础

老师准备进入编码器代码，但注意力中的 mask 会用到三角矩阵，所以先单独讲上三角和下三角。方阵的主对角线从左上走到右下；主对角线上方保留数值、下方为 0，称为上三角；反过来则是下三角。

三角矩阵不是只有 0 和 1，定义看的是哪些区域必须为 0。课程用全 1 矩阵演示，是为了让允许区和禁止区一眼可见。

### 1:47–4:34　因果生成为什么需要遮住未来

老师用“今天不能知道明天下午发生什么”类比 Decoder 生成。预测第 1 个 token 时，未来全部未知；预测第 2 个时，只能使用第 1 个和当前位置的目标输入；越往后，已知前缀逐步增长。

训练时完整目标句已经装在张量中，若不主动遮住右侧未来位置，模型会从答案中抄信息。因果 mask 的任务就是让第 i 个 Query 只访问 j≤i 的 Key。

### 4:34–8:24　用 np.triu 观察上三角及 diagonal

老师先记忆“上三角：主对角线下方为 0”。NumPy 的 \`np.triu(m, k)\` 保留指定对角线及其上方：

- k=0：保留主对角线；
- k=1：从主对角线上方第一条线开始；
- k>1：继续上移；
- k<0：向下移动，允许保留部分下方区域。

课程反复改变 k 并打印矩阵，目的不是背 API，而是看懂“对角线偏移决定哪一格开始保留”。

### 8:24–13:05　代码文件与后续注意力的关系

这一部分写在 Encoder/Attention 相关模块中，并导入前面已经实现的输入组件。老师再次提醒文件名不能只用纯数字，后面的文件还需要正常 import。

实际注意力公式包含 QKᵀ/√d_k，mask 会作用在这个分数矩阵上。此刻只演示三角矩阵，还没有做 Softmax，也没有真正得到注意力权重。

### 13:05–18:30　diagonal 正负方向的课堂调试

老师用小矩阵依次打印默认 k=0、上移和下移后的结果。现场口述容易把“上移/下移”说反，最安全的办法是直接记坐标：

> 对 scores[i,j]，j>i 是未来；若要标记严格未来区，就使用主对角线上方。

生成“未来禁止区”通常用 \`torch.triu(torch.ones(L,L), diagonal=1)\`；之所以从 1 开始，是当前位置 i 可以看自己。若从 0 开始并把保留区当禁止区，主对角线也会被遮掉。

本节得到的只是区域模板。下一节还要决定 True/False 到底代表允许还是禁止，并增加广播维度。`,

  `### 0:00–1:56　subsequent_mask 从全 1 方阵开始

老师把“不能看未来”封装成函数，输入 size=L，先建立 [1,L,L] 的全 1 数组。最前面的 1 是预留广播维，不是句长，也不是注意力头数。

接着用 \`np.triu(..., k=1)\` 取严格上三角。这里得到的是未来区域：第 i 行中 j>i 的格子为 1，其他位置为 0。

### 1:56–4:52　为什么还要做一次逻辑取反

课程后面的约定是 mask=True/1 表示允许，False/0 表示禁止。但 triu(k=1) 刚好把未来标成 1，语义相反，所以需要比较 \`== 0\` 或逻辑取反：

\`\`\`python
future = np.triu(np.ones((1, L, L)), k=1).astype("uint8")
allowed = torch.from_numpy(future) == 0
\`\`\`

结果是下三角可见区：当前位置和左侧过去为 True，右侧未来为 False。老师在现场先打印上三角，再修改返回值，这段纠正展示了为什么必须先定义 mask 语义。

### 4:52–6:50　函数没有 return，会出现“打印有结果、调用却是 None”

课堂调试中，上三角矩阵在函数内部能打印，但外部拿不到下三角结果，原因是函数忘记 return。补上返回后，测试又同时出现函数内部打印与外部打印，导致看起来输出两次。

这是很典型的 Python 调试现象：

- print 只显示临时信息；
- return 才把张量交给调用者；
- 若两处都 print，就会重复显示。

完成验证后，应删除函数内部的调试打印，让 subsequent_mask 只负责产生 mask。

### 6:50–8:31　三维 mask 与二维切片

函数返回 [1,L,L]。老师用 \`mask[0]\` 取出第一个切片，得到 [L,L]，便于打印和画图。方括号数量对应维度层级：三维张量索引一次，少一维。

真实注意力一般保留广播维。进入多头模块时还可能再 unsqueeze 成 [B,1,L,L]，从而对 h 个头共享。

### 本节最终约定

第 i 行代表当前 Query，第 j 列代表要读取的 Key：

- j≤i：True，允许读取自己和过去；
- j>i：False，禁止读取未来。

不同库可能用相反布尔语义，因此不能只凭“三角形朝哪边”抄代码；必须同时检查后面 masked_fill 使用的是 \`mask==0\`、\`~mask\` 还是直接 mask。`,

  `### 0:00–1:56　先约定颜色代表什么

老师把 subsequent_mask 画成热力图。课堂图中黄色/亮色代表允许读取，暗色代表被遮住。读任何热力图前要先看图例，因为有些实现会专门画“禁止区”，颜色含义正好相反。

mask 本体为 [1,L,L]，可视化时取第 0 个切片变成 [L,L]。横纵轴应标成 Key 位置和 Query 位置，而不是笼统写“词编号”。

### 1:56–3:55　imshow 只是把布尔矩阵画出来

老师用 Matplotlib 的 imshow 显示 5×5 mask，设置 figure size 后调用 show。若图太小，先调整画布，而不是改 mask 数值。

更适合学习的图应同时显示网格和 0/1 数字。颜色帮助看整体三角形，数字帮助检查具体格子。例如 mask[1,2] 必须为 False，因为第 2 个 Query（从 0 计）之前，索引 2 对它仍是未来。

### 3:55–5:49　逐行读“当前可见前缀”

按“行是 Query、列是 Key”读取：

- 第 0 行：只允许第 0 列；
- 第 1 行：允许第 0、1 列；
- 第 2 行：允许第 0、1、2 列；
- 最后一行：整行都允许，因为已经没有更晚位置。

老师口头用“竖着看”帮助学生观察，但矩阵语义最终必须落实到行列约定。若转置了热力图，三角方向会翻转，肉眼仍像三角形，却会遮错对象。

### 可视化之后还要做的两项断言

热力图只能说明 mask 本身看起来合理。完整验证还需检查：

1. mask 形状能广播到注意力 scores；
2. Softmax 后，被遮住的严格上三角权重确实为 0。

若某一行全被遮住，Softmax 没有合法 Key，可能产生 NaN 或异常分布。因果 mask 保留主对角线，正好保证每行至少能看自己。`,

  `### 0:00–2:54　写注意力前先单独练 masked_fill

老师准备实现自注意力，但先把遮盖动作拆成最小实验。Self-Attention 中 Q、K、V 来自同一份序列；真正计算权重前，需要知道怎样把不允许的位置从 scores 中排除。

测试代码仍放在语义化文件中，避免纯数字文件无法正常 import。把 masked_fill 单独验证后，后面 attention 函数只需复用这条已知正确的操作。

### 2:54–6:24　准备同形 input 与 mask

老师创建一个 3×5 浮点 input，并建立同形 mask。mask 中可能含 0、1 或其他非零数；先通过 \`mask != 0\` 转成布尔张量：

- 非零 → True；
- 0 → False。

这一步明确后续条件语义。实际 attention 的 mask 可能通过广播扩展，不一定与 scores 完全同形，但最后必须能对应每个要替换的格子。

### 6:24–9:38　masked_fill 的三个参数角色

核心调用可读成：

\`\`\`python
result = input.masked_fill(mask == 0, -1e9)
\`\`\`

逐项解释：

1. input：谁要被修改；
2. mask==0：哪些位置满足条件；
3. -1e9：满足条件的位置替换成什么。

老师把 mask==0 的格子换成极小负数，其余 input 元素保持原值。它不是删除元素，所以 shape 不变。

### 9:38–12:39　为什么是极小负数，而不是 0

masked_fill 发生在 Softmax 之前。Softmax 内部包含指数运算：极小负数的 exp 接近 0，于是被禁止位置最终权重接近 0。

若填 0，exp(0)=1，被遮位置仍可能获得明显概率。若先 Softmax 再把概率乘 0，剩余权重之和不再等于 1，除非再次归一化。因此标准顺序是：

> scores → masked_fill(极小负数) → Softmax。

### 12:39–13:51　数值和布尔约定都不是固定写死的

老师演示 -1e9、-1e10 等值，表达的都是“近似负无穷”。更稳健的实现可使用 \`torch.finfo(scores.dtype).min\`，尤其在 float16、bfloat16 等精度下避免常数不合适。

同样，mask 是 0 禁止还是 1 禁止取决于项目约定。本课程使用 0/False 表示遮盖，所以条件写 \`mask == 0\`。换一个库时必须先读接口说明，不能盲抄取反。`,

  `### 0:00–3:45　attention 函数的输入和形状

老师开始把前面的零件组成缩放点积注意力。函数接收 query、key、value、可选 mask 和可选 dropout。

在单头教学示例中，Q、K、V 可是 [B,L,D]；多头模块调用时则通常是 [B,h,L,d_k]。不要把 d_model 与 d_k 混用：单头时两者可能恰好相同，多头拆分后 d_k=d_model/h。

mask 可以为 None，是为了让同一函数同时服务不需要因果遮盖的 Encoder 自注意力、需要 PAD mask 的注意力，以及 Decoder 因果注意力。

### 3:45–7:26　第一步：QKᵀ/√d_k

先取 \`d_k = query.size(-1)\`，再计算：

\`\`\`python
scores = torch.matmul(query, key.transpose(-2, -1))
scores = scores / math.sqrt(d_k)
\`\`\`

transpose(-2,-1) 交换最后两个轴，不把 batch、head 等前导维写死，因此三维和四维张量都适用。若 Q=[B,Lq,d_k]、K=[B,Lk,d_k]，scores=[B,Lq,Lk]。

每个 scores[...,i,j] 表示第 i 个 Query 与第 j 个 Key 的匹配程度。

### 7:26–12:09　第二步：有 mask 才遮盖

老师使用 \`if mask is not None\` 让函数通用。如果提供 mask，就把 mask==0 的 scores 替换为极小负数。Encoder 不需要因果 mask，但通常仍需 PAD mask；Decoder 目标自注意力需要把 PAD 与未来遮盖组合起来。

课堂重点追问“为什么 0 位置要替换成极小值”。答案不是直接让分数等于权重 0，而是让下一步 Softmax 后的指数贡献接近 0。

### 12:09–15:03　第三步：Softmax、Dropout 和 Value 汇总

沿 scores 最后一维，也就是 Key 维做 Softmax：

\`\`\`python
p_attn = torch.softmax(scores, dim=-1)
\`\`\`

对固定 Query，它分配给所有 Key 的权重和为 1。若传入 dropout，就对权重做随机失活。最后：

\`\`\`python
output = torch.matmul(p_attn, value)
\`\`\`

权重 [B,Lq,Lk] 乘 V [B,Lk,d_v]，得到 [B,Lq,d_v]。输出位置数由 Query 决定；“读什么内容”来自 Value。

### 15:03–18:47　函数为什么返回两个结果

attention 返回 \`output, p_attn\`。output 继续传给网络；p_attn 既可用于多头模块保存，也可可视化每个词关注了哪些位置。

老师提醒不要重复手工构造前面已经完成的 Embedding 与位置编码结果，可以从已写模块导入。组件化的意义正是让测试数据和函数能复用。

### 18:47–23:27　自注意力测试：Q=K=V

课程把输入端结果作为 Q、K、V，因此三者 shape 相同，内容也相同。若输入 [2,4,512]，单头注意力输出仍是 [2,4,512]；权重矩阵是 [2,4,4]。

“输出 shape 与输入相同”不表示计算没有效果。每个输出位置已经按权重混合了四个 Value。应同时打印权重 shape，并检查沿最后一维求和约等于 1。

### 23:27–30:44　老师用“我爱武汉中国”解释 4×4 权重

对长度为 4 的句子，每个词都要和四个词比较：第一个 Query 产生 1×4 权重，四个 Query 叠成 4×4；batch 中两句话再叠成 [2,4,4]。

例如“爱”这一行分别表示它对“我、爱、武汉、中国”的依赖程度。若有因果 mask，右侧未来格子为 0；若是 Encoder 双向自注意力，非 PAD 位置通常都可参与。

本节完整口述路线是：

> 匹配 Q 与 K → 缩放 → 遮盖非法位置 → Softmax 得权重 → Dropout → 加权读取 V。

调试时优先检查四个性质：scores 末两维 Lq×Lk、权重和为 1、mask 区权重为 0、output 长度等于 Lq。`,

  `### 0:00–3:42　为什么一个“头”不够

老师用同学分组作类比：只问一个人，很可能只得到一种观察角度；问多个人，再汇总各自看到的优点、缺点和偏好，信息更丰富。多头注意力让多组投影在不同表示子空间里提取关系，而不是把完全相同的单头结果复制八遍。

一个头可能偏向主谓关系，一个头偏向局部搭配，另一个头可能捕捉远距离指代。具体每个头学什么不是人工硬编码，而由参数和训练目标决定；也不能保证所有头都同样有用。

### 3:42–6:58　投影、分头、独立注意、合并

老师先给出完整动作：

1. Q、K、V 分别经过线性投影；
2. 把 d_model 拆成 h 个头；
3. 每个头独立做 Scaled Dot-Product Attention；
4. Concatenate 拼接各头；
5. 输出线性层再次融合。

课程用火锅分格作类比：同一批食材进入不同格子，因锅底或搭配不同产生不同味道；最后夹回同一个盘子。类比只帮助理解“分开加工再合并”，数学上每个头真正不同来自各自的投影参数。

### 6:58–10:28　从 [2,4,512] 拆成 8 头

假设输入 [B,L,D]=[2,4,512]，头数 h=8，每头维度 d_k=512/8=64。

第一步 reshape：

\`\`\`text
[2,4,512] → [2,4,8,64]
\`\`\`

这一步没有增加或删除元素，只把 512 拆成 8×64。随后 transpose 序列轴与头轴：

\`\`\`text
[2,4,8,64] → [2,8,4,64]
\`\`\`

现在每个 batch 有 8 个独立头，每头都保留完整 4 个 token，各 token 用 64 维表示。

### 10:28–15:24　为什么头轴必须放在序列轴前

若保持 [B,L,h,d_k]，矩阵乘法的最后两维会把 h 与 d_k 当作配对轴，无法对每个头的整段序列分别计算 L×L 权重。转成 [B,h,L,d_k] 后，最后两维正好是“序列 × 每头特征”，QKᵀ 得到 [B,h,L,L]。

老师反复画格子，是为了说明每个头都要看到四个词，而不是把四个词分别分配给四个头。错误理解会变成“一个头只处理一个词”，这会破坏注意力跨位置比较。

### 15:24–18:20　合头后为什么仍回到 512 维

每头输出 [B,h,L,64]。先 transpose 回 [B,L,h,64]，再把 h×64 合并：

\`\`\`text
[2,8,4,64] → [2,4,8,64] → [2,4,512]
\`\`\`

输出形状与输入一致，才能做残差相加并送入后续 FFN。形状没变，但每个位置已经汇总了八种投影视角。

### 18:20–19:19　本节应牢记的约束

d_model 必须能被 h 整除，否则无法让每头拥有相同 d_k。课程常用 512 和 8；真实模型也可使用不同 D、h，但要满足整除或采用专门的非等宽设计。

多头的主线不是背“八个头”，而是能口述：

> 线性投影 → 拆 D → 换轴 → 每头注意力 → 换轴 → 合 D → 输出投影。`,

  `### 0:00–1:54　把图解压缩成可复述的五步

老师不再画完整火锅图，而是把多头注意力写成文字流程：

1. Q、K、V 做线性变换；
2. reshape 并 transpose，拆成多头；
3. 每个头独立做注意力；
4. concatenate 各头结果；
5. 再做一次线性变换。

这里先投影再分头很重要。若只把原始 512 维机械切八段，不经过不同参数的 Q/K/V 投影，各头的表达能力和分工会受限。

### 1:54–4:45　分头前后元素总数不变

老师继续使用 [2,4,512]。512 拆成 8×64 后为 [2,4,8,64]，换轴后是 [2,8,4,64]。这只是重排，不是把一句话复制八份，也没有让 batch 变大。

可以用元素数自检：

\`\`\`text
2×4×512 = 2×4×8×64 = 4096
\`\`\`

若 reshape 前后元素数对不上，参数一定写错。

### 4:45–6:34　Concatenate 做什么

八个头各自得到 [2,1,4,64] 一类结果，沿特征方向拼回 [2,4,512]。老师把它说成汇总不同组的信息，让输出规整。

拼接本身只把分开的特征排回一条向量。随后输出 Linear 才能在各头特征之间重新混合，让模型学习怎样综合这些视角。

### 6:34–8:25　“八只眼睛”是帮助记忆的类比

老师用“八只眼睛看不同细节，最后汇总成整体”总结。类比要配合三条技术事实：

- 每只“眼睛”由不同的可学习投影形成；
- 每个头仍然看到整段可见序列；
- mask 通常沿 head 维广播，但各头算出的权重可以不同。

最后要能不靠类比写出形状路线。对于任意 B、L、D、h：

\`\`\`text
[B,L,D] → [B,h,L,D/h] → attention → [B,h,L,D/h] → [B,L,D]
\`\`\`

transpose 后张量内存可能不连续，代码在最终 view 前通常要调用 contiguous，或直接使用 reshape。`,

  `### 0:00–3:56　先写 clones：复制结构，但参数必须独立

老师先实现：

\`\`\`python
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
\`\`\`

这个工具不只服务四个 Linear，后面 EncoderLayer、DecoderLayer 和残差子层也会反复用。deepcopy 会复制结构和参数张量；若只把同一个 module 引用放 N 次，所有位置会共享同一组参数。

ModuleList 让 PyTorch 正确注册子模块，优化器和 state_dict 才能看到它们。普通 Python list 虽能存对象，但不会自动完成模型参数管理。

### 3:56–7:55　初始化时先做整除断言

MultiHeadedAttention 接收 h、d_model、dropout。第一条检查是：

\`\`\`python
assert d_model % h == 0
\`\`\`

随后 \`d_k=d_model//h\`。例如 512//8=64。整除失败时尽早报错，比运行到 view 时出现难懂的元素数错误更容易定位。

类还保存 h、四个线性层、注意力权重占位和 Dropout。前三个 Linear 用于 Q/K/V，第四个用于合头后的输出投影。

### 7:55–10:58　为什么是四个 Linear

基础 Linear 的形状都是 D→D：

- linear[0]：WQ；
- linear[1]：WK；
- linear[2]：WV；
- linear[3]：WO。

前三个生成不同的 Q、K、V 表示。即使 Self-Attention 的原始输入相同，投影后也不再数值相等。第四个在线程结果拼接后融合各头信息。

### 10:58–15:18　先处理 mask 的 head 广播维

若 mask 不为 None，代码通常：

\`\`\`python
mask = mask.unsqueeze(1)
\`\`\`

例如 [B,L,L] 变成 [B,1,L,L]。这个 1 会广播到 h 个头，所以每个头遵守相同的因果/PAD 可见规则。若原 mask 已经带 head 维，再重复 unsqueeze 就会多一轴，因此项目要统一入口形状。

接着记录 \`nbatches=query.size(0)\`，不要把 batch 写死为 2。

### 15:18–21:34　用 zip 同时投影并拆分 Q/K/V

老师用列表推导和 zip，把三个 Linear 与 query、key、value 对应起来。每个张量依次经过：

\`\`\`text
[B,L,D]
→ Linear [B,L,D]
→ view [B,L,h,d_k]
→ transpose [B,h,L,d_k]
\`\`\`

若 Q 与 K 的长度不同，view 中必须使用各自 \`x.size(1)\`，不能统一拿 Query 长度。这样同一实现才能支持 Cross-Attention 的 Lt 与 Ls。

### 21:34–24:38　调用单头函数，再合并各头

拆好后调用前面完成的 \`attention(query,key,value,mask,dropout)\`，得到 x=[B,h,Lq,d_k] 与权重 [B,h,Lq,Lk]。

合头路线：

\`\`\`python
x = x.transpose(1, 2).contiguous()
x = x.view(nbatches, -1, h * d_k)
\`\`\`

transpose 后先 contiguous，因为换轴后的 stride 可能不允许直接 view。最终恢复 [B,Lq,D]。

### 24:38–25:07　最后输出投影

\`return self.linears[-1](x)\` 用第四个 Linear 混合拼接后的头特征。输出保持 [B,Lq,D]，以便残差连接。

实现完成后，用三条路线复查：

- 参数路线：4 个独立 Linear；
- 形状路线：[B,L,D]↔[B,h,L,d_k]；
- mask 路线：[B,Lq,Lk]→[B,1,Lq,Lk]。

这三条任意一条混乱，代码都可能形状报错或悄悄做错注意力。`,

  `### 0:00–3:41　测试函数继续复用输入端结果

老师创建 MultiHeadedAttention(h=8,d_model=512)，并把已经经过 Embedding 与位置编码的 [2,4,512] 张量作为 Self-Attention 的 Q、K、V。

因为是自注意力，三者来自同一输入；但进入模块后会分别通过 WQ、WK、WV，所以投影结果不同。测试代码返回多头输出，方便后续 EncoderLayer 直接使用。

### 3:41–6:42　先故意改头数，看整除检查是否生效

课堂把头数从可整除设置改成不合适值，触发 assert 或 reshape 错误，再改回 8。这个实验验证了 d_model%h==0 不是装饰。

当 D=512、h=8，每头 64 维；输出仍应为 [2,4,512]。若只看最终 shape，可能漏掉内部错误，因此还要查看 \`self.attn\` 的 [2,8,4,4]。

### 6:42–10:39　课堂现场定位 transpose/view 问题

老师沿源码回看 view 与 transpose 的轴顺序，并解释 batch_size 不应写死。拆头需要 [B,L,h,d_k]→[B,h,L,d_k]；合头要反过来，再 contiguous/view。

当 shape 报错时，最有效的做法不是随意改 -1，而是逐行打印：

- Linear 后；
- view 后；
- transpose 后；
- attention 后；
- 合头后。

每一步与纸面路线比对，错误会被限制在一行。

### 10:39–13:36　transpose、view 与内存顺序

老师补充张量在内存中按顺序存储。transpose 只交换两个轴的视图，底层 stride 会改变；view 要求新形状与内存布局兼容，所以合头前调用 contiguous。

\`view\` 可一次重塑多个维度，\`transpose\` 一次交换两个指定轴。二者职责不同，不能把“改变显示形状”统称为同一个操作。

### 13:36–15:20　权重 shape 的语义

对 batch=2、长度=4、头数=8：

\`\`\`text
output: [2,4,512]
attn:   [2,8,4,4]
\`\`\`

attn 中每个 [4,4] 表示一个头里四个 Query 对四个 Key 的权重。每一行沿最后维应和为 1；若传因果 mask，上三角应为 0。

课堂主要打印形状。更严格测试还应将 dropout=0，并断言：

\`\`\`python
assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)))
\`\`\`

不要使用全 1 mask 作为唯一测试，因为它无法证明遮盖代码真正生效；至少准备一个同时含允许与禁止格子的 mask。`,

  `### 0:00–2:47　Attention 交流位置，FFN 加工特征

老师先解释前馈全连接子层的作用。注意力把不同 token 的信息汇总到当前位置，但一次线性加权未必能充分提取复杂特征；FFN 再对每个位置做更强的非线性变换。

“Position-wise”不是每个位置各有一套参数，而是所有位置共享同一组网络，分别处理各自向量。可以类比同一位老师用同一套方法分别辅导每名学生；学生输入不同，结果不同，但规则相同。

### 2:47–4:45　为什么先升维再降维

原始 Transformer 常用 d_model=512、d_ff=2048：

\`\`\`text
512 → 2048 → 512
\`\`\`

先扩到更宽的隐藏空间并加入 ReLU，模型能组合出更多非线性特征；再投回 512，才能与子层输入做残差相加，并继续保持各层统一接口。

扩维不是增加序列长度。输入 [B,L,512] 经第一层变 [B,L,2048]，B、L 不变。

### 4:45–7:43　类中保存两个 Linear 与 Dropout

PositionwiseFeedForward 初始化 d_model、d_ff、dropout：

\`\`\`python
self.w_1 = nn.Linear(d_model, d_ff)
self.w_2 = nn.Linear(d_ff, d_model)
self.dropout = nn.Dropout(dropout)
\`\`\`

前向路线：

\`\`\`python
return self.w_2(self.dropout(F.relu(self.w_1(x))))
\`\`\`

顺序是 Linear→ReLU→Dropout→Linear。课程版本通常不在第二个 Linear 后再加激活，因为后面还有残差与 LayerNorm。

### 7:43–10:40　测试只改变中间维，最终 shape 还原

用 [2,4,512] 输入，w_1 输出 [2,4,2048]，w_2 输出 [2,4,512]。老师建议把 d_ff 作为初始化参数而不是写死，测试时可换小维度观察。

FFN 不会把位置 0 和位置 3 直接相乘；每个位置的 512 维向量独立经过相同两层网络。跨 token 关系已经由前面的 Attention 写入每个位置的向量。

### 10:40–11:57　老师强调“先能讲出来，再扣代码”

课程最后把 Transformer 与现代大模型联系起来：Encoder 路线常服务理解任务，Decoder 路线常服务生成任务。课堂口述中把 NLU/NLG 名称说得有些混乱，标准含义是 Natural Language Understanding 与 Natural Language Generation。

老师建议晚上重新梳理代码，第一目标不是逐字符背诵，而是能用自己的话说清每个组件的作用和 shape。对 FFN，合格回答至少包含：

> 注意力完成位置间通信；FFN 对每个位置共享参数地做 512→2048→512 非线性加工；输出维度还原是为了残差连接。`,

  `### 0:00–4:51　为什么深层网络需要归一化

老师从装修铺砖、汽车保养作类比：每一步只有一点小偏差，层层累积后可能越来越严重。神经网络也会遇到中间激活尺度漂移、梯度不稳定等问题，归一化帮助各层输入保持更可控的数值分布。

LayerNorm 先对指定特征求均值与标准差，标准化后再乘可学习缩放 a、加可学习偏移 b。后两步很重要：模型可以在稳定训练的同时，重新学回任务需要的尺度和中心。

### 4:51–9:48　代码放在哪里，以及类的职责

老师继续在组件文件中定义 LayerNorm，并提醒把长文件中的代码折叠管理。后面的 Sublayer、EncoderLayer、Encoder、DecoderLayer、Decoder 都会复用它，所以现在必须把接口写稳定。

类继承 nn.Module，接收 features=d_model 和 eps。features 决定 a、b 的长度；eps 是防止标准差为 0 时除零的小常数。

### 9:48–13:46　可学习参数 a 和 b

构造函数创建：

\`\`\`python
self.a_2 = nn.Parameter(torch.ones(features))
self.b_2 = nn.Parameter(torch.zeros(features))
\`\`\`

a 初始为 1，b 初始为 0，所以初始化时不会额外改变标准化结果。nn.Parameter 让它们参与反向传播。名称 a_2/b_2 是课程/经典实现的命名，不表示平方运算。

老师用线性公式 y=kx+b 回顾缩放和平移：a 控制每个特征维的幅度，b 控制中心偏移。

### 13:46–16:43　沿最后一维求均值与标准差

输入 x=[B,L,D]，LayerNorm 对每个 batch、每个 token 的 D 个特征独立归一化：

\`\`\`python
mean = x.mean(-1, keepdim=True)  # [B,L,1]
std = x.std(-1, keepdim=True)    # [B,L,1]
\`\`\`

keepdim=True 保留最后一个长度为 1 的轴，后面才能自动广播回 [B,L,D]。它不会把 batch 中不同句子混在一起，也不会把不同 token 的均值合并。

### 16:43–21:20　标准化、缩放和平移

前向返回：

\`\`\`python
self.a_2 * (x - mean) / (std + self.eps) + self.b_2
\`\`\`

x-mean 让每个位置的特征中心接近 0；除以 std 让尺度接近 1；eps 防止常量向量造成除零；a、b 再提供可学习自由度。

需要注意课程手写实现常用 PyTorch \`std\` 默认的样本标准差，而框架 \`nn.LayerNorm\` 使用总体方差定义。为了严格对齐官方实现，配套代码使用 \`var(unbiased=False)\` 或 \`std(unbiased=False)\`。差异在小 D 测试中会看得比较明显。`,

  `### 0:00–3:40　测试沿真实子层路线取数据

老师复用前面多头注意力和 FFN 的输出，再送入 LayerNorm：

\`\`\`text
输入 → 多头注意力 → Position-wise FFN → LayerNorm
\`\`\`

这样可验证组件衔接时 shape 一直是 [2,4,512]。不过它属于小型集成测试；LayerNorm 还应直接使用可控输入做单元测试。

### 3:40–5:30　归一化不会改变 shape

老师打印注意力、FFN 与 LayerNorm 的结果，它们都是 [2,4,512]。LayerNorm 改变每个 token 特征的数值分布，不改变 batch、序列长度和特征数。正因为接口不变，才能插在深层网络的子层之间。

### 5:30–7:08　真正该验证的是均值、方差和边界

沿最后一维检查：

\`\`\`python
y = norm(x)
assert torch.allclose(y.mean(-1), torch.zeros_like(y.mean(-1)), atol=1e-5)
assert torch.allclose(
    y.var(-1, unbiased=False),
    torch.ones_like(y.var(-1, unbiased=False)),
    atol=1e-4,
)
\`\`\`

由于 eps、浮点误差和可学习 a/b，不能用完全相等。实现和测试还必须使用一致的 unbiased 定义。

再加入所有特征相同的常量向量，确认输出不含 NaN/Inf，才能证明 eps 在零方差场景生效。LayerNorm 与残差共同提高深层训练稳定性，但它不是保证梯度永不消失的万能公式。`,

  `### 0:00–1:53　两者公式相似，关键差异是统计轴

老师把 Layer Normalization（LN）与 Batch Normalization（BN）画在同一张表里。二者都做“减均值、除标准差、再缩放平移”，最重要的区别不是公式，而是哪些元素一起计算统计量。

### 1:53–4:52　BatchNorm 竖着看：同一特征跨样本

老师用表格的每一行表示样本、每一列表示特征。BatchNorm 对同一列跨多个样本统计，因此可形象理解为“竖着归一化”。

它依赖 batch 统计量：训练时用当前批次的均值/方差并更新运行统计；推理时通常使用累计的 running mean/variance。batch 太小或样本长度/分布变化大时，统计可能不稳定。

在卷积网络中还会把空间维一起纳入统计，不能只把二维表格规则原封不动套到所有张量；核心仍是“同一通道/特征跨样本统计”。

### 4:52–6:49　LayerNorm 横着看：单个 token 跨特征

LayerNorm 对每个样本内部、每个 token 的 D 个特征统计，可理解为“横着归一化”。它不依赖同 batch 还有多少句子，训练和推理使用相同的即时统计方式。

对 [B,L,D]：

- BN 的具体轴取决于布局和模块设计；
- LN(D) 明确沿最后的 D 维做。

Transformer 常面对变长文本、小 batch 或自回归逐 token 推理，LayerNorm 更自然，因此成为常用选择。

### 结论不要背成绝对规则

“文本只能用 LN、图片只能用 BN”过于绝对。选择归一化方式要看数据布局、batch 大小、训练方式和架构。卷积视觉模型仍广泛使用 BN，视觉 Transformer 也常使用 LN；还有 RMSNorm、GroupNorm 等选择。

面试回答最好先画统计轴，再说训练/推理差异和适用场景，这比只背“Transformer 用 LN”更可靠。`,

  `### 0:00–2:39　子层像积木外壳，Attention 与 FFN 都能装进去

前面的 Multi-Head Attention 和 FFN 是不同计算，但它们外面都需要归一化、Dropout 与残差连接。老师把共同流程封装成 SublayerConnection，避免在 Encoder/Decoder 中重复同一段代码。

EncoderLayer 有两个这样的外壳：一个包 Self-Attention，一个包 FFN；DecoderLayer 有三个。外壳只要求内部函数接收 x 并返回与 x 同形状张量。

### 2:39–6:17　新文件导入已有组件

老师新建语义化模块，并导入 LayerNorm、MultiHeadedAttention、PositionwiseFeedForward 等。现在进入“用零件组层”的阶段，文件依赖应从上层指向下层，基础组件不要反向导入完整 Encoder。

SublayerConnection 构造函数接收 size 与 dropout，内部创建 LayerNorm(size) 和 Dropout。

### 6:17–10:03　forward 为什么接收一个可调用 sublayer

接口大致为：

\`\`\`python
def forward(self, x, sublayer):
    ...
\`\`\`

sublayer 不是固定类名，而是一个可调用对象。传入注意力 lambda 时，它完成 Q/K/V 与 mask；传入 FFN 时，它直接加工 x。这样同一外壳不必知道内部细节。

### 10:03–14:54　残差路线必须保留原始 x

课程采用经典 Annotated Transformer 的 Pre-LN 写法：

\`\`\`python
return x + self.dropout(sublayer(self.norm(x)))
\`\`\`

数据分两路：

- 直连路：原始 x 不变地走到加号；
- 变换路：x→LayerNorm→具体子层→Dropout。

两路同形相加，输出 shape 不变。残差让深层网络容易保留已有信息，也为梯度提供较短通道。

### 14:54–16:21　Pre-LN 与原论文图不要混

原始论文架构图常表示 Post-LN：\`Norm(x+Dropout(Sublayer(x)))\`；课程累计代码采用 Pre-LN：\`x+Dropout(Sublayer(Norm(x)))\`。两者都是真实设计，但顺序不同。

学习本项目时以实际代码为准，不能图上背一套、代码里拼另一套。无论哪种形式，残差相加都要求子层输出与 x 完全同形；若 FFN 忘记投回 d_model，或多头没合回 D，就会在这里报错。`,

  `### 0:00–3:55　测试要同时准备数据、外壳和内部函数

老师先创建 SublayerConnection(size=512,dropout=...)。完整调用还需要两样东西：

- x：待处理的 [B,L,D]；
- sublayer：接收一个张量并返回同形状结果的可调用对象。

这揭示了外壳设计：SublayerConnection 管“怎么包”，具体 Attention/FFN 管“里面算什么”。

### 3:55–7:53　用普通函数包装 Self-Attention

老师先定义一个函数：

\`\`\`python
def sublayer(x):
    return self_attn(x, x, x, mask)
\`\`\`

Self-Attention 的 Q=K=V 都使用传进来的 x。注意在 Pre-LN 外壳中，这个 x 已经是 \`norm(original_x)\`；若函数错误捕获外层旧变量，就会绕过 LayerNorm，shape 仍对但逻辑错误。

### 7:53–11:57　lambda 是同一件事的短写法

一行 lambda 可以替代上面的函数：

\`\`\`python
lambda x: self_attn(x, x, x, mask)
\`\`\`

lambda 适合这种只在调用处使用一次的简单包装，并能捕获 mask。交叉注意力还会捕获 memory，让 Query 使用 x、Key/Value 使用 memory。

老师现场让同学口述 lambda 写法，是为了练习“参数 x 来自外壳，其他依赖来自闭包”，而不是炫技。

### 11:57–15:58　同一外壳也能包 FFN

FFN 本来就只接收 x，因此可以直接传对象：

\`\`\`python
connection(x, feed_forward)
\`\`\`

也可以写 \`lambda x: feed_forward(x)\`，但没有额外参数时显得多余。注意创建 FFN 需要 d_model 与 d_ff；创建多头需要 h 与 d_model，不能把这些构造参数混在一起。

### 15:58–18:43　输出 shape 与更小的单元测试

老师打印最终 [2,4,512]，说明 norm、子层、dropout、残差整条路线接口一致。

还应使用两个确定性假子层单独验证外壳：

- \`lambda x: torch.zeros_like(x)\`，dropout=0 时输出必须等于原始 x；
- \`lambda x: x\`，可手算输出是否等于 original_x + norm(original_x)。

这些测试能把残差错误与注意力随机数分开。测试真实 Attention 时再检查 mask 和权重性质。`,

  `### 0:00–3:45　从组件进入 EncoderLayer

老师指出，现在已经有输入端、Attention、FFN、LayerNorm 和 SublayerConnection，可以开始搭一个完整编码器层。EncoderLayer 不是整个 Encoder；它是后面要重复 N 次的基本积木。

新模块导入前面各文件，体现依赖方向：EncoderLayer 使用基础组件，基础组件不依赖 EncoderLayer。

### 3:45–7:34　初始化保存两个核心组件和两个外壳

EncoderLayer 构造函数接收 d_model、self_attn、feed_forward、dropout。内部保存注意力和 FFN，并用：

\`\`\`python
self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
self.size = d_model
\`\`\`

两个 SublayerConnection 必须是独立对象，分别拥有自己的 LayerNorm 参数。size 供上层 Encoder 创建最终 LayerNorm。

### 7:34–12:24　第一个子层：源侧 Self-Attention

前向接收 x 与 src_mask：

\`\`\`python
x = self.sublayer[0](
    x,
    lambda x: self.self_attn(x, x, x, src_mask),
)
\`\`\`

Pre-LN 外壳先归一化 x，再把同一张量作为 Q、K、V。src_mask 用于遮源侧 PAD；Encoder 通常没有因果限制，可以同时看左右文。

老师现场反复区分命名：类属性可以叫 self_attn，内部实现类型是 MultiHeadedAttention；“self”描述 Q/K/V 来源，不是另一个新注意力公式。

### 12:24–14:54　第二个子层：Position-wise FFN

第一个子层输出继续进入：

\`\`\`python
x = self.sublayer[1](x, self.feed_forward)
return x
\`\`\`

FFN 不需要 mask，也不需要 lambda 捕获额外参数，可以直接作为可调用对象传入。

一个 EncoderLayer 的完整数据流是：

> x → LN → Self-Attention → Dropout → 残差 → LN → FFN → Dropout → 残差。

两个子层都保持 [B,Ls,D]。交叉注意力不属于 EncoderLayer，因为它需要目标侧 Query；它只会出现在 DecoderLayer。`,

  `### 0:00–3:49　创建 EncoderLayer 需要四样东西

老师准备测试数据 x、源 mask、多头注意力对象和 FFN 对象，再创建：

\`\`\`python
layer = EncoderLayer(
    d_model=512,
    self_attn=MultiHeadedAttention(8, 512),
    feed_forward=PositionwiseFeedForward(512, 2048),
    dropout=0.1,
)
\`\`\`

构造参数很多时可把鼠标悬停看 IDE 签名，但最终仍要理解每个参数代表什么，不能只靠自动提示填数。

### 3:49–8:41　执行前向并检查大形状

输入 [2,4,512] 经一个 EncoderLayer 后仍为 [2,4,512]。老师只打印前几个 token、前几个特征，避免控制台被 4096 个数淹没。

切片输出时要明确方括号语义，例如 \`out[0,:2,:5]\` 是第一个句子、前两个 token、每个 token 前五个特征。

### 8:41–12:34　打印模块树检查组件是否注册

正确对象树应出现：

- self_attn，其中包含四个 Linear；
- feed_forward，其中包含两层 Linear；
- sublayer，其中有两个 SublayerConnection；
- 每个外壳内的 LayerNorm 与 Dropout。

打印模型不仅是看结构。只有赋为 Module 属性、ModuleList 等已注册容器的子层，参数才会进入 \`model.parameters()\`、优化器和 state_dict。普通 Python list 可能让参数悄悄缺失。

### 12:34–15:15　用类比重新对应两条子层路线

老师回到火锅“多头分格再汇总”与 FFN“先升维再降维”的类比。一个 EncoderLayer 正是把这两种处理用两个残差外壳串起来。

测试不能只看 shape。还应确认：

- 两个 SublayerConnection 不是同一对象；
- src_mask 会让 PAD 权重为 0；
- dropout=0/eval 时输出可复现；
- backward 后各注册参数能收到梯度。

这几项能发现“结构打印正常但逻辑共享/遮罩错误”的问题。`,

  `### 0:00–3:53　Encoder 是“层的容器”

老师先回到架构：完整 Encoder 由 N 个 EncoderLayer 堆叠，原论文常用 N=6。它不重新实现 Attention，而是复用已经验证过的一层。

代码越往上越短，因为复杂度被封装在下层。读代码时要能在脑中展开：Encoder 的一行 layer(x,mask)，内部其实包含两次 LayerNorm、一次多头自注意力、一次 FFN、两条残差和 Dropout。

### 3:53–7:47　初始化克隆层并准备最终 LayerNorm

Encoder 构造函数接收 base layer 和 N：

\`\`\`python
self.layers = clones(layer, N)
self.norm = LayerNorm(layer.size)
\`\`\`

N 不写死为 6，方便构建小型测试模型或更深变体。BERT-base 等具体模型的层数也不等同于原始 Transformer 默认值，不能把“6”当作 Encoder 的定义。

### 7:47–10:47　逐层更新 x

前向：

\`\`\`python
for layer in self.layers:
    x = layer(x, mask)
return self.norm(x)
\`\`\`

第 1 层输出成为第 2 层输入，依次到第 N 层。各层共享同一份 src_mask 可见规则，但拥有独立参数。

堆叠中 shape 一直是 [B,Ls,D]；变化的是每个 token 已融合的上下文层次。最后 LayerNorm 产生 memory，供 Decoder 读取。

### 10:47–18:49　课堂创建基础层并克隆多层

老师先创建 MultiHeadedAttention 与 FFN，再组成 EncoderLayer，最后传入 Encoder(layer,N)。教学测试可令 N=3 减少输出，正式原论文配置常为 6。

最容易犯错的是 \`[layer] * N\`：它重复同一对象引用，使所有“层”共享参数。deepcopy 才得到结构相同、参数独立的多个层。

### 18:49–24:54　测试、返回值和 memory 的边界

输入 [2,4,512]，经过多层仍输出 [2,4,512]。老师打印前几个特征，并强调必须 return；Decoder 后面要接收这个结果。

Encoder 输出不是词表概率，也不是已经翻译出的句子。它是源句 memory。每个 DecoderLayer 的交叉注意力都会读取同一份 memory。

本节最后总结层级：

> 组件 → SublayerConnection → EncoderLayer（2 子层）→ Encoder（N 层）→ memory。`,

  `### 0:00–2:55　DecoderLayer 比 EncoderLayer 多一个注意力子层

老师说解码器代码更容易，是因为 clones、SublayerConnection、MultiHeadedAttention 与 FFN 都已经实现。DecoderLayer 只需按正确来源把它们连成三个子层：

1. 目标侧 masked self-attention；
2. 源—目标 cross-attention；
3. FFN。

### 2:55–7:45　初始化保存两套 Attention、一个 FFN、三个外壳

构造函数接收 d_model、self_attn、src_attn、feed_forward、dropout。self_attn 与 src_attn 结构相同但参数应独立，所以创建整层时通常对基础 attention 做 deepcopy。

\`\`\`python
self.sublayer = clones(SublayerConnection(d_model, dropout), 3)
\`\`\`

三个外壳也各有独立 LayerNorm 参数。

### 7:45–10:46　forward 的五个输入

DecoderLayer 前向接收：

- x：[B,Lt,D]，目标隐藏状态；
- memory：[B,Ls,D]，Encoder 输出；
- src_mask：源 PAD 可见性；
- tgt_mask：目标 PAD + 因果可见性。

课堂口述中 src/source、tgt/target 名称来回切换，代码里应固定命名，避免把两个 mask 传反。

### 10:46–12:54　第一子层：目标 Masked Self-Attention

\`\`\`python
x = self.sublayer[0](
    x,
    lambda x: self.self_attn(x, x, x, tgt_mask),
)
\`\`\`

Q=K=V 都来自目标 x，使用 tgt_mask 禁止读取未来与 PAD。权重最后两维是 Lt×Lt。

### 12:54–13:51　第二、三子层：Cross-Attention 与 FFN

交叉注意力：

\`\`\`python
x = self.sublayer[1](
    x,
    lambda x: self.src_attn(x, memory, memory, src_mask),
)
\`\`\`

Query 来自目标 x，Key/Value 来自源 memory，权重为 Lt×Ls。src_mask 遮源 PAD。输出位置由 Query 决定，因此仍是 [B,Lt,D]。

最后 \`self.sublayer[2](x,self.feed_forward)\` 做逐位置非线性加工。理解 DecoderLayer 时每到注意力就问：谁在提问、去哪里匹配、从哪里读值、用哪个 mask。`,

  `### 0:00–4:28　目标输入仍复用 Embedding 与位置编码

老师创建目标词表与目标 token ID，再调用前面同一个输入模块得到 [B,Lt,D]。目标端不需要重新发明 Embedding/PE，只是词表参数和输入内容不同。

接着创建 tgt_mask。真实训练中它应组合“目标不是 PAD”与 subsequent_mask；课堂小测试可先用简单 mask，但必须知道最终两种限制都需要。

### 4:28–8:25　Self-Attention 与 Source-Attention 参数要独立

老师创建一个基础 MultiHeadedAttention，再用 deepcopy 得到 self_attn 和 src_attn。它们结构相同，却承担不同关系：

- self_attn 学目标内部依赖；
- src_attn 学目标到源的对齐。

若直接让两个属性指向同一对象，会错误共享 WQ/WK/WV/WO。

### 8:25–12:25　memory 来自 Encoder 测试结果

课堂复用上一节 Encoder 输出 [B,Ls,D] 作为 memory。随后准备 src_mask、FFN、DecoderLayer。构造参数顺序很多，老师通过 IDE 提示反复核对 d_model、self_attn、src_attn、feed_forward、dropout。

这里不要把“源序列 token ID”直接当 memory；必须先经过 src_embed 与 Encoder。

### 12:25–16:22　执行整层并追踪三种权重

DecoderLayer 输出形状应为 [B,Lt,D]。内部三条路线：

1. target self-attention：[B,h,Lt,Lt]；
2. cross-attention：[B,h,Lt,Ls]；
3. FFN：保持 [B,Lt,D]。

老师在模型树中指出三个 SublayerConnection，并确认最后 shape 不变。

### 16:22–18:15　测试最好让 Ls≠Lt

课堂示例源和目标可能都恰好长度 4，这会让错误 mask 侥幸通过。更强测试应故意设置 Ls=5、Lt=3：

- tgt_mask 最后两维必须 3×3；
- src_mask 的 Key 长度必须 5；
- cross-attention 权重必须 3×5；
- 输出目标长度仍为 3。

“破坏巧合相等”是调试张量程序的重要技巧。batch、词表大小、头数测试也应避免所有数字恰好相同。`,

  `### 0:00–4:53　Decoder 与 Encoder 的容器写法几乎相同

完整 Decoder 把基础 DecoderLayer 深拷贝 N 次，再准备最终 LayerNorm：

\`\`\`python
self.layers = clones(layer, N)
self.norm = LayerNorm(layer.size)
\`\`\`

原论文常用 N=6，但构造参数不写死。代码之所以短，是因为每层三个子层已经封装完成。

### 4:53–8:36　前向让同一份 memory 进入每一层

\`\`\`python
for layer in self.layers:
    x = layer(x, memory, src_mask, tgt_mask)
return self.norm(x)
\`\`\`

x 是不断更新的目标隐藏状态；memory 是 Encoder 输出，所有 DecoderLayer 都能读取同一份源上下文。memory 不因 Decoder 加工而被改写，但不同层的 Cross-Attention 参数独立。

### 8:36–14:17　搭建测试对象并检查 [B,Lt,D]

老师复用 DecoderLayer 测试中已经创建的 self_attn、src_attn、FFN、x、memory 与两种 mask，再创建多层 Decoder。

每层输入输出保持 [2,4,512]，完整 Decoder 最终也保持这个 shape。最后输出不是词概率，只是目标侧上下文隐藏表示，仍需 Generator 把 D 映射到目标词表。

课堂从复制测试代码时出现函数名/变量名未改全的风险，提醒我们复制粘贴后应逐项核对：测试的是 DecoderLayer 还是 Decoder、调用对象和打印标签是否一致。

### 14:17–20:47　三种注意力在整图中对应哪里

老师回到架构图逐一对应：

- Encoder 每层的注意力：源侧 Self-Attention；
- Decoder 每层下方：Masked Target Self-Attention；
- Decoder 每层中间：Source–Target Cross-Attention。

前两者 Q、K、V 都来自同一侧，所以叫 self；交叉注意力的 Q 来自目标，K/V 来自 memory。

### 20:47–28:12　为什么所有 DecoderLayer 都读 Encoder 输出

老师用 7 个英语词翻译为 9 个法语词举例。生成法语第 3 个位置时，Decoder 要判断七个源词中哪些与当前目标最相关；Cross-Attention 权重因此是 Lt×Ls。

每一层都可在自己当前的抽象层次重新对齐源句，而不是只有第一层读一次 memory 后永久丢掉。完整主线：

> 目标前缀逐层更新；每层先看目标过去，再查询同一源 memory，再做 FFN；最终仍保留目标长度。`,

  `### 0:00–3:51　Decoder 输出还不是“词”

老师从 [2,4,512] 解释 Generator 的任务：两个句子、每个目标句四个位置、每个位置是一条 512 维隐藏向量。模型还不知道该选词表中的哪个 token。

若目标词表有 1000 个 token，每个位置都要产生 1000 个候选分数，所以形状应变为 [2,4,1000]。

### 3:51–6:46　Generator 只需一个 Linear

类初始化接收 d_model 与 vocab：

\`\`\`python
self.proj = nn.Linear(d_model, vocab)
\`\`\`

Linear 独立处理每个 B、L 位置，把最后一维 512 映射到 1000。若源、目标词表不同，这里的 vocab 必须是目标词表大小。

### 6:46–9:44　为什么使用 log_softmax

前向：

\`\`\`python
return F.log_softmax(self.proj(x), dim=-1)
\`\`\`

Linear 输出 logits，不是概率；log_softmax 沿最后的词表维归一化并返回对数概率。对数空间计算数值更稳定，也可直接与 NLLLoss 配合。

\`log_probs.exp()\` 才是普通概率，其最后一维和应为 1。不能说“对数概率本身相加等于 1”。

### 9:44–10:40　模块导入与职责边界

老师讨论把多个组件集中 re-export，减少后续导包长度。无论文件如何组织，Generator 的职责都应保持单一：

> 隐藏状态 D → 目标词表分布 Vt。

它不直接返回 token ID，也不完成贪心、采样或 beam search。训练时它提供所有位置分布；真实推理时解码算法会根据最后一个位置分布选下一个 token，再把新前缀送回 Decoder。`,

  `### 0:00–3:55　先保证 Decoder 测试函数真的 return

老师复用上一节 Decoder 输出，现场提醒：测试辅助函数若只 print 不 return，Generator 就会收到 None。把输出显式 return 后，才能连接下一组件。

随后创建 Generator(d_model=512,vocab=1000)，把 [2,4,512] 传入。

### 3:55–6:42　第一项检查：shape 是 [2,4,1000]

输出每个位置有 1000 个候选，因此 shape 为 [2,4,1000]。老师再次用“两个句子、每句四个位置、每位置 1000 个词概率”解释三维。

打印完整张量会非常长，可只查看 \`out[0,0,:10]\` 与 shape。

### 6:42–10:58　第二项检查：把对数概率还原

老师选第一个样本的第一个词位置 \`out[0,0]\`，得到 1000 个 log probability。调用 \`.exp()\` 还原普通概率，再求和：

\`\`\`python
probs = log_probs.exp()
total = probs[0, 0].sum()
\`\`\`

total 应接近 1。若直接对 log_probs 求和，没有概率意义。

### 10:58–12:43　浮点结果要用近似比较

控制台可能显示 1.0000，也可能内部为 0.9999999。测试应：

\`\`\`python
assert torch.allclose(
    probs.sum(-1),
    torch.ones_like(probs.sum(-1)),
    atol=1e-6,
)
\`\`\`

这还能发现 softmax 维度写错：沿序列维归一化时 shape 仍正确，但每个位置的词表概率和不为 1。

### 12:43–18:45　把完整组件链再口述一遍

老师回到图中从 [B,L] token ID 开始：

> Embedding+PE → Encoder/Decoder 得 [B,Lt,D] → Linear 得 [B,Lt,Vt] → log_softmax 得词表对数概率。

真正选词可取最大概率，但训练通常同时计算所有目标位置的损失。Generator 测试跑通说明输出头正确，不代表整模已经学会翻译；还没有训练数据、loss、优化器和自回归生成循环。`,

  `### 0:00–2:55　所有零件完成后，整体类只负责连接

老师确认输入、Encoder、Decoder 与 Generator 都已实现，开始定义 EncoderDecoder 顶层类。它对应机器翻译 Seq2Seq 架构：源序列经编码，目标前缀结合源 memory 解码，再映射到目标词表。

### 2:55–7:46　构造函数保存五个顶层模块

顶层对象接收：

- encoder；
- decoder；
- src_embed；
- tgt_embed；
- generator。

Embedding 要分源侧和目标侧，因为两个词表可能大小不同。Generator 使用目标 vocab。

### 7:46–12:49　forward 的输入与两种 mask

完整前向通常接收 src、tgt、src_mask、tgt_mask。课堂还提到 padding mask：PAD 只是补齐长度，不应被模型当成有语义词。

src_mask 会用于 Encoder 自注意力和 Decoder Cross-Attention；tgt_mask 会用于 Decoder 目标自注意力，组合 PAD 与未来遮盖。

### 12:49–15:53　主数据流

逻辑可写为：

\`\`\`python
memory = self.encode(src, src_mask)
hidden = self.decode(memory, src_mask, tgt, tgt_mask)
return self.generator(hidden)
\`\`\`

课堂累计实现有时让 forward 只返回 Decoder hidden，再由训练代码单独调用 generator；两种 API 都可以，但项目内部必须一致。配套实现选择明确返回词表 log_probs，避免“顶层 forward 是否含 Generator”含糊。

### 15:53–18:05　为什么还要拆 encode 与 decode

encode 封装 src_embed→encoder；decode 封装 tgt_embed→decoder。拆开后，推理时源句只编码一次，随后每生成一个新 token 反复 decode，不必重复计算 memory。

读名称时区分：

- encoder/decoder：nn.Module 组件；
- encode/decode：调用这些组件的方法。

顶层组装的意义不是加入新数学公式，而是把数据与 mask 按正确顺序送到已验证组件。`,

  `### 0:00–3:52　encode：源 ID 变成 memory

老师把 Encoder 路线单独实现：

\`\`\`python
def encode(self, src, src_mask):
    return self.encoder(self.src_embed(src), src_mask)
\`\`\`

src=[B,Ls]，src_embed 先变 [B,Ls,D]，Encoder 多层处理后仍是 [B,Ls,D]，这就是 memory。

src_mask 主要遮源 PAD。它不是为了防止源词看未来；普通 Encoder 通常双向读取完整非 PAD 源句。

### 3:52–7:40　decode：目标前缀查询 memory

\`\`\`python
def decode(self, memory, src_mask, tgt, tgt_mask):
    return self.decoder(
        self.tgt_embed(tgt),
        memory,
        src_mask,
        tgt_mask,
    )
\`\`\`

tgt=[B,Lt] 经目标输入端变 [B,Lt,D]；Decoder 输出仍保留 Lt。

src_mask 进入 Cross-Attention，控制可读源位置；tgt_mask 进入 Target Self-Attention，控制 PAD 与未来位置。

### 7:40–8:42　每个长度在哪保留

分开接口后，形状路线清楚：

\`\`\`text
src [B,Ls] → memory [B,Ls,D]
tgt [B,Lt] + memory → hidden [B,Lt,D]
hidden → Generator → log_probs [B,Lt,Vt]
\`\`\`

Cross-Attention 是唯一同时接触 Lt 和 Ls 的核心位置，权重 [B,h,Lt,Ls]。Generator 只替换最后一维 D→Vt，不改变 Lt。

这两个方法也为自回归推理做准备：memory 缓存一次，tgt 前缀逐步增长，每步只重复目标侧 decode 与 Generator。`,

  `### 0:00–3:47　make_model 是“装配工厂”

老师把整个 Transformer 类比为科研/工程团队自己搭完整架构。前面已经分别制造零件，make_model 负责按照配置创建并连接，不再加入新公式。

函数接收源/目标词表大小、N、d_model、d_ff、h、dropout 等。默认配置可接近论文，测试时应允许缩小。

### 3:47–7:37　先创建可复用的基础组件

\`\`\`python
attn = MultiHeadedAttention(h, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
position = PositionalEncoding(d_model, dropout)
\`\`\`

接着 EncoderLayer 使用 attn、ff；DecoderLayer 需要两套独立 attention，因此通过 deepcopy 复制。Encoder/Decoder 再各堆 N 层。

深拷贝的原则贯穿全章：结构模板可相同，学习参数必须独立。

### 7:37–12:07　源侧与目标侧输入要分别建立

源输入与目标输入都可用 Sequential：

\`\`\`python
src_embed = nn.Sequential(
    Embeddings(src_vocab, d_model),
    copy.deepcopy(position),
)
tgt_embed = nn.Sequential(
    Embeddings(tgt_vocab, d_model),
    copy.deepcopy(position),
)
\`\`\`

两侧词表可能不同，Embedding 绝不能无条件共享。位置编码可以按相同固定公式构造，但各 Sequential 内保留自己的模块对象更清楚。

### 12:07–16:56　三类注意力如何装入整模

老师再次口述：

- Encoder self-attn：Q=K=V=源表示；
- Decoder self-attn：Q=K=V=目标表示，带 tgt_mask；
- Cross-attn：Q=目标，K=V=源 memory，带 src_mask。

这些都使用同一个 MultiHeadedAttention 类，区别由输入来源与 mask 决定。

### 16:56–19:49　完成 Decoder 与 Generator

Decoder 由 DecoderLayer×N 与最终 LayerNorm 组成；Generator 接收 d_model 与 tgt_vocab。

完整顶层包含 src_embed、encoder、tgt_embed、decoder、generator。随后可遍历参数，对维度>1 的权重应用 Xavier 初始化。初始化改善训练起点，但不会修复错误连线。

本节真正检查点是：两侧 vocab 是否用对、注意力对象是否独立、N 是否正确、所有子模块是否注册。`,

  `### 0:00–3:54　Sequential 把 Embedding 与 PE 串成输入端

老师继续完成 make_model 中源/目标输入处理。nn.Sequential 保证 token ID 先经过 Embeddings，再经过 PositionalEncoding。只需传入一个 x，内部按顺序调用。

目标输入也必须使用 tgt_vocab；Generator 同样使用 tgt_vocab。位置编码对象可 deepcopy，避免模块层级引用混乱。

### 3:54–6:51　打印模型树先看五个顶层名字

整棵树非常长，老师建议先忽略内部 Linear/Dropout，找到：

1. src_embed；
2. encoder；
3. tgt_embed；
4. decoder；
5. generator。

再按 forward 把它们连成主线。若一开始钻进第四个 Linear 的细节，很容易失去整体方向。

### 6:51–9:49　展开 Encoder 层级

Encoder 下面有 N 个 EncoderLayer；每层两个 SublayerConnection：

- Multi-Head Self-Attention；
- Position-wise FFN。

每个外壳中还能看到 LayerNorm 与 Dropout；Attention 中有四个 Linear，FFN 中有两个。模型树缩进正对应组合关系。

### 9:49–13:06　展开 Decoder 与输出

目标输入仍是 Embedding+PE。每个 DecoderLayer 有三个子层：

1. masked self-attention，防止偷看未来；
2. cross-attention，读取 Encoder memory；
3. FFN。

Decoder 顶部再 LayerNorm，Generator 将 D 映射到 Vt 并 log_softmax。

### 三条复习线

老师整节实际上给出三种读法：

- **模块线**：输入→Encoder→Decoder→Generator；
- **形状线**：[B,L]→[B,L,D]→拆出 h/d_k→[B,Lt,D]→[B,Lt,Vt]；
- **QKV 线**：源自注意力同源、目标自注意力同源、交叉注意力 Q 与 K/V 异源。

能同时沿三条线解释模型树，才算真正看懂“组合完整模型”。`,

  `### 0:00–3:56　准备整模、源目标 ID 与 mask

老师最后创建完整 Transformer，并准备 src、tgt、src_mask、tgt_mask。ID 必须在各自词表范围内；src 与 tgt 可以有不同长度。

tgt 训练输入应是右移后的目标前缀，tgt_mask 组合因果下三角与 PAD 可见性。课堂为了跑通可能使用简化 mask，学习笔记需要记住真实训练不能省略这些语义。

### 3:56–6:54　端到端前向第一次暴露连接错误

执行前向后，控制台出现维度/调用问题。老师没有重写整个模型，而是沿 traceback 回到 EncoderDecoder.forward、encode 与 encoder 对象的调用处。

整模测试的价值正是覆盖局部测试没覆盖的参数顺序、方法返回与组件连接。

### 6:54–10:49　课堂现场修复：调用错对象/名称多写字符

老师逐层定位到 forward 中调用编码结果的位置，发现名称或方法写错（音轨中可听到多写一个字母、把 encoder/encode 混淆的排查）。修复后重新运行。

这与前面提醒呼应：

- encoder 是模块；
- encode 是方法；
- 辅助函数必须 return；
- 复制代码后要检查变量名。

看 traceback 时从最底部异常消息向上找自己项目的第一行，而不是被长模型树吓住。

### 10:49–12:48　跑通后按层级总结

老师回顾 N 个 EncoderLayer、N 个 DecoderLayer 与最终全局 LayerNorm。还强调完整代码是自己按论文组件搭出来的教育实现，能帮助理解底层，但生产库可能采用更高效内核、不同归一化顺序和缓存。

### 端到端验收应做什么

完整输出应为 [B,Lt,Vt]。除“不报错”外，还应：

- \`exp(log_probs).sum(-1)\` 接近 1；
- 不同 Ls/Lt 能正常广播；
- mask 区注意力为 0；
- eval/no_grad 下重复输出一致；
- backward 能让主要参数获得梯度。

整模前向跑通只是架构实现终点，不是翻译训练终点。真正系统还需要数据批处理、目标右移、PAD loss 忽略、优化器与调度、保存加载，以及贪心/采样/beam search 自回归生成。`,
];

if (articles.length !== lessons.length) {
  throw new Error(`expected ${lessons.length} articles, received ${articles.length}`);
}

function noteFile(index, lesson) {
  return `${String(index).padStart(2, "0")}-${lesson.slug}.md`;
}

function mermaidLabel(value) {
  return String(value).replaceAll('"', "'").replaceAll("\n", "<br/>");
}

function makeFlowchart(chain) {
  const nodes = chain.map(
    (item, index) => `    N${index}["${mermaidLabel(item)}"]`,
  );
  const edges = chain.slice(0, -1).map(
    (_, index) => `    N${index} --> N${index + 1}`,
  );
  return `\`\`\`mermaid
flowchart LR
${nodes.join("\n")}
${edges.join("\n")}
\`\`\``;
}

function makeArchitectureVisual(index) {
  if (index === 1) {
    return `### 模型发展关系图

\`\`\`mermaid
flowchart LR
    A["RNN / LSTM / GRU<br/>按时间步递归"] --> B["Attention<br/>按任务分配权重"]
    B --> C["Transformer<br/>注意力成为主干"]
    C --> D["BERT<br/>偏 Encoder 理解路线"]
    C --> E["GPT<br/>偏 Decoder 生成路线"]
\`\`\`

### 本章四大组件图

\`\`\`mermaid
flowchart LR
    I["输入部分<br/>文字 → 数字向量"] --> E["Encoder<br/>理解源句"]
    E --> D["Decoder<br/>结合前缀生成"]
    D --> O["输出部分<br/>目标词表概率"]
\`\`\``;
  }
  if ([2, 3, 4, 26, 29, 34, 35, 37].includes(index)) {
    return `### 组件层级图

\`\`\`mermaid
classDiagram
    class Transformer {
      +encode(src, src_mask)
      +decode(tgt, memory, masks)
      +forward(src, tgt, masks)
    }
    class Encoder {
      +EncoderLayer × N
      +LayerNorm
    }
    class Decoder {
      +DecoderLayer × N
      +LayerNorm
    }
    class Attention {
      +Q
      +K
      +V
      +mask
    }
    class Generator {
      +Linear
      +log_softmax
    }
    Transformer *-- Encoder
    Transformer *-- Decoder
    Transformer *-- Generator
    Encoder o-- Attention
    Decoder o-- Attention
\`\`\``;
  }
  if ([15, 16, 17, 18, 19].includes(index)) {
    return `### 注意力张量时序图

\`\`\`mermaid
sequenceDiagram
    participant X as 输入 [B,L,D]
    participant P as Q/K/V 投影
    participant H as 多个注意力头
    participant M as 合并与输出投影
    X->>P: 生成 Q、K、V
    P->>H: reshape 为 [B,h,L,d_k]
    H->>H: scores → mask → softmax → V
    H->>M: 拼回 [B,L,D]
    M-->>X: 输出同形状表示
\`\`\``;
  }
  return "";
}

function makeNote(lesson, index) {
  const previous = index > 1 ? lessons[index - 2] : null;
  const next = index < lessons.length ? lessons[index] : null;
  const nav = [
    previous ? `[← 上一节：${index - 1} ${previous.title}](./${noteFile(index - 1, previous)})` : "← 已是第一节",
    "[返回总目录](./README.md)",
    next ? `[下一节：${index + 1} ${next.title} →](./${noteFile(index + 1, next)})` : "已是最后一节 →",
  ].join(" · ");
  return `# 第 ${index} 节：${lesson.title}

> 笔记编号 ${index}/38 · 对应原视频 P${lesson.p} · [打开这一集](https://www.bilibili.com/video/BV14mdfBDE4Q?p=${lesson.p})

${nav}

## 这节解决什么问题

${lesson.plain}

![第 ${index} 节原创概念图](./diagrams/${String(index).padStart(2, "0")}-concept.svg)

图要沿箭头或结构层级阅读。先说清楚数据从哪里来、形状怎样变化，再记组件名称。

## 老师原声整理稿（按讲解顺序）

${articles[index - 1]}

## 辅助流程图

${makeFlowchart(lesson.chain)}

${makeArchitectureVisual(index)}

## 完整原声逐段记录

[查看本节按时间戳整理的完整音轨转写](./transcripts/p${lesson.p}.md)

这份逐段记录用于核查老师讲过的内容是否遗漏；学习时优先阅读上面的校正文章，遇到想追溯的细节再按时间戳查看原声记录。

## 零基础先记住

${lesson.points.map(x => `- ${x}`).join("\n")}

## 最小可运行代码

下面代码默认从项目根目录运行。涉及模型组件时，使用 [transformer_from_scratch](../../transformer_from_scratch/README.md) 中经过测试的 PyTorch 实现。

\`\`\`python
${lesson.code}
\`\`\`

### 输入和输出怎么看

${lesson.output}

## 最容易踩的坑

${lesson.pit}

## 本节知识链

\`${lesson.chain.join(" → ")}\`

Transformer 学习的主线始终是形状。每经过一个箭头，都问自己：batch、序列长度、特征维、头数和词表维中的哪一个发生了变化？

## 自测

**问题：${lesson.quiz}**

<details>
<summary>点开核对答案</summary>

${lesson.answer}

</details>

## 学完检查

- [ ] 我能不用术语解释本节组件解决的问题
- [ ] 我能在运行前写出关键张量形状
- [ ] 我能指出 Q、K、V 或 mask 的来源
- [ ] 我知道代码“形状正确但逻辑可能错误”的情况
- [ ] 我能独立回答自测题

${nav}
`;
}

lessons.forEach((lesson, i) => {
  fs.writeFileSync(path.join(outDir, noteFile(i + 1, lesson)), makeNote(lesson, i + 1));
});

const stages = [
  ["一、认识整体架构", 1, 5],
  ["二、输入与位置编码", 6, 10],
  ["三、掩码与缩放点积注意力", 11, 15],
  ["四、多头注意力与前馈网络", 16, 20],
  ["五、LayerNorm 与残差子层", 21, 25],
  ["六、Encoder 与 Decoder", 26, 31],
  ["七、输出与完整模型", 32, 38],
];

const stageMarkdown = stages.map(([name, start, end]) => {
  const links = lessons.slice(start - 1, end).map((lesson, offset) => {
    const index = start + offset;
    return `${index}. [${lesson.title}](./${noteFile(index, lesson)})（原视频 P${lesson.p}）`;
  }).join("\n");
  return `## ${name}\n\n${links}`;
}).join("\n\n");

const readme = `# Transformer：38 节零基础学习笔记

这套笔记覆盖原视频 **P106–P143**，重新编号为 **第 1–38 节**。每一节采用与文本预处理笔记相同的结构：问题目标、零基础解释、原创 SVG、最小可运行代码、输入输出、易错点、知识链、自测答案和前后导航。

图片只承载知识内容，不包含视频分集、课程来源或平台标识。

## 建议学习方法

1. 先写出输入形状，不看答案预测输出形状。
2. 对注意力始终标出 Q、K、V 分别来自哪里。
3. 对 mask 始终标出 True/False 的语义和可广播形状。
4. 运行最小代码，再到配套实现中定位对应类或函数。
5. 一阶段学完后，从头口述一次完整数据流。

${stageMarkdown}

## 一张形状主线

\`\`\`text
src/tgt ids                 [B, L]
Embedding + Position        [B, L, D]
split heads                 [B, h, L, d_k]
attention scores            [B, h, Lq, Lk]
merge heads                 [B, Lq, D]
Encoder memory              [B, Ls, D]
Decoder hidden              [B, Lt, D]
Generator log probabilities [B, Lt, Vt]
\`\`\`

## 三组必须会写的 Q/K/V

| 位置 | Q | K | V | 主要 mask |
|---|---|---|---|---|
| Encoder 自注意力 | 源表示 | 源表示 | 源表示 | 源 PAD |
| Decoder 自注意力 | 目标表示 | 目标表示 | 目标表示 | 目标 PAD + 因果 |
| Decoder 交叉注意力 | 目标表示 | memory | memory | 源 PAD |

## 配套代码

- [完整实现](../../transformer_from_scratch/model.py)
- [自动测试](../../transformer_from_scratch/test_model.py)
- [代码说明](../../transformer_from_scratch/README.md)
- 运行：\`python3 -m unittest transformer_from_scratch.test_model -v\`

## 已校准的关键点

- token ID 必须严格小于对应词表大小。
- LayerNorm 方差使用 \`unbiased=False\` 才与标准实现一致。
- 本项目明确采用 Pre-LN：\`x + Dropout(Sublayer(LayerNorm(x)))\`。
- 多头 mask 的第一维是 batch 或 1，不是头数。
- 源/目标 Embedding 不能在词表不同时盲目共享。
- Cross-Attention 的 Q 来自目标侧，K/V 来自 Encoder memory。
`;

fs.writeFileSync(path.join(outDir, "README.md"), readme);
console.log(`generated ${lessons.length} Transformer notes in the unified beginner-friendly format`);

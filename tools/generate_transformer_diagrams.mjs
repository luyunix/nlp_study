import fs from "node:fs";
import path from "node:path";

const root = "/Users/lyn/Documents/nlp_study";
const notesDir = path.join(root, "notes", "transformer");
const outputDir = path.join(notesDir, "diagrams");

const lessons = [
  { n: 106, title: "Transformer 从哪里来", type: "flow", items: ["RNN\n按时间步递归", "Attention\n直接寻找关联", "Transformer\n完全基于注意力", "BERT / GPT\n规模化预训练"], note: "关键变化：序列建模从“逐步传递”转向“全局交互”。" },
  { n: 107, title: "Transformer 总体架构", type: "arch", items: ["源序列", "Embedding + 位置编码", "Encoder × N", "Decoder × N", "Linear + Softmax", "目标序列"], note: "Encoder 理解输入，Decoder 在已生成前缀与编码结果的共同条件下预测下一个词。" },
  { n: 108, title: "Encoder 上半部分", type: "stack", items: ["输入词元", "词嵌入 Embedding", "位置编码 Positional Encoding", "Multi-Head Self-Attention", "Add & Norm", "前馈网络 FFN"], note: "Self-Attention 中 Q、K、V 都来自同一份 Encoder 表示。" },
  { n: 109, title: "Decoder 与完整连接", type: "arch", items: ["源序列", "Encoder × N", "memory", "目标前缀 + mask", "Cross-Attention", "FFN → 下一个词"], note: "因果掩码阻止偷看未来；Cross-Attention 让 Decoder 查询源序列。" },
  { n: 110, title: "从零实现学习路线", type: "flow", items: ["输入组件", "注意力与 FFN", "Encoder / Decoder Layer", "完整 Encoder / Decoder", "Generator", "整模测试"], note: "实现顺序遵循“组件 → 子层 → 层 → 堆叠 → 完整模型”。" },
  { n: 111, title: "Token Embedding", type: "code", items: ["token id\n[B, L]", "查表 nn.Embedding", "乘 √d_model", "词向量\n[B, L, D]"], note: "合法 token id 必须小于 vocab_size；缩放用于平衡后续位置编码的量级。" },
  { n: 112, title: "正弦位置编码原理", type: "wave", items: ["pos：词的位置", "2i / 2i+1：维度", "偶数维使用 sin", "奇数维使用 cos"], note: "不同频率的正弦/余弦让模型既能分辨绝对位置，也容易表达相对位移。" },
  { n: 113, title: "位置编码的价值", type: "compare", items: ["没有位置编码", "词序被打乱仍近似相同", "注意力本身不含先后顺序", "加入位置编码", "每个位置拥有独特信号", "保留并行计算能力"], note: "内容向量负责“是什么”，位置向量负责“在哪里”。" },
  { n: 114, title: "PositionalEncoding 代码", type: "code", items: ["构造 position", "计算 div_term", "偶数列填 sin", "奇数列填 cos", "register_buffer(pe)", "x + pe[:L] → dropout"], note: "pe 不是可训练参数，但要跟随模型迁移设备并写入 state_dict。" },
  { n: 115, title: "位置编码形状测试", type: "flow", items: ["输入 x\n[B,L,D]", "截取 pe[:L]\n[1,L,D]", "广播相加", "Dropout", "输出\n[B,L,D]"], note: "测试重点是长度切片、batch 广播、设备一致性与输出形状不变。" },
  { n: 116, title: "上三角矩阵", type: "matrix-upper", items: ["对角线", "未来位置", "triu(k=1)", "上三角区域"], note: "在因果注意力中，对角线上方表示当前位置不应访问的未来词。" },
  { n: 117, title: "可见区域：下三角", type: "matrix-lower", items: ["过去位置", "当前位置", "逻辑取反", "允许注意"], note: "第 i 行只允许查看第 0…i 列，形成自回归的信息边界。" },
  { n: 118, title: "因果掩码可视化", type: "matrix-lower", items: ["行：Query 位置", "列：Key 位置", "亮色：可见", "暗色：屏蔽"], note: "热力图的下三角结构直观对应“只能看自己和过去”。" },
  { n: 119, title: "masked_fill 如何工作", type: "mask", items: ["注意力分数 scores", "mask == 0", "填入 −∞", "Softmax", "被屏蔽位置概率 = 0"], note: "必须在 Softmax 之前屏蔽；使用极小值使对应指数项趋近于零。" },
  { n: 120, title: "Scaled Dot-Product Attention", type: "attention", items: ["Q × Kᵀ", "÷ √dₖ", "应用 mask", "Softmax", "Dropout", "权重 × V"], note: "缩放避免维度变大时点积过大、Softmax 过早饱和。" },
  { n: 121, title: "多头注意力：拆分", type: "heads", items: ["线性投影 Q/K/V", "拆成 h 个头", "每头维度 dₖ=D/h", "并行关注不同关系"], note: "拆头不是复制输入，而是让不同投影子空间学习不同的关联模式。" },
  { n: 122, title: "多头注意力：五步", type: "flow", items: ["Q/K/V 投影", "reshape + transpose", "每个头做 Attention", "transpose + concat", "最终线性投影"], note: "拆头与合头只改变视图和轴顺序，最终维度仍回到 d_model。" },
  { n: 123, title: "MultiHeadedAttention 代码流", type: "code", items: ["[B,L,D]", "[B,L,h,dₖ]", "[B,h,L,dₖ]", "Attention", "[B,L,h·dₖ]", "Linear → [B,L,D]"], note: "mask 通常扩成 [B,1,Lq,Lk]，由广播机制共享给所有头。" },
  { n: 124, title: "多头注意力测试", type: "test", items: ["检查 D % h == 0", "检查 mask 可广播", "检查输出 [B,L,D]", "检查注意力权重和为 1"], note: "用非全零、非全一的 mask 才能真正检验屏蔽逻辑。" },
  { n: 125, title: "Position-wise FFN", type: "flow", items: ["每个位置向量 D", "Linear D→d_ff", "ReLU / GELU", "Dropout", "Linear d_ff→D"], note: "同一组参数独立作用于每个 token；它混合特征维，不混合序列位置。" },
  { n: 126, title: "Layer Normalization", type: "code", items: ["输入 x", "按最后一维求 mean", "按最后一维求 var", "(x−mean)/√(var+ε)", "乘 α 加 β"], note: "实现时使用 unbiased=False，才与深度学习框架中的 LayerNorm 方差定义一致。" },
  { n: 127, title: "LayerNorm 测试", type: "test", items: ["输入 [B,L,D]", "最后一维均值 ≈ 0", "最后一维方差 ≈ 1", "形状保持不变"], note: "测试统计量时应允许浮点误差，并确认归一化轴不是 batch 轴。" },
  { n: 128, title: "BatchNorm 与 LayerNorm", type: "compare", items: ["BatchNorm", "跨 batch 统计同一特征", "依赖批大小与训练/推理状态", "LayerNorm", "单个 token 内跨特征统计", "适合变长序列与小批量"], note: "Transformer 通常使用 LayerNorm，因为它不依赖其他样本。" },
  { n: 129, title: "SublayerConnection", type: "compare", items: ["Post-LN", "LayerNorm(x + Dropout(Sublayer(x)))", "原论文常见写法", "Pre-LN", "x + Dropout(Sublayer(LayerNorm(x)))", "深层训练通常更稳定"], note: "本项目完整实现采用标准 Pre-LN；残差支路必须保留原始 x。" },
  { n: 130, title: "子层连接测试", type: "test", items: ["准备 Attention/FFN 子层", "用 lambda 传入额外参数", "验证残差形状一致", "验证 Dropout 与 LayerNorm 顺序"], note: "SublayerConnection 只负责包装，具体子层通过可调用对象注入。" },
  { n: 131, title: "EncoderLayer", type: "stack", items: ["输入 x", "Self-Attention 子层", "残差 + LayerNorm", "Position-wise FFN", "残差 + LayerNorm", "输出 x′"], note: "一个 EncoderLayer 有两个子层；Self-Attention 的 Q=K=V=x。" },
  { n: 132, title: "EncoderLayer 对象树", type: "hierarchy", items: ["EncoderLayer", "self_attn", "feed_forward", "sublayer[0]", "sublayer[1]", "size = d_model"], note: "模块树检查能发现层是否被正确注册、复制和参与参数更新。" },
  { n: 133, title: "Encoder 堆叠", type: "stack", items: ["Embedding + PE", "EncoderLayer 1", "EncoderLayer 2", "…", "EncoderLayer N", "最终 LayerNorm → memory"], note: "每层结构相同但参数独立；memory 会被所有 DecoderLayer 复用。" },
  { n: 134, title: "DecoderLayer 三个子层", type: "stack", items: ["目标表示 x", "Masked Self-Attention", "Cross-Attention：Q=x，K=V=memory", "Position-wise FFN", "残差与规范化", "输出 x′"], note: "Decoder 比 Encoder 多一个跨注意力子层，用于读取源序列信息。" },
  { n: 135, title: "DecoderLayer 测试", type: "test", items: ["x [B,Lt,D]", "memory [B,Ls,D]", "tgt_mask [B,1,Lt,Lt]", "src_mask [B,1,1,Ls]", "输出 [B,Lt,D]"], note: "测试应让源长度与目标长度不同，以暴露错误的维度假设。" },
  { n: 136, title: "Decoder 堆叠与共享 memory", type: "arch", items: ["源序列", "Encoder × N", "memory", "DecoderLayer 1", "DecoderLayer 2", "DecoderLayer N"], note: "每个 DecoderLayer 都读取同一份 memory；变化的是目标侧隐藏状态 x。" },
  { n: 137, title: "Generator", type: "flow", items: ["Decoder 输出\n[B,L,D]", "Linear\nD→V", "log_softmax", "词表对数概率\n[B,L,V]"], note: "Generator 把隐藏特征映射到目标词表；V 应等于目标词表大小。" },
  { n: 138, title: "Generator 测试", type: "test", items: ["输入 [2,4,512]", "Linear 到 vocab", "输出 [2,4,V]", "exp(log_probs) 沿 V 求和 ≈ 1"], note: "不要只检查最后一维大小，还应验证概率归一化性质。" },
  { n: 139, title: "EncoderDecoder.forward", type: "flow", items: ["src", "encode(src, src_mask)", "得到 memory", "decode(memory, src_mask, tgt, tgt_mask)", "Decoder 隐藏状态"], note: "forward 负责组织编码和解码；词表概率通常由 Generator 另行计算。" },
  { n: 140, title: "encode / decode 接口", type: "compare", items: ["encode", "src_embed → encoder", "输出 memory", "decode", "tgt_embed → decoder", "接收 memory 与两种 mask"], note: "区分 encoder 模块与 encode 方法，可避免命名混淆。" },
  { n: 141, title: "完整模型组装", type: "hierarchy", items: ["make_model", "Embeddings + PE", "Encoder × N", "Decoder × N", "Generator", "Xavier 参数初始化"], note: "源词表与目标词表可能不同，因此两侧 Embedding 不应盲目共享。" },
  { n: 142, title: "组件总复习", type: "arch", items: ["src_embed", "encoder", "memory", "tgt_embed", "decoder", "generator"], note: "整模调试时沿数据流逐段检查张量形状，比只看最终报错更有效。" },
  { n: 143, title: "完整模型端到端测试", type: "test", items: ["src / tgt token id 合法", "构造 src_mask / tgt_mask", "model(src,tgt,…)", "generator(hidden)", "输出 [B,Lt,Vt]"], note: "最终测试同时覆盖词表范围、掩码广播、编码解码连接和概率输出。" },
];

function esc(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function lines(text, x, y, options = {}) {
  const { size = 24, color = "#dbeafe", weight = 600, anchor = "middle", gap = 32 } = options;
  return String(text)
    .split("\n")
    .map((line, index) => `<text x="${x}" y="${y + index * gap}" text-anchor="${anchor}" font-size="${size}" font-weight="${weight}" fill="${color}">${esc(line)}</text>`)
    .join("\n");
}

function wrapText(text, maxUnits) {
  const widthOf = (token) => [...token].reduce((sum, ch) => {
    if (/\s/.test(ch)) return sum + 0.35;
    return sum + (/[\u2e80-\u9fff\uff00-\uffef]/.test(ch) ? 1 : 0.56);
  }, 0);

  return String(text).split("\n").flatMap((paragraph) => {
    const tokens = paragraph.match(/[\u2e80-\u9fff\uff00-\uffef]|[^\u2e80-\u9fff\uff00-\uffef\s]+|\s+/g) ?? [paragraph];
    const result = [];
    let current = "";
    let units = 0;
    for (const token of tokens) {
      const tokenUnits = widthOf(token);
      if (current.trim() && units + tokenUnits > maxUnits) {
        result.push(current.trim());
        current = token.trimStart();
        units = widthOf(current);
      } else {
        current += token;
        units += tokenUnits;
      }
    }
    if (current.trim() || result.length === 0) result.push(current.trim());
    return result;
  });
}

function card(x, y, w, h, text, accent = "#38bdf8") {
  const wrapped = wrapText(text, Math.max(7, w / 21));
  const gap = 29;
  const textY = y + h / 2 - ((wrapped.length - 1) * gap) / 2 + 8;
  return `
    <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="22" fill="#111c33" stroke="${accent}" stroke-width="2"/>
    <rect x="${x}" y="${y}" width="8" height="${h}" rx="4" fill="${accent}"/>
    ${lines(wrapped.join("\n"), x + w / 2 + 4, textY, { size: 21, gap })}
  `;
}

function arrow(x1, y1, x2, y2, color = "#64748b") {
  return `<path d="M ${x1} ${y1} L ${x2} ${y2}" stroke="${color}" stroke-width="4" fill="none" marker-end="url(#arrow)"/>`;
}

function flow(items) {
  const count = items.length;
  const gap = 26;
  const w = Math.min(230, (1080 - gap * (count - 1)) / count);
  const total = w * count + gap * (count - 1);
  const start = (1200 - total) / 2;
  let out = "";
  items.forEach((item, i) => {
    const x = start + i * (w + gap);
    out += card(x, 270, w, 150, item, ["#38bdf8", "#818cf8", "#c084fc", "#fb7185", "#fbbf24", "#34d399"][i % 6]);
    if (i < count - 1) out += arrow(x + w + 4, 345, x + w + gap - 6, 345);
  });
  return out;
}

function stack(items) {
  const visible = items.slice(0, 6);
  let out = "";
  visible.forEach((item, i) => {
    const y = 170 + i * 63;
    const x = 260 + i * 34;
    const w = 680 - i * 68;
    out += `<rect x="${x}" y="${y}" width="${w}" height="54" rx="16" fill="${i % 2 ? "#172554" : "#102a43"}" stroke="${["#38bdf8", "#818cf8", "#c084fc", "#fb7185", "#fbbf24", "#34d399"][i]}" stroke-width="2"/>`;
    out += lines(item, 600, y + 35, { size: 21 });
    if (i < visible.length - 1) out += arrow(600, y + 56, 600, y + 66);
  });
  return out;
}

function compare(items) {
  const left = items.slice(0, 3);
  const right = items.slice(3, 6);
  const panel = (x, values, accent) => `
    <rect x="${x}" y="180" width="480" height="330" rx="28" fill="#101a30" stroke="${accent}" stroke-width="3"/>
    ${lines(values[0] ?? "", x + 240, 235, { size: 30, color: accent, weight: 800 })}
    <line x1="${x + 44}" y1="260" x2="${x + 436}" y2="260" stroke="#334155" stroke-width="2"/>
    ${values.slice(1).map((v, i) => `
      <circle cx="${x + 65}" cy="${320 + i * 92}" r="8" fill="${accent}"/>
      ${lines(wrapText(v, 30).join("\n"), x + 92, 320 + i * 92, { size: 20, anchor: "start", weight: 500, gap: 25 })}
    `).join("")}
  `;
  return panel(90, left, "#38bdf8") + panel(630, right, "#c084fc");
}

function matrix(kind) {
  const lower = kind !== "matrix-upper";
  let out = `<g transform="translate(375,160)">`;
  const size = 58;
  for (let r = 0; r < 6; r += 1) {
    for (let c = 0; c < 6; c += 1) {
      const active = lower ? c <= r : c > r;
      out += `<rect x="${c * size}" y="${r * size}" width="${size - 4}" height="${size - 4}" rx="7" fill="${active ? "#38bdf8" : "#172033"}" stroke="${active ? "#7dd3fc" : "#334155"}"/>`;
      out += lines(active ? "1" : "0", c * size + 27, r * size + 36, { size: 19, color: active ? "#082f49" : "#64748b" });
    }
  }
  out += `</g>`;
  out += lines("Key 位置 →", 550, 145, { size: 20, color: "#94a3b8" });
  out += `<text x="340" y="338" transform="rotate(-90 340 338)" text-anchor="middle" font-size="20" fill="#94a3b8">Query 位置 →</text>`;
  out += card(790, 215, 290, 105, lower ? "亮色：可以查看" : "亮色：未来位置", lower ? "#34d399" : "#fb7185");
  out += card(790, 350, 290, 105, lower ? "暗色：未来被屏蔽" : "暗色：保留区域", "#64748b");
  return out;
}

function attention() {
  let out = "";
  [["Q", 190, "#38bdf8"], ["K", 190, "#818cf8"], ["V", 190, "#34d399"]].forEach(([label, y, color], i) => {
    out += card(70 + i * 175, y, 130, 80, label, color);
  });
  out += card(140, 340, 240, 90, "QKᵀ / √dₖ", "#fbbf24");
  out += arrow(135, 275, 210, 335);
  out += arrow(310, 275, 310, 335);
  out += card(480, 340, 190, 90, "Mask", "#fb7185");
  out += arrow(382, 385, 475, 385);
  out += card(750, 340, 220, 90, "Softmax", "#c084fc");
  out += arrow(675, 385, 745, 385);
  out += card(880, 460, 240, 80, "权重 × V", "#34d399");
  out += arrow(860, 435, 950, 455);
  out += arrow(485, 275, 940, 455, "#34d399");
  return out;
}

function heads() {
  let out = card(70, 285, 180, 110, "输入\n[B,L,D]", "#38bdf8");
  out += arrow(255, 340, 320, 340);
  const colors = ["#38bdf8", "#818cf8", "#c084fc", "#fb7185"];
  colors.forEach((color, i) => {
    const y = 145 + i * 105;
    out += card(330, y, 210, 74, `Head ${i + 1}  [B,L,dₖ]`, color);
    out += arrow(542, y + 37, 650, 270 + i * 18, color);
  });
  out += card(660, 255, 200, 170, "Concat\n[B,L,D]", "#fbbf24");
  out += arrow(865, 340, 930, 340);
  out += card(940, 285, 190, 110, "输出投影", "#34d399");
  return out;
}

function maskFlow() {
  let out = flow(["scores", "masked_fill", "Softmax", "概率权重"]);
  out += `<rect x="390" y="475" width="420" height="64" rx="16" fill="#2a1320" stroke="#fb7185" stroke-width="2"/>`;
  out += lines("mask = 0  →  score = −∞  →  probability = 0", 600, 515, { size: 21, color: "#fecdd3" });
  return out;
}

function wave() {
  const sinPoints = [];
  const cosPoints = [];
  for (let x = 0; x <= 780; x += 8) {
    sinPoints.push(`${210 + x},${330 - Math.sin(x / 65) * 95}`);
    cosPoints.push(`${210 + x},${330 - Math.cos(x / 65) * 95}`);
  }
  return `
    <line x1="170" y1="330" x2="1030" y2="330" stroke="#475569" stroke-width="2"/>
    <line x1="210" y1="190" x2="210" y2="475" stroke="#475569" stroke-width="2"/>
    <polyline points="${sinPoints.join(" ")}" fill="none" stroke="#38bdf8" stroke-width="5"/>
    <polyline points="${cosPoints.join(" ")}" fill="none" stroke="#c084fc" stroke-width="5"/>
    ${lines("sin", 1020, 250, { size: 22, color: "#38bdf8" })}
    ${lines("cos", 1020, 405, { size: 22, color: "#c084fc" })}
    ${lines("位置 pos →", 600, 510, { size: 20, color: "#94a3b8" })}
  `;
}

function hierarchy(items) {
  let out = card(430, 160, 340, 80, items[0], "#fbbf24");
  const children = items.slice(1);
  children.forEach((item, i) => {
    const x = 80 + i * (1040 / children.length);
    out += arrow(600, 245, x + 90, 335, "#64748b");
    out += card(x, 340, 180, 120, item, ["#38bdf8", "#818cf8", "#c084fc", "#fb7185", "#34d399"][i % 5]);
  });
  return out;
}

function architecture(items) {
  const left = items.slice(0, 3);
  const right = items.slice(3, 6);
  let out = `<rect x="120" y="150" width="350" height="380" rx="30" fill="#0d2238" stroke="#38bdf8" stroke-width="3"/>`;
  out += `<rect x="730" y="150" width="350" height="380" rx="30" fill="#21143a" stroke="#c084fc" stroke-width="3"/>`;
  out += lines("ENCODER", 295, 205, { size: 27, color: "#38bdf8", weight: 800 });
  out += lines("DECODER", 905, 205, { size: 27, color: "#c084fc", weight: 800 });
  left.forEach((item, i) => { out += card(165, 245 + i * 88, 260, 62, item, "#38bdf8"); });
  right.forEach((item, i) => { out += card(775, 245 + i * 88, 260, 62, item, "#c084fc"); });
  out += arrow(475, 340, 725, 340, "#fbbf24");
  out += lines("memory", 600, 325, { size: 20, color: "#fbbf24" });
  return out;
}

function testLayout(items) {
  let out = `<rect x="95" y="165" width="1010" height="350" rx="28" fill="#0c1c2c" stroke="#34d399" stroke-width="3"/>`;
  out += lines("✓  可验证条件", 170, 220, { size: 28, color: "#34d399", anchor: "start", weight: 800 });
  items.slice(0, 5).forEach((item, i) => {
    const x = i % 2 === 0 ? 150 : 625;
    const y = 275 + Math.floor(i / 2) * 82;
    out += `<circle cx="${x}" cy="${y - 7}" r="13" fill="#064e3b" stroke="#34d399" stroke-width="2"/>`;
    out += lines("✓", x, y, { size: 18, color: "#6ee7b7" });
    out += lines(item, x + 30, y, { size: 21, anchor: "start", weight: 500 });
  });
  return out;
}

function codeLayout(items) {
  let out = `<rect x="80" y="155" width="1040" height="385" rx="26" fill="#081426" stroke="#334155" stroke-width="2"/>`;
  items.slice(0, 6).forEach((item, i) => {
    const x = 125 + (i % 3) * 350;
    const y = 220 + Math.floor(i / 3) * 150;
    out += `<text x="${x}" y="${y}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="18" fill="#64748b">${String(i + 1).padStart(2, "0")}</text>`;
    out += card(x + 42, y - 40, 270, 94, item, ["#38bdf8", "#818cf8", "#c084fc", "#fb7185", "#fbbf24", "#34d399"][i]);
  });
  return out;
}

function bodyFor(lesson) {
  switch (lesson.type) {
    case "arch": return architecture(lesson.items);
    case "stack": return stack(lesson.items);
    case "compare": return compare(lesson.items);
    case "matrix-upper":
    case "matrix-lower": return matrix(lesson.type);
    case "mask": return maskFlow();
    case "attention": return attention();
    case "heads": return heads();
    case "wave": return wave();
    case "hierarchy": return hierarchy(lesson.items);
    case "test": return testLayout(lesson.items);
    case "code": return codeLayout(lesson.items);
    default: return flow(lesson.items);
  }
}

function svgFor(lesson) {
  const sequence = lesson.n - 105;
  const sequenceId = String(sequence).padStart(2, "0");
  return `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="680" viewBox="0 0 1200 680" role="img" aria-labelledby="title desc">
  <title id="title">${esc(lesson.title)}</title>
  <desc id="desc">${esc(lesson.note)}</desc>
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#07111f"/>
      <stop offset="0.55" stop-color="#0b1730"/>
      <stop offset="1" stop-color="#15102b"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="10" stdDeviation="16" flood-color="#000" flood-opacity=".3"/>
    </filter>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/>
    </marker>
  </defs>
  <rect width="1200" height="680" rx="34" fill="url(#bg)"/>
  <circle cx="1120" cy="70" r="150" fill="#7c3aed" opacity=".09"/>
  <circle cx="70" cy="650" r="190" fill="#0284c7" opacity=".08"/>
  <g filter="url(#shadow)">
    <rect x="55" y="45" width="1090" height="590" rx="30" fill="#081426" fill-opacity=".6" stroke="#26334d"/>
  </g>
  <text x="90" y="100" font-family="Inter, PingFang SC, Microsoft YaHei, sans-serif" font-size="20" font-weight="700" fill="#38bdf8">第 ${sequenceId} 节 · 概念图</text>
  <text x="90" y="145" font-family="Inter, PingFang SC, Microsoft YaHei, sans-serif" font-size="34" font-weight="800" fill="#f8fafc">${esc(lesson.title)}</text>
  <g font-family="Inter, PingFang SC, Microsoft YaHei, sans-serif">
    ${bodyFor(lesson)}
  </g>
  <rect x="90" y="565" width="1020" height="48" rx="14" fill="#0f2741" stroke="#1e3a5f"/>
  <text x="600" y="596" text-anchor="middle" font-family="Inter, PingFang SC, Microsoft YaHei, sans-serif" font-size="19" font-weight="600" fill="#bae6fd">${esc(lesson.note)}</text>
</svg>
`;
}

fs.mkdirSync(outputDir, { recursive: true });

for (const lesson of lessons) {
  const sequence = lesson.n - 105;
  const sequenceId = String(sequence).padStart(2, "0");
  const fileName = `${sequenceId}-concept.svg`;
  fs.writeFileSync(path.join(outputDir, fileName), svgFor(lesson), "utf8");
}
console.log(`Generated ${lessons.length} original SVG diagrams without course identifiers.`);

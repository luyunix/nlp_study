#!/usr/bin/env python3
"""Repository-local completeness and frontend contract checks for NLP Study."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
EXPECTED_TOTAL = 189
REQUIRED_HEADINGS = (
    "## 这节解决什么问题",
    "## 老师原声整理稿",
    "## 完整原声逐段记录",
    "## 最容易踩的坑",
    "## 自测",
    "## 学完检查",
)


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)


def main() -> int:
    failures: list[str] = []
    notes = sorted(ROOT.glob("notes/*/[0-9][0-9]-*.md"))
    transcripts = sorted(ROOT.glob("notes/*/transcripts/p[0-9][0-9][0-9].md"))
    diagrams = sorted(ROOT.glob("notes/*/diagrams/[0-9][0-9]-concept.svg"))

    for label, paths in (("课程文章", notes), ("原声转写", transcripts), ("概念图", diagrams)):
        if len(paths) != EXPECTED_TOTAL:
            fail(f"{label}数量应为 {EXPECTED_TOTAL}，实际为 {len(paths)}", failures)

    for note in notes:
        text = note.read_text(encoding="utf-8")
        for heading in REQUIRED_HEADINGS:
            if heading not in text:
                fail(f"{note.relative_to(ROOT)} 缺少章节：{heading}", failures)
        timed = re.findall(r"^###\s+\d{1,2}:\d{2}", text, re.MULTILINE)
        if len(timed) < 2:
            fail(f"{note.relative_to(ROOT)} 时间戳讲解少于 2 段", failures)
        visible = re.sub(r"```[\s\S]*?```", "", text)
        visible = re.sub(r"[#>*_`|\[\]()\s]", "", visible)
        if len(visible) < 1000:
            fail(f"{note.relative_to(ROOT)} 可读正文过短：{len(visible)} 字", failures)
        for paragraph in re.split(r"\n\s*\n", text):
            if paragraph.startswith("```"):
                continue
            compact = re.sub(r"\s+", "", paragraph)
            if len(compact) > 650:
                fail(f"{note.relative_to(ROOT)} 存在超过 650 字的连续段落", failures)

        public_note = WEB / "public" / note.relative_to(ROOT)
        if not public_note.exists() or public_note.read_bytes() != note.read_bytes():
            fail(f"站点公开副本未同步：{note.relative_to(ROOT)}", failures)

    targeted = {
        ROOT / "notes/attention/03-attention-steps.md": (5, "eat 作查询"),
        ROOT / "notes/seq2seq-translation/06-preprocessing.md": (5, "这里还没有实现 PAD 或 UNK"),
        ROOT / "notes/seq2seq-translation/13-test-plain-decoder.md": (5, "不能被表述成完整推理算法"),
    }
    for note, (minimum_sections, marker) in targeted.items():
        text = note.read_text(encoding="utf-8")
        if len(re.findall(r"^###\s+\d{1,2}:\d{2}", text, re.MULTILINE)) < minimum_sections:
            fail(f"重点补写文章时间段不足：{note.relative_to(ROOT)}", failures)
        if marker not in text:
            fail(f"重点补写文章缺少老师讲解标记：{note.relative_to(ROOT)}", failures)

    accuracy_contracts = {
        "notes/seq2seq-translation/07-dataset.md": {
            "required": ("追加 EOS", "torch.long", "没有定义 UNK"),
            "forbidden": ("padding 更适合放进 collate_fn",),
        },
        "notes/seq2seq-translation/08-dataloader.md": {
            "required": ("batch_size=1", "没有实现 PAD、mask"),
            "forbidden": ("补齐、长度和 mask 一起产出",),
        },
        "notes/seq2seq-translation/12-plain-decoder-code.md": {
            "required": ("LogSoftmax", "NLLLoss"),
            "forbidden": ("CrossEntropyLoss 接 logits",),
        },
        "notes/seq2seq-translation/14-attention-decoder-plan.md": {
            "required": ("Linear(512,10)", "不是直接计算 QK 点积", "没有单独的 PAD mask"),
            "forbidden": ("hidden 作 Q", "mask PAD，Softmax"),
        },
        "notes/seq2seq-translation/17-test-attention-decoder.md": {
            "required": ("[1,10,256]", "喂的是已知法语 token", "不是从 SOS 开始"),
            "forbidden": ("PAD 权重为 0",),
        },
        "notes/seq2seq-translation/20-train-one-batch.md": {
            "required": ("真实 Y 始终用于监督损失", "topk", "NLLLoss"),
            "forbidden": ("CrossEntropy+反传", "PAD 用 ignore_index"),
        },
        "notes/seq2seq-translation/21-view-function.md": {
            "required": ("单个 token 标量", "view(1,-1)", "[1,1]"),
            "forbidden": ("为什么训练损失前要把 [B,T,V]",),
        },
        "notes/seq2seq-translation/22-full-training.md": {
            "required": ("nn.NLLLoss()", "每个 epoch", "五个 Encoder 文件和五个 Decoder 文件"),
            "forbidden": ("验证 no_grad", "保存最佳 checkpoint"),
        },
        "notes/seq2seq-translation/24-prediction-code.md": {
            "required": ("torch.no_grad()", ".eval()", "attention_matrix[:index+1]"),
            "forbidden": ("beam 选 token",),
        },
        "notes/seq2seq-translation/26-tensorboard-graph.md": {
            "required": ("注意力矩阵", "列/x 轴是英语源位置", "行/y 轴是法语目标步骤"),
            "forbidden": ("绘制张量图：用图检查模块连接", "TensorBoard 计算图能帮助什么"),
        },
        "notes/transfer-learning/22-mlm-preprocessing.md": {
            "required": ("input_ids[:, 16]", "自然计数第 17 个 token", "不能擅自改成下标 15"),
            "forbidden": ("pos=15", "保存下标 15", "把下标 15 改成 MASK"),
        },
        "notes/transfer-learning/23-mlm-model.md": {
            "required": ("hidden[:,16,:]", "input_ids[:,16]", "自然计数第 17 个 token"),
            "forbidden": ("hidden[:,15,:]", "取位置 15"),
        },
        "notes/transfer-learning/19-classification-model.md": {
            "required": ("Linear 输出 [B,2] logits", "交给 CrossEntropyLoss", "不需要 forward 先 softmax"),
            "forbidden": ("softmax 得两类概率",),
        },
        "notes/transfer-learning/20-classification-training.md": {
            "required": ("训练 3 轮", "CrossEntropyLoss", "classification1/2/3"),
            "forbidden": ("学习率调度",),
        },
        "notes/transfer-learning/21-classification-evaluation.md": {
            "required": ("选择第三轮", "validation 选最佳轮次", "整个测试集准确率"),
            "forbidden": ("保存最佳模型", "保存/加载最佳 checkpoint"),
        },
        "notes/foundation-models/03-bert-mlm-nsp.md": {
            "required": ("约 15%", "80%", "10%", "NSP 并非所有后续模型都保留"),
            "forbidden": ("所有 token 都替换为 `[MASK]`",),
        },
        "notes/foundation-models/06-gpt-introduction.md": {
            "required": ("交叉注意力被移除", "位置 `≤t`", "位置 t 的隐藏状态用来预测 t+1", "未来位置的 attention score 加上负无穷", "GPT-1 常见是 12 层"),
            "forbidden": ("位置 t 只能看 `<t`", "标准实现直接把 score 乘 0"),
        },
    }
    for relative, contract in accuracy_contracts.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        timed = re.findall(r"^###\s+\d{1,2}:\d{2}", text, re.MULTILINE)
        if len(timed) < 3:
            fail(f"事实重点文章时间段不足：{relative}", failures)
        for marker in contract["required"]:
            if marker not in text:
                fail(f"事实重点文章缺少：{relative} → {marker}", failures)
        for phrase in contract["forbidden"]:
            if phrase in text:
                fail(f"事实重点文章仍含旧错误：{relative} → {phrase}", failures)

    version_file = (WEB / "app/generated-content-version.ts").read_text(encoding="utf-8")
    match = re.search(r'CONTENT_VERSION = "([a-f0-9]{12})"', version_file)
    if not match:
        fail("无法读取前端内容版本", failures)
    else:
        version = match.group(1)
        catalog_file = WEB / f"public/catalog-{version}.json"
        search_file = WEB / f"public/search-{version}.json"
        if not catalog_file.exists() or not search_file.exists():
            fail("版本化目录或全文搜索索引缺失", failures)
        else:
            catalog = json.loads(catalog_file.read_text(encoding="utf-8"))
            search = json.loads(search_file.read_text(encoding="utf-8"))
            if catalog.get("total") != EXPECTED_TOTAL:
                fail("前端目录课程数不是 189", failures)
            if len(search.get("lessons", [])) != EXPECTED_TOTAL:
                fail("全文搜索索引课程数不是 189", failures)

    frontend = (WEB / "app/page.tsx").read_text(encoding="utf-8")
    mermaid = (WEB / "app/MermaidDiagram.tsx").read_text(encoding="utf-8")
    for marker in ("nlp-study-last-lesson", "pushState", "popstate", "handleMarkdownLink", "search-${CONTENT_VERSION}"):
        if marker not in frontend:
            fail(f"前端缺少能力标记：{marker}", failures)
    for marker in ("IntersectionObserver", "useId", "DiagramLightbox", "放大查看"):
        if marker not in mermaid:
            fail(f"流程图缺少能力标记：{marker}", failures)

    if failures:
        print("NLP Study 审计未通过：", file=sys.stderr)
        for item in failures:
            print(f"- {item}", file=sys.stderr)
        return 1

    print(
        f"NLP Study 审计通过：{len(notes)} 篇文章、{len(transcripts)} 份原声转写、"
        f"{len(diagrams)} 张概念图，全文搜索与深链接能力齐全。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

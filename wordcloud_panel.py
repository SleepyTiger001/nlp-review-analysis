"""词云面板：jieba 分词 + 停用词 + wordcloud 出图。

只做三件事：
    1. 用带领域词的 jieba 把影评切词，滤掉停用词与单字；
    2. 按评分把评论分成好评（>=4 星）/ 差评（<=2 星）两组，各出一张词云；
    3. 按「每万词出现次数」归一化后对比两组，挑出各自的特色词。

中文字体必须显式指定，否则 wordcloud 用默认字体，满屏方框。
"""

from __future__ import annotations

import base64
import io
from collections import Counter
from functools import lru_cache
from pathlib import Path

# 电影专有词：不进词表会被切成「阿 / 嬷」「侨 / 批」这种碎字
DOMAIN_WORDS = [
    "阿嬷", "阿公", "南枝", "淑柔", "木生", "晓伟", "潮汕", "潮汕话", "侨批", "银信",
    "下南洋", "红头船", "南洋", "暹罗", "泰国", "唐人街", "橄榄菜", "侨眷", "批局",
    "重男轻女", "女性主义", "女性", "宗族", "乡愁", "情义", "书信", "长镜头", "方言",
]

FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/Deng.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]

# 好评 / 差评各自配色（浅色底、与工作台主色一致）
PALETTES = {
    "pos": "GnBu",
    "neg": "OrRd",
}


def find_font() -> str:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise RuntimeError(
        "找不到中文字体，词云会显示成方框。"
        f"请在 wordcloud_panel.py 的 FONT_CANDIDATES 里补一个本机字体路径，"
        f"已尝试：{FONT_CANDIDATES}"
    )


@lru_cache(maxsize=4)
def load_stopwords(path: str) -> frozenset[str]:
    words: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words.update(line.split())
    return frozenset(words)


def segment(texts: list[str], stopwords: frozenset[str]) -> Counter[str]:
    """切词并计数；只保留长度 >= 2 且不在停用词表里的词。"""
    import jieba

    for word in DOMAIN_WORDS:
        jieba.add_word(word, freq=200_000)

    counter: Counter[str] = Counter()
    for text in texts:
        for token in jieba.cut(text):
            token = token.strip()
            if len(token) < 2 or token in stopwords:
                continue
            if not any("\u4e00" <= ch <= "\u9fff" for ch in token):
                continue  # 纯英文/数字/emoji 不计
            counter[token] += 1
    return counter


def render_cloud(
    freqs: dict[str, int], palette: str, width: int = 900, height: int = 520
) -> str:
    """把词频画成 PNG，返回 data URI。"""
    from wordcloud import WordCloud

    cloud = WordCloud(
        font_path=find_font(),
        width=width,
        height=height,
        background_color="#ffffff",
        max_words=130,
        colormap=palette,
        collocations=False,  # 中文必须关掉，否则会拼出「电影好看」这类伪短语
        prefer_horizontal=0.94,
        relative_scaling=0.42,
        min_font_size=11,
        max_font_size=110,
        margin=3,
        random_state=42,
    ).generate_from_frequencies(freqs)

    buffer = io.BytesIO()
    cloud.to_image().save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _per_10k(counter: Counter[str], total: int) -> dict[str, float]:
    return {w: c / max(total, 1) * 10_000 for w, c in counter.items()}


def analyze(
    texts: list[str],
    scores: list[float],
    stopwords_path: str,
    min_count: int = 3,
    top_n: int = 26,
) -> dict:
    """生成两组词云与对比词。"""
    stopwords = load_stopwords(stopwords_path)

    groups = {
        "pos": [t for t, s in zip(texts, scores, strict=True) if s >= 4],
        "neg": [t for t, s in zip(texts, scores, strict=True) if s <= 2],
        "mid": [t for t, s in zip(texts, scores, strict=True) if s == 3],
    }

    counters = {k: segment(v, stopwords) for k, v in groups.items()}
    totals = {k: sum(c.values()) for k, c in counters.items()}

    panels = []
    for key, title, palette in (
        ("pos", "好评词云 · 4-5 星", PALETTES["pos"]),
        ("neg", "差评词云 · 1-2 星", PALETTES["neg"]),
    ):
        counter = counters[key]
        picked = {w: c for w, c in counter.items() if c >= min_count}
        # 词太少时退回全量，避免空白图
        picked = picked or dict(counter.most_common(60))
        top = [{"word": w, "count": c} for w, c in counter.most_common(top_n)]
        panels.append(
            {
                "key": key,
                "title": title,
                "n": len(groups[key]),
                "unique_words": len(counter),
                "image": render_cloud(picked, palette) if picked else None,
                "top": top,
            }
        )

    pos_rate = _per_10k(counters["pos"], totals["pos"])
    neg_rate = _per_10k(counters["neg"], totals["neg"])
    shared = set(pos_rate) | set(neg_rate)
    delta = {
        w: (pos_rate.get(w, 0.0) - neg_rate.get(w, 0.0), pos_rate.get(w, 0.0), neg_rate.get(w, 0.0))
        for w in shared
        if counters["pos"][w] + counters["neg"][w] >= 3
    }
    pos_specific = [
        {"word": w, "pos_per_10k": round(p, 1), "neg_per_10k": round(n, 1)}
        for w, (d, p, n) in sorted(delta.items(), key=lambda kv: -kv[1][0])[:8]
        if d > 0
    ]
    neg_specific = [
        {"word": w, "pos_per_10k": round(p, 1), "neg_per_10k": round(n, 1)}
        for w, (d, p, n) in sorted(delta.items(), key=lambda kv: kv[1][0])[:8]
        if d < 0
    ]

    return {
        "panels": panels,
        "sizes": {k: len(v) for k, v in groups.items()},
        "tokens": totals,
        "specific": {"pos": pos_specific, "neg": neg_specific},
    }

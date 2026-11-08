"""任务二：复用登录态翻页抓豆瓣短评（文本 / 星级 / 时间）。

输出 data/douban/<电影名>.jsonl，每行 {"text": ..., "score": 1-5, "time": ...}，
`data.py` 用 data.path 指向该文件即可直接读入（text_column=text, label_column=score）。

用法：
    python scripts/douban_scrape.py                       # 默认抓 200 条
    python scripts/douban_scrape.py --target 400 --sort time
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

STATE_PATH = Path("data/douban/storage_state.json")
OUT_DIR = Path("data/douban")


def log(msg: str) -> None:
    print(f"[scrape] {msg}", flush=True)


def parse_page(page) -> list[dict]:
    """把一个短评列表页解析成记录；无星级的（未打分/想看）跳过。"""
    rows: list[dict] = []
    for item in page.query_selector_all("div.comment-item"):
        short = item.query_selector("span.short")
        if short is None:
            continue
        text = (short.inner_text() or "").strip()
        text = re.sub(r"\s*展开\s*$", "", text)
        if not text:
            continue

        info = item.query_selector("span.comment-info")
        if info is None:
            continue

        star = info.query_selector("span.rating")
        score = None
        if star is not None:
            cls = star.get_attribute("class") or ""
            m = re.search(r"allstar(\d+)", cls)
            if m:
                score = int(m.group(1)) // 10  # allstar50 -> 5 星
        if not score or not 1 <= score <= 5:
            continue

        # 注意：comment-time 是 <a> 标签，不是 span，别用 span.comment-time
        stamp = ""
        t = info.query_selector(".comment-time")
        if t is not None:
            stamp = (t.get_attribute("title") or t.inner_text() or "").strip()

        rows.append({"text": text, "score": score, "time": stamp})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="37116446")
    parser.add_argument("--title", default="给阿嬷的情书")
    parser.add_argument("--target", type=int, default=200, help="目标条数")
    parser.add_argument(
        "--sort",
        default="new_score",
        choices=["new_score", "time"],
        help="new_score=热门优先，time=最新优先",
    )
    parser.add_argument("--sleep", type=float, default=3.5, help="翻页间隔基准秒数")
    parser.add_argument(
        "--out",
        default=None,
        help="输出文件名（默认 <电影名>.jsonl）；不同排序分别抓时用来避免互相覆盖",
    )
    args = parser.parse_args()

    if not STATE_PATH.exists():
        raise SystemExit(f"找不到登录态 {STATE_PATH}，先运行 scripts/douban_login.py")

    from playwright.sync_api import sync_playwright

    seen: set[str] = set()
    records: list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="msedge", headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            storage_state=str(STATE_PATH),
            locale="zh-CN",
            timezone_id="Asia/Shanghai",
            viewport={"width": 1440, "height": 900},
        )
        page = context.new_page()
        log(f"登录态已载入，目标 {args.target} 条，排序 {args.sort}")

        start = 0
        empty_pages = 0
        while len(records) < args.target and start < 700:
            url = (
                f"https://movie.douban.com/subject/{args.subject}/comments"
                f"?start={start}&limit=20&status=P&sort={args.sort}"
            )
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(1200)

            rows = parse_page(page)
            fresh = 0
            for r in rows:
                key = r["text"][:80]
                if key in seen:
                    continue
                seen.add(key)
                records.append(r)
                fresh += 1
            log(f"start={start:3d} 解析 {len(rows):2d} 条，新增 {fresh:2d}，累计 {len(records)}")

            if not rows:
                empty_pages += 1
                if empty_pages >= 3:
                    log("连续 3 页为空，停止翻页")
                    break
            else:
                empty_pages = 0

            start += 20
            time.sleep(args.sleep + random.uniform(0, 1.5))

        browser.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / (args.out or f"{args.title}.jsonl")
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dist: dict[int, int] = {}
    for r in records:
        dist[r["score"]] = dist.get(r["score"], 0) + 1
    log(f"写入 {len(records)} 条 → {out.resolve()}")
    log("星级分布 " + "｜".join(f"{k}星:{v}" for k, v in sorted(dist.items())))
    log(f"平均分 {sum(r['score'] for r in records) / max(len(records), 1):.3f}")


if __name__ == "__main__":
    main()

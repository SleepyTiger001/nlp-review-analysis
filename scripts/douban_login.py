"""任务一：有头浏览器登录豆瓣，保存登录态。

流程：
    1. 用本机 Edge（Playwright channel=msedge）打开有头窗口，停在豆瓣登录页；
    2. 由用户本人在窗口中登录（扫码或账号密码），脚本不接触任何凭据；
    3. 检测到登录 cookie（dbcl2）后，把登录态写入 data/douban/storage_state.json；
    4. 紧接着在同一个浏览器会话里打开电影短评页，验证不再跳登录。

用法：
    python scripts/douban_login.py
    python scripts/douban_login.py --subject 37116446 --timeout 600
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

DOUBAN_HOME = "https://www.douban.com/"
LOGIN_URL = "https://accounts.douban.com/passport/login"
STATE_PATH = Path("data/douban/storage_state.json")
# dbcl2 是豆瓣登录后的会话 cookie，未登录时没有
AUTH_COOKIES = ("dbcl2",)


def log(msg: str) -> None:
    print(f"[login] {msg}", flush=True)


def has_auth(context) -> bool:
    names = {c["name"] for c in context.cookies()}
    return all(n in names for n in AUTH_COOKIES)


def probe_comments(page, subject: str) -> tuple[int, str]:
    """在给定会话中打开短评页，统计解析到的评论条数。"""
    url = (
        f"https://movie.douban.com/subject/{subject}/comments"
        "?start=0&limit=20&status=P&sort=new_score"
    )
    page.goto(url, wait_until="domcontentloaded", timeout=45000)
    page.wait_for_timeout(2500)
    items = page.query_selector_all("div.comment-item")
    login_prompt = page.query_selector("a[href*='accounts.douban.com']") is not None
    title = page.title()
    log(f"短评页 title={title!r}｜comment-item={len(items)}｜有登录入口={login_prompt}")
    return len(items), page.url


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="37116446", help="豆瓣电影条目 id")
    parser.add_argument("--timeout", type=int, default=600, help="等待登录的秒数")
    args = parser.parse_args()

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="msedge",
            headless=False,
            args=["--start-maximized", "--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            locale="zh-CN",
            timezone_id="Asia/Shanghai",
        )
        page = context.new_page()
        page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=45000)

        log("浏览器窗口已打开，请在其中完成豆瓣登录（扫码或账号密码均可）")
        log(f"最多等待 {args.timeout} 秒；登录成功后会自动继续，无需手动关闭窗口")

        deadline = time.time() + args.timeout
        logged = False
        while time.time() < deadline:
            if has_auth(context):
                logged = True
                break
            page.wait_for_timeout(1500)

        if not logged:
            log("超时仍未检测到登录态，未保存文件")
            browser.close()
            raise SystemExit(2)

        log("检测到登录 cookie，等待 3 秒让页面完全就绪")
        page.wait_for_timeout(3000)
        context.storage_state(path=str(STATE_PATH))
        log(f"登录态已保存 → {STATE_PATH.resolve()}")

        cookies = context.cookies()
        who = next((c["value"] for c in cookies if c["name"] == "dbcl2"), "")
        log(f"cookie 数量 {len(cookies)}｜dbcl2 长度 {len(who)}")

        # 同一会话内验证短评页
        count, url = probe_comments(page, args.subject)
        summary = {
            "subject": args.subject,
            "saved_to": str(STATE_PATH),
            "cookies": len(cookies),
            "comments_on_first_page": count,
            "final_url": url,
        }
        (STATE_PATH.parent / "login_report.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log("RESULT " + json.dumps(summary, ensure_ascii=False))
        browser.close()


if __name__ == "__main__":
    main()

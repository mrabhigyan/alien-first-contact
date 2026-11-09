"""Screenshot the running dashboard (for checking the design and for the README).

    streamlit run app.py   # in another terminal
    python tools/screenshot.py [run_id] [tab_name] [out.png]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".cache" / "ms-playwright"))

URL = "http://localhost:8501"


def shoot(run_id: str | None, tab: str | None, out: Path, height: int = 1500) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": height}, device_scale_factor=1)
        page.goto(f"{URL}/?run={run_id}" if run_id else URL, wait_until="networkidle")
        page.wait_for_selector("text=Alien First Contact", timeout=60_000)
        page.wait_for_timeout(2500)
        if tab:
            page.get_by_role("tab", name=tab).click()
            page.wait_for_timeout(2500)
        page.screenshot(path=str(out), full_page=True)
        browser.close()
    print(f"saved {out}")


if __name__ == "__main__":
    args = sys.argv[1:] + [None] * 3
    shoot(args[0], args[1], Path(args[2] or ROOT / "logs" / "dashboard.png"))

"""Full-page screenshot capture via Playwright."""
from __future__ import annotations

import io
from typing import Tuple

from PIL import Image


_AUTO_SCROLL_JS = """
async () => {
    await new Promise((resolve) => {
        let total = 0;
        const step = 600;
        const timer = setInterval(() => {
            const before = document.documentElement.scrollHeight;
            window.scrollBy(0, step);
            total += step;
            if (total >= document.documentElement.scrollHeight - window.innerHeight - 5) {
                clearInterval(timer);
                window.scrollTo(0, 0);
                resolve();
            }
        }, 120);
    });
}
"""


def capture_url(
    url: str,
    viewport: Tuple[int, int] = (1440, 900),
    full_page: bool = True,
    timeout_ms: int = 30000,
    auto_scroll: bool = True,
) -> Image.Image:
    """Render ``url`` headlessly and return a PIL image of the page.

    When ``full_page`` is True, the entire scrollable document is captured
    (including content below the viewport). ``auto_scroll`` triggers a slow
    scroll-to-bottom first so lazy-loaded images/sections appear in the shot.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(
                viewport={"width": viewport[0], "height": viewport[1]},
                device_scale_factor=1,
            )
            page = context.new_page()
            page.goto(url, wait_until="networkidle", timeout=timeout_ms)

            if full_page and auto_scroll:
                try:
                    page.evaluate(_AUTO_SCROLL_JS)
                    page.wait_for_load_state("networkidle", timeout=timeout_ms)
                except Exception:
                    pass

            png_bytes = page.screenshot(full_page=full_page, type="png")
        finally:
            browser.close()

    return Image.open(io.BytesIO(png_bytes)).convert("RGB")

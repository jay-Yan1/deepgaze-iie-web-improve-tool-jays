"""Full-page screenshot capture via Playwright."""
from __future__ import annotations

import io
from typing import Tuple

from PIL import Image


def capture_url(
    url: str,
    viewport: Tuple[int, int] = (1440, 900),
    full_page: bool = True,
    timeout_ms: int = 30000,
) -> Image.Image:
    """Render ``url`` headlessly and return a PIL image of the page."""
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
            png_bytes = page.screenshot(full_page=full_page, type="png")
        finally:
            browser.close()

    return Image.open(io.BytesIO(png_bytes)).convert("RGB")

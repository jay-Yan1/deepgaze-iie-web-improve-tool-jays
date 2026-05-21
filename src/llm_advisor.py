"""Use Claude to translate saliency stats into concrete UX improvement suggestions."""
from __future__ import annotations

import base64
import io
import os
from typing import Dict, List, Optional

from PIL import Image

from .aoi import AOIResult
from .visualization import Hotspot

DEFAULT_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a senior UX/conversion analyst.

You receive:
1. An original website screenshot.
2. A second image showing the DeepGaze IIE saliency heatmap overlaid on the same screenshot.
3. Structured statistics: top-K attention hotspots and per-AOI attention share.

DeepGaze IIE is a state-of-the-art free-viewing saliency model. The heatmap predicts where human eyes are *likely* to land in the first seconds before reading begins. High saliency means visual pull, not necessarily business value.

Your job:
- Diagnose where attention is wasted versus where it is well-spent.
- Compare attention share against likely business priority (CTA buttons, value proposition, primary content).
- Output crisp, actionable redesign suggestions (visual hierarchy, contrast, position, whitespace, CTA placement).

Style:
- Write in the same language the user used in their request.
- Be specific: refer to AOI names and hotspot ranks from the data.
- Avoid generic advice. Tie every suggestion to a measurement.
- Keep it under 400 words, use short bullet points grouped by theme."""


def _image_to_b64(image: Image.Image, max_side: int = 1280) -> str:
    img = image.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _format_hotspots(hotspots: List[Hotspot], img_w: int, img_h: int) -> str:
    if not hotspots:
        return "(no significant hotspots detected)"
    lines = []
    for hs in hotspots:
        x, y, w, h = hs.bbox
        lines.append(
            f"- #{hs.rank}: center=({hs.cx},{hs.cy}) "
            f"bbox=(x={x},y={y},w={w},h={h}) "
            f"x_pct={hs.cx / img_w * 100:.0f}%, y_pct={hs.cy / img_h * 100:.0f}%, "
            f"attention_share={hs.attention_share * 100:.1f}%"
        )
    return "\n".join(lines)


def _format_aoi(aoi: List[AOIResult]) -> str:
    if not aoi:
        return "(no AOI provided)"
    lines = []
    for r in aoi:
        lines.append(
            f"- {r.name}: attention={r.attention_share * 100:.1f}%, "
            f"area={r.area_share * 100:.1f}%, intensity_ratio={r.intensity_ratio:.2f}"
        )
    return "\n".join(lines)


def suggest_improvements(
    original: Image.Image,
    overlay: Image.Image,
    hotspots: List[Hotspot],
    aoi: List[AOIResult],
    user_goal: str = "",
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> str:
    """Call Claude to produce written UX suggestions. Returns the assistant text."""
    from anthropic import Anthropic

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Provide it in the sidebar or as an env var."
        )

    client = Anthropic(api_key=api_key)
    w, h = original.size

    data_block = (
        f"Image size: {w} x {h} pixels\n"
        f"User goal / context: {user_goal or '(not specified)'}\n\n"
        f"Top hotspots (ranked by total saliency mass):\n{_format_hotspots(hotspots, w, h)}\n\n"
        f"Per-AOI attention breakdown:\n{_format_aoi(aoi)}"
    )

    response = client.messages.create(
        model=model,
        max_tokens=1200,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": _image_to_b64(original),
                        },
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": _image_to_b64(overlay),
                        },
                    },
                    {"type": "text", "text": data_block},
                ],
            }
        ],
    )

    parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    return "\n".join(parts).strip()

"""Translate saliency stats into concrete UX suggestions via Claude or Gemini."""
from __future__ import annotations

import base64
import io
import os
from typing import Dict, List, Optional

from PIL import Image

from .aoi import AOIResult
from .visualization import Hotspot

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Backwards-compatible alias
DEFAULT_MODEL = DEFAULT_CLAUDE_MODEL

CLAUDE_MODELS = [
    "claude-sonnet-4-6",
    "claude-opus-4-7",
    "claude-haiku-4-5-20251001",
]
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
]

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


def _resize_for_llm(image: Image.Image, max_side: int = 1280) -> Image.Image:
    img = image.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def _image_to_b64(image: Image.Image, max_side: int = 1280) -> str:
    img = _resize_for_llm(image, max_side)
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


def _build_data_block(
    original: Image.Image,
    hotspots: List[Hotspot],
    aoi: List[AOIResult],
    user_goal: str,
) -> str:
    w, h = original.size
    return (
        f"Image size: {w} x {h} pixels\n"
        f"User goal / context: {user_goal or '(not specified)'}\n\n"
        f"Top hotspots (ranked by total saliency mass):\n{_format_hotspots(hotspots, w, h)}\n\n"
        f"Per-AOI attention breakdown:\n{_format_aoi(aoi)}"
    )


def _suggest_claude(
    original: Image.Image,
    overlay: Image.Image,
    data_block: str,
    model: str,
    api_key: str,
) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
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


def _suggest_gemini(
    original: Image.Image,
    overlay: Image.Image,
    data_block: str,
    model: str,
    api_key: str,
) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=[
            _resize_for_llm(original),
            _resize_for_llm(overlay),
            data_block,
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=1500,
        ),
    )
    return (response.text or "").strip()


def suggest_improvements(
    original: Image.Image,
    overlay: Image.Image,
    hotspots: List[Hotspot],
    aoi: List[AOIResult],
    user_goal: str = "",
    provider: str = "claude",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Produce written UX suggestions using the chosen provider.

    ``provider`` is "claude" or "gemini". ``model`` defaults to that provider's
    recommended model. ``api_key`` falls back to the provider's env var.
    """
    provider = (provider or "claude").lower()
    data_block = _build_data_block(original, hotspots, aoi, user_goal)

    if provider == "gemini":
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "Gemini API key is not set. Provide it in the sidebar or set "
                "GEMINI_API_KEY / GOOGLE_API_KEY."
            )
        return _suggest_gemini(original, overlay, data_block, model or DEFAULT_GEMINI_MODEL, key)

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Provide it in the sidebar or as an env var."
        )
    return _suggest_claude(original, overlay, data_block, model or DEFAULT_CLAUDE_MODEL, key)

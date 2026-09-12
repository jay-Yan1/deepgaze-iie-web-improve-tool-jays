#!/usr/bin/env python3
"""MCP server for DeepGaze IIE web-page saliency analysis.

Exposes the same pipeline as ``analyze_cli.py`` over stdio MCP. The reason to
prefer this over the CLI is the process stays alive: DeepGaze's weights are
loaded once and reused, so only the first analysis pays the 10-30s load cost.

Run it directly for a stdio server:

    python mcp_server.py

Register it with Claude Code:

    claude mcp add deepgaze -- python /absolute/path/to/mcp_server.py
"""
from __future__ import annotations

import asyncio
import json
import time
from enum import Enum
from typing import Any, Dict, List, Optional

import anyio
from mcp.server.mcpserver import Context, MCPServer
from mcp_types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.pipeline import (
    TargetError,
    analyze_target,
    diff_reports,
    get_predictor,
    parse_aoi,
    predictor_is_loaded,
)

server = MCPServer("deepgaze_mcp")

# DeepGaze inference is CPU/GPU-bound and the model is not re-entrant, so calls
# are serialized; the event loop stays free because the work runs in a thread.
_INFERENCE_LOCK = asyncio.Lock()

DEFAULT_OUT_DIR = "./saliency_out"

# analyze_target emits at most this many progress messages per page.
_ANALYSIS_STEPS = 6


class ResponseFormat(str, Enum):
    """Output format for tool responses."""

    MARKDOWN = "markdown"
    JSON = "json"


class Layout(str, Enum):
    """Named AOI layouts."""

    BANDS = "bands"
    GRID = "grid"
    CUSTOM = "custom"


class AnalyzeInput(BaseModel):
    """Input model for a single-page saliency analysis."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    target: str = Field(
        ...,
        description=(
            "What to analyze: a URL ('https://example.com', 'http://localhost:3000'), "
            "a local page file ('./index.html'), or an existing screenshot ('./shot.png')."
        ),
        min_length=1,
    )
    out_dir: str = Field(
        default=DEFAULT_OUT_DIR,
        description=(
            "Directory the rendered PNGs are written to, resolved relative to the SERVER's working "
            "directory — pass an absolute path if you need them somewhere specific. Use a fresh "
            "directory per run so before/after images aren't overwritten."
        ),
        min_length=1,
    )
    top_k: int = Field(default=5, description="How many attention hotspots to report.", ge=1, le=20)
    threshold: float = Field(
        default=0.6,
        description="Hotspot cutoff as a fraction of peak saliency; higher isolates only the strongest blobs.",
        ge=0.1,
        le=0.99,
    )
    layout: Layout = Field(
        default=Layout.BANDS,
        description="AOI split: 'bands' (Header/Hero/Body/Footer), 'grid' (3x3), or 'custom' (only the aoi boxes).",
    )
    aoi: List[str] = Field(
        default_factory=list,
        description=(
            "Named regions to measure, each 'NAME:X,Y,W,H' in page pixels (e.g. 'CTA:820,340,300,90'). "
            "Measuring the element you actually care about beats reading the coarse layout bands."
        ),
        max_length=20,
    )
    full_page: bool = Field(
        default=True,
        description="Capture the whole scrollable page. Set false to judge the first screen only (above the fold).",
    )
    viewport_width: int = Field(default=1440, description="Browser viewport width in px.", ge=320, le=3840)
    viewport_height: int = Field(default=900, description="Browser viewport height in px.", ge=320, le=2160)
    max_side: int = Field(
        default=1024,
        description="Longest edge used for inference; lower is faster on CPU.",
        ge=256,
        le=2048,
    )
    timeout_s: int = Field(default=45, description="Page load timeout in seconds.", ge=5, le=180)
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="'markdown' for a compact readable summary, 'json' for the full structured report.",
    )

    @field_validator("aoi")
    @classmethod
    def validate_aoi(cls, v: List[str]) -> List[str]:
        for item in v:
            parse_aoi(item)  # raises ValueError with a precise message
        return v


class ComparePagesInput(BaseModel):
    """Input model for a before/after comparison."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    before: str = Field(..., description="Baseline target (URL, .html file, or screenshot).", min_length=1)
    after: str = Field(..., description="Revised target to compare against the baseline.", min_length=1)
    out_dir: str = Field(
        default=DEFAULT_OUT_DIR,
        description=(
            "Parent directory; images land in <out_dir>/before and <out_dir>/after. Resolved "
            "relative to the SERVER's working directory — pass an absolute path to be sure."
        ),
        min_length=1,
    )
    top_k: int = Field(default=5, description="How many attention hotspots to report per page.", ge=1, le=20)
    layout: Layout = Field(default=Layout.BANDS, description="AOI split used for both pages.")
    aoi: List[str] = Field(
        default_factory=list,
        description="Named regions measured on both pages, each 'NAME:X,Y,W,H'. Keep them identical across runs to compare like for like.",
        max_length=20,
    )
    full_page: bool = Field(default=True, description="Capture the whole scrollable page for both targets.")
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN, description="'markdown' for a readable diff, 'json' for the structured diff."
    )

    @field_validator("aoi")
    @classmethod
    def validate_aoi(cls, v: List[str]) -> List[str]:
        for item in v:
            parse_aoi(item)
        return v


class WarmUpInput(BaseModel):
    """Input model for preloading the model."""

    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    device: Optional[str] = Field(
        default=None,
        description="Torch device to load onto, e.g. 'cpu' or 'cuda'. Omit to auto-select (cuda when available).",
    )


def _handle_error(exc: Exception) -> str:
    """Turn a pipeline failure into a message that says what to do next."""
    if isinstance(exc, TargetError):
        return f"Error: {exc}"
    if isinstance(exc, ValueError):
        return f"Error: {exc}"
    name = type(exc).__name__
    if name in {"TimeoutError", "Error", "TargetClosedError"}:  # playwright surfaces these
        return (
            f"Error: the page failed to load ({exc}). Check the URL is reachable from this machine, "
            "raise timeout_s, or pass a screenshot file instead."
        )
    if isinstance(exc, ImportError):
        return (
            f"Error: a dependency is missing ({exc}). Install with "
            "'pip install -r requirements.txt && python -m playwright install chromium'."
        )
    return f"Error: {name}: {exc}"


def _progress_reporter(ctx: Context, total: float, done_before: float = 0.0):
    """Build a pipeline progress callback that emits MCP progress notifications.

    The pipeline runs in the worker thread anyio.to_thread.run_sync started, so
    the callback hops back to the event loop to await ctx.report_progress. That
    call is a no-op when the client didn't ask for progress.
    """
    state = {"done": done_before}

    def report(message: str) -> None:
        state["done"] = min(state["done"] + 1, total)
        anyio.from_thread.run(ctx.report_progress, state["done"], total, message)

    return report


async def _run_analysis(ctx: Context, total: float, done_before: float = 0.0, **kwargs: Any) -> dict:
    """Run one analysis off the event loop, serialized against other analyses."""
    reporter = _progress_reporter(ctx, total, done_before)
    async with _INFERENCE_LOCK:
        return await anyio.to_thread.run_sync(lambda: analyze_target(progress=reporter, **kwargs))


def _analysis_kwargs(params: AnalyzeInput | ComparePagesInput, target: str, out_dir: str) -> Dict[str, Any]:
    """Map validated tool input onto ``analyze_target`` arguments."""
    kwargs: Dict[str, Any] = {
        "target": target,
        "out_dir": out_dir,
        "top_k": params.top_k,
        "layout": params.layout.value,
        "aoi": [parse_aoi(a) for a in params.aoi],
        "full_page": params.full_page,
    }
    if isinstance(params, AnalyzeInput):
        kwargs.update(
            threshold=params.threshold,
            viewport=(params.viewport_width, params.viewport_height),
            max_side=params.max_side,
            timeout_s=params.timeout_s,
        )
    return kwargs


def _format_report_markdown(report: dict) -> str:
    """Compact readable rendering of one analysis."""
    size = report["size"]
    lines = [
        f"# Saliency: {report['source']}",
        "",
        f"{size['width']}x{size['height']} px · "
        f"{'full page' if report['capture']['full_page'] else 'first screen'} · "
        f"inference {report['inference']['elapsed_ms']}ms on {report['inference']['device']}",
        "",
        "**Look at these before concluding anything:**",
        f"- heatmap: `{report['images']['heatmap']}`",
        f"- hotspots: `{report['images']['hotspots']}`",
        f"- original: `{report['images']['original']}`",
        "",
        "## Hotspots (most eye-catching first)",
        "",
    ]
    if report["hotspots"]:
        lines += ["| # | centre | bbox (x,y,w,h) | attention |", "|---|---|---|---|"]
        for hs in report["hotspots"]:
            lines.append(
                f"| {hs['rank']} | {hs['center'][0]},{hs['center'][1]} | "
                f"{','.join(str(v) for v in hs['bbox'])} | {hs['attention_share'] * 100:.1f}% |"
            )
    else:
        lines.append("_No hotspot passed the threshold — try a lower `threshold`._")

    lines += [
        "",
        "## Regions",
        "",
        "`intensity` = attention share ÷ area share. >1.5 pulls more than its size, <0.5 is being skipped.",
        "",
        "| region | attention | area | intensity |",
        "|---|---|---|---|",
    ]
    for a in report["aoi"]:
        lines.append(
            f"| {a['name']} | {a['attention_share'] * 100:.1f}% | "
            f"{a['area_share'] * 100:.1f}% | {a['intensity_ratio']:.2f} |"
        )
    lines += [
        "",
        "DeepGaze predicts free-viewing eye movement, not clicks or conversion, and it reads visual "
        "features only — it cannot judge whether the copy or the information architecture is right.",
    ]
    return "\n".join(lines)


def _format_diff_markdown(diff: dict) -> str:
    """Readable before/after rendering."""
    lines = [
        "# Saliency comparison",
        "",
        f"- before: {diff['before']['source']} → `{diff['before']['images']['heatmap']}`",
        f"- after: {diff['after']['source']} → `{diff['after']['images']['heatmap']}`",
        "",
        "## Region intensity change",
        "",
        "| region | before | after | delta |",
        "|---|---|---|---|",
    ]
    for row in diff["aoi_changes"]:
        if "intensity_ratio_delta" in row:
            delta = row["intensity_ratio_delta"]
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "–")
            lines.append(
                f"| {row['name']} | {row['before_intensity_ratio']:.2f} | "
                f"{row['after_intensity_ratio']:.2f} | {arrow} {delta:+.2f} |"
            )
        else:
            before = f"{row['before_intensity_ratio']:.2f}" if row["before_intensity_ratio"] is not None else "—"
            after = f"{row['after_intensity_ratio']:.2f}" if row["after_intensity_ratio"] is not None else "—"
            lines.append(f"| {row['name']} | {before} | {after} | _{row['note']}_ |")

    for label, key in (("Before", "hotspots_before"), ("After", "hotspots_after")):
        lines += ["", f"## {label} hotspots", ""]
        if not diff[key]:
            lines.append("_none above threshold_")
            continue
        for hs in diff[key]:
            lines.append(
                f"- #{hs['rank']} at {hs['center'][0]},{hs['center'][1]} — {hs['attention_share'] * 100:.1f}%"
            )
    lines += [
        "",
        "Hotspot ranks are per-page: rank 1 before and rank 1 after are not the same element unless the "
        "coordinates say so. Compare the heatmaps to be sure.",
    ]
    return "\n".join(lines)


@server.tool(
    name="deepgaze_analyze_page",
    title="Analyze Web Page Attention",
    annotations=ToolAnnotations(
        read_only_hint=False,  # writes PNGs to out_dir
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=True,  # loads live URLs
    ),
)
async def deepgaze_analyze_page(params: AnalyzeInput, ctx: Context) -> str:
    """Predict where a user's eyes land first on a web page, using DeepGaze IIE.

    Screenshots the target (unless given an image), runs the saliency model, and
    reports the most eye-catching regions plus per-region attention share. Writes
    three PNGs — original, heatmap overlay, hotspot boxes — and returns their
    paths; open the heatmap image, the numbers alone don't say which element is hot.

    Use it after changing a page's layout, hero, CTA placement, contrast or visual
    hierarchy, to check attention lands where the page's goal needs it. It does not
    modify the page and it does not judge copy or information architecture.

    Args:
        params (AnalyzeInput): Validated input containing:
            - target (str): URL, local .html path, or screenshot path
            - out_dir (str): where the PNGs go (default './saliency_out')
            - top_k (int): hotspots to report, 1-20 (default 5)
            - threshold (float): hotspot cutoff vs peak saliency, 0.1-0.99 (default 0.6)
            - layout (Layout): 'bands' | 'grid' | 'custom' (default 'bands')
            - aoi (List[str]): extra regions as 'NAME:X,Y,W,H'
            - full_page (bool): whole page vs first screen (default True)
            - viewport_width / viewport_height (int): browser size
            - max_side (int): inference resolution cap (default 1024)
            - timeout_s (int): page load timeout (default 45)
            - response_format (ResponseFormat): 'markdown' (default) or 'json'
        ctx (Context): MCP context, used to emit progress notifications.

    Returns:
        str: Markdown summary, or with response_format='json' a JSON object:
        {
            "source": str,                    # URL or file actually analyzed
            "size": {"width": int, "height": int},
            "capture": {"full_page": bool, "viewport": [int, int]},
            "inference": {"device": str, "max_side": int, "elapsed_ms": int},
            "images": {"original": str, "heatmap": str, "hotspots": str},
            "hotspots": [
                {"rank": int, "center": [int, int], "bbox": [int, int, int, int],
                 "area_px": int, "attention_share": float}
            ],
            "aoi": [
                {"name": str, "bbox": [int, int, int, int], "attention_share": float,
                 "area_share": float, "intensity_ratio": float}
            ]
        }
        On failure: "Error: <what went wrong and what to try>".

    Examples:
        - "Does my new hero pull attention?" -> target='http://localhost:3000', full_page=False
        - "Is the CTA getting looked at?" -> aoi=['CTA:820,340,300,90']
        - "Analyze this competitor screenshot" -> target='./competitor.png'
        - Don't use when comparing a redesign against its baseline — use
          deepgaze_compare_pages, which measures both with identical settings.

    Error Handling:
        - Bad AOI strings and out-of-range numbers are rejected by the input model.
        - Unreachable page or load timeout: returns an Error naming the timeout knob.
        - First call loads the model weights (10-30s on CPU); call deepgaze_warm_up
          ahead of time to move that cost out of the first analysis.
    """
    try:
        report = await _run_analysis(
            ctx, _ANALYSIS_STEPS, **_analysis_kwargs(params, params.target, params.out_dir)
        )
    except Exception as exc:
        return _handle_error(exc)

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(report, ensure_ascii=False, indent=2)
    return _format_report_markdown(report)


@server.tool(
    name="deepgaze_compare_pages",
    title="Compare Attention Before and After",
    annotations=ToolAnnotations(
        read_only_hint=False,  # writes PNGs to out_dir
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=True,
    ),
)
async def deepgaze_compare_pages(params: ComparePagesInput, ctx: Context) -> str:
    """Measure two versions of a page with identical settings and diff the attention.

    This is the tool for judging a redesign: it analyzes both targets the same way
    and reports how each region's intensity ratio moved, so "the CTA got more
    attention" becomes a number instead of an impression. Images for each side are
    written to <out_dir>/before and <out_dir>/after.

    Pass the same aoi boxes you used on the baseline. If the layout shifted enough
    that a box no longer covers the same element, re-measure both sides with
    updated boxes rather than trusting the delta.

    Args:
        params (ComparePagesInput): Validated input containing:
            - before (str): baseline URL, .html path, or screenshot
            - after (str): revised URL, .html path, or screenshot
            - out_dir (str): parent directory for both image sets
            - top_k (int): hotspots per page, 1-20 (default 5)
            - layout (Layout): AOI split applied to both (default 'bands')
            - aoi (List[str]): regions measured on both, as 'NAME:X,Y,W,H'
            - full_page (bool): whole page vs first screen (default True)
            - response_format (ResponseFormat): 'markdown' (default) or 'json'
        ctx (Context): MCP context, used to emit progress notifications.

    Returns:
        str: Markdown diff, or with response_format='json':
        {
            "before": {"source": str, "images": {...}},
            "after": {"source": str, "images": {...}},
            "aoi_changes": [
                {"name": str,
                 "before_intensity_ratio": float | null,
                 "after_intensity_ratio": float | null,
                 "before_attention_share": float | null,
                 "after_attention_share": float | null,
                 "intensity_ratio_delta": float,    # present only when in both
                 "attention_share_delta": float,    # present only when in both
                 "note": str}                       # present only when in one
            ],
            "hotspots_before": [...],   # same shape as deepgaze_analyze_page
            "hotspots_after": [...]
        }
        On failure: "Error: <what went wrong and what to try>".

    Examples:
        - "Did moving the CTA up help?" -> before='./before.png', after='http://localhost:3000',
          aoi=['CTA:820,340,300,90']
        - "Compare our landing page to the competitor's" -> before='./ours.png', after='./theirs.png'
        - Don't use for a single page — use deepgaze_analyze_page.

    Error Handling:
        - Either target failing (unreachable, bad file) returns an Error naming it.
        - Runs two inferences, so expect roughly double the single-page time.
    """
    out_root = params.out_dir.rstrip("/")
    total = _ANALYSIS_STEPS * 2
    try:
        before = await _run_analysis(
            ctx, total, 0, **_analysis_kwargs(params, params.before, f"{out_root}/before")
        )
        after = await _run_analysis(
            ctx, total, _ANALYSIS_STEPS, **_analysis_kwargs(params, params.after, f"{out_root}/after")
        )
    except Exception as exc:
        return _handle_error(exc)

    diff = diff_reports(before, after)
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(diff, ensure_ascii=False, indent=2)
    return _format_diff_markdown(diff)


@server.tool(
    name="deepgaze_warm_up",
    title="Preload the DeepGaze Model",
    annotations=ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
)
async def deepgaze_warm_up(params: WarmUpInput, ctx: Context) -> str:
    """Load the DeepGaze IIE weights now so the first real analysis isn't slow.

    The server keeps the model in memory for its lifetime, so this cost is paid
    once per server process. Call it at the start of a design session — while you
    are still writing markup — and later analyses skip straight to inference.
    On the very first ever run this also downloads the weights and centerbias.

    Args:
        params (WarmUpInput): Validated input containing:
            - device (Optional[str]): torch device such as 'cpu' or 'cuda';
              omit to auto-select (cuda when available)
        ctx (Context): MCP context, used to emit progress notifications.

    Returns:
        str: A line stating the device and how long loading took, e.g.
        "Model ready on cpu (loaded in 22.4s)." or
        "Model already loaded on cuda (0.0s)." On failure:
        "Error: <what went wrong and what to try>".

    Examples:
        - Use when: starting a design session that will call deepgaze_analyze_page later.
        - Don't use when: you are about to analyze immediately anyway — the analysis
          loads the model itself.

    Error Handling:
        - Missing torch/DeepGaze packages return an Error naming the install command.
        - An invalid device string surfaces the torch error verbatim.
    """
    already = predictor_is_loaded(params.device)
    await ctx.report_progress(0, 1, "Model already loaded" if already else "Loading DeepGaze IIE weights")
    t0 = time.time()
    try:
        predictor = await anyio.to_thread.run_sync(lambda: get_predictor(params.device))
    except Exception as exc:
        return _handle_error(exc)

    await ctx.report_progress(1, 1, "Model ready")
    elapsed = time.time() - t0
    state = "already loaded" if already else "ready"
    return f"Model {state} on {predictor.device} ({elapsed:.1f}s)."


if __name__ == "__main__":
    server.run(transport="stdio")

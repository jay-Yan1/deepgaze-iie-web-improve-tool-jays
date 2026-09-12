"""Shared saliency analysis pipeline.

The Streamlit app drives the ``src`` modules directly; this module assembles the
same steps into one headless call that returns plain JSON-able dicts, so both
``analyze_cli.py`` and ``mcp_server.py`` produce identical reports.

Heavy imports (torch, playwright, cv2) stay inside the functions that need them,
so importing this module — and validating input — costs nothing.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"})
PAGE_SUFFIXES = frozenset({".html", ".htm"})

Region = Tuple[str, Tuple[int, int, int, int]]
ProgressFn = Callable[[str], None]

# One predictor per device, reused across calls: constructing it loads the
# DeepGaze weights, which is the expensive part (10-30s on CPU).
_PREDICTORS: Dict[str, object] = {}
_PREDICTOR_LOCK = threading.Lock()


class TargetError(ValueError):
    """The requested target can't be turned into an image."""


def parse_aoi(value: str) -> Region:
    """Parse ``Name:x,y,w,h`` into the region tuple ``src.aoi.analyze`` expects."""
    if ":" not in value:
        raise ValueError(f"AOI must look like 'CTA:820,340,300,90', got {value!r}")
    name, coords = value.split(":", 1)
    parts = coords.split(",")
    if len(parts) != 4:
        raise ValueError(f"AOI needs exactly 4 numbers (x,y,w,h), got {coords!r}")
    try:
        x, y, w, h = (int(p.strip()) for p in parts)
    except ValueError:
        raise ValueError(f"AOI coordinates must be integers, got {coords!r}") from None
    if w <= 0 or h <= 0:
        raise ValueError(f"AOI width/height must be positive, got {w}x{h}")
    if not name.strip():
        raise ValueError(f"AOI needs a name before the colon, got {value!r}")
    return name.strip(), (x, y, w, h)


def resolve_target(target: str):
    """Return ``(image_or_None, source)``.

    A ``None`` image means ``source`` is a URL that still needs a screenshot.
    """
    if target.startswith(("http://", "https://", "file://")):
        return None, target

    path = Path(target).expanduser()
    if not path.exists():
        raise TargetError(f"no such file: {path} (pass a URL if you meant a live page)")

    suffix = path.suffix.lower()
    if suffix in PAGE_SUFFIXES:
        return None, path.resolve().as_uri()
    if suffix in IMAGE_SUFFIXES:
        from PIL import Image

        return Image.open(path), str(path)
    raise TargetError(
        f"don't know how to handle {path.name!r}; expected a URL, an .html file, or an image "
        f"({', '.join(sorted(IMAGE_SUFFIXES))})"
    )


def get_predictor(device: Optional[str] = None):
    """Return a cached ``DeepGazePredictor``, loading the weights on first use."""
    from src.saliency import DeepGazePredictor

    key = device or "auto"
    with _PREDICTOR_LOCK:
        predictor = _PREDICTORS.get(key)
        if predictor is None:
            predictor = DeepGazePredictor(device=device)
            _PREDICTORS[key] = predictor
        return predictor


def predictor_is_loaded(device: Optional[str] = None) -> bool:
    """Whether ``get_predictor`` would return without paying the weight-load cost."""
    with _PREDICTOR_LOCK:
        return (device or "auto") in _PREDICTORS


def build_regions(
    layout: str,
    width: int,
    height: int,
    extra_aoi: Sequence[Region] = (),
) -> List[Region]:
    """Combine a named layout with caller-supplied AOI boxes."""
    from src import aoi as aoi_mod

    extra = list(extra_aoi)
    if layout == "custom":
        if not extra:
            raise ValueError("layout 'custom' needs at least one AOI box")
        return extra
    if layout == "grid":
        return aoi_mod.grid_layout(width, height, rows=3, cols=3) + extra
    if layout == "bands":
        return aoi_mod.vertical_band_layout(width, height) + extra
    raise ValueError(f"unknown layout {layout!r}; expected 'bands', 'grid' or 'custom'")


def analyze_target(
    target: str,
    out_dir: str = "./saliency_out",
    *,
    top_k: int = 5,
    threshold: float = 0.6,
    alpha: float = 0.55,
    max_side: int = 1024,
    device: Optional[str] = None,
    layout: str = "bands",
    aoi: Sequence[Region] = (),
    viewport: Tuple[int, int] = (1440, 900),
    full_page: bool = True,
    auto_scroll: bool = True,
    timeout_s: int = 45,
    progress: Optional[ProgressFn] = None,
) -> dict:
    """Screenshot (if needed) -> DeepGaze IIE -> hotspots + AOI -> rendered PNGs.

    Returns a JSON-able report; see ``analyze_cli.py --help`` or the MCP tool
    docstrings for the schema.
    """
    say: ProgressFn = progress or (lambda _msg: None)

    image, source = resolve_target(target)

    from src.aoi import analyze as analyze_aoi
    from src.screenshot import capture_url
    from src.visualization import draw_hotspots, find_hotspots, overlay_heatmap

    if image is None:
        say(f"capturing {source}")
        image = capture_url(
            source,
            viewport=viewport,
            full_page=full_page,
            auto_scroll=auto_scroll,
            timeout_ms=timeout_s * 1000,
        )
    width, height = image.size
    say(f"page is {width}x{height} px")

    # Fail before inference rather than after it.
    regions = build_regions(layout, width, height, aoi)

    say("loading DeepGaze IIE" if not predictor_is_loaded(device) else "reusing loaded DeepGaze IIE")
    predictor = get_predictor(device)
    say("running inference")
    t0 = time.time()
    saliency = predictor.predict(image, max_side=max_side)
    infer_ms = (time.time() - t0) * 1000
    say(f"inference done in {infer_ms / 1000:.1f}s on {predictor.device}")

    out_path = Path(out_dir).expanduser()
    out_path.mkdir(parents=True, exist_ok=True)

    hotspots = find_hotspots(saliency, top_k=top_k, threshold_pct=threshold)
    images = {
        "original": out_path / "original.png",
        "heatmap": out_path / "heatmap.png",
        "hotspots": out_path / "hotspots.png",
    }
    image.save(images["original"])
    overlay_heatmap(image, saliency, alpha=alpha).save(images["heatmap"])
    draw_hotspots(image, hotspots).save(images["hotspots"])
    say(f"wrote 3 PNGs to {out_path}")

    return {
        "source": source,
        "size": {"width": width, "height": height},
        "capture": {"full_page": full_page, "viewport": list(viewport)},
        "inference": {
            "device": predictor.device,
            "max_side": max_side,
            "elapsed_ms": round(infer_ms),
        },
        "images": {name: str(path) for name, path in images.items()},
        "hotspots": [
            {
                "rank": hs.rank,
                "center": [hs.cx, hs.cy],
                "bbox": list(hs.bbox),
                "area_px": hs.area_px,
                "attention_share": round(hs.attention_share, 4),
            }
            for hs in hotspots
        ],
        "aoi": [
            {
                "name": r.name,
                "bbox": list(r.bbox),
                "attention_share": round(r.attention_share, 4),
                "area_share": round(r.area_share, 4),
                "intensity_ratio": round(r.intensity_ratio, 2),
            }
            for r in analyze_aoi(saliency, regions)
        ],
    }


def diff_reports(before: dict, after: dict) -> dict:
    """Compare two reports so a redesign can be judged against its baseline.

    AOIs are matched by name; hotspots by rank. An AOI present in only one of
    the two runs is reported with a null on the missing side rather than
    silently dropped, since a changed layout is exactly when that happens.
    """
    before_aoi = {a["name"]: a for a in before["aoi"]}
    after_aoi = {a["name"]: a for a in after["aoi"]}

    aoi_rows = []
    for name in list(before_aoi) + [n for n in after_aoi if n not in before_aoi]:
        b, a = before_aoi.get(name), after_aoi.get(name)
        row = {
            "name": name,
            "before_intensity_ratio": b["intensity_ratio"] if b else None,
            "after_intensity_ratio": a["intensity_ratio"] if a else None,
            "before_attention_share": b["attention_share"] if b else None,
            "after_attention_share": a["attention_share"] if a else None,
        }
        if b and a:
            row["intensity_ratio_delta"] = round(a["intensity_ratio"] - b["intensity_ratio"], 2)
            row["attention_share_delta"] = round(a["attention_share"] - b["attention_share"], 4)
        else:
            row["note"] = "only in before" if b else "only in after"
        aoi_rows.append(row)

    return {
        "before": {"source": before["source"], "images": before["images"]},
        "after": {"source": after["source"], "images": after["images"]},
        "aoi_changes": aoi_rows,
        "hotspots_before": before["hotspots"],
        "hotspots_after": after["hotspots"],
    }

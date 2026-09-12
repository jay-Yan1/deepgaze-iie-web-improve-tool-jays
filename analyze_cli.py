#!/usr/bin/env python3
"""Headless DeepGaze IIE saliency analysis.

Same pipeline as the Streamlit app, but driven from the command line and
emitting JSON on stdout so an agent (e.g. Claude Code) can call it while
iterating on a web page design.

Unlike the Streamlit app this deliberately does NOT call an LLM: it only
produces measurements plus rendered PNGs. The caller is expected to look at
the images and interpret the numbers itself.

Examples:
    python analyze_cli.py https://example.com
    python analyze_cli.py ./index.html --first-screen
    python analyze_cli.py shot.png --layout grid --top-k 3
    python analyze_cli.py http://localhost:3000 --aoi "CTA:820,340,300,90"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}
PAGE_SUFFIXES = {".html", ".htm"}


def log(msg: str) -> None:
    """Progress goes to stderr so stdout stays pure JSON."""
    print(msg, file=sys.stderr, flush=True)


def parse_viewport(value: str) -> Tuple[int, int]:
    try:
        w, h = value.lower().split("x", 1)
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError(f"viewport must look like 1440x900, got {value!r}")


def parse_aoi(value: str) -> Tuple[str, Tuple[int, int, int, int]]:
    """Parse ``Name:x,y,w,h`` into the region tuple ``src.aoi.analyze`` expects."""
    if ":" not in value:
        raise argparse.ArgumentTypeError(f"--aoi must look like 'CTA:820,340,300,90', got {value!r}")
    name, coords = value.split(":", 1)
    parts = coords.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"--aoi needs exactly 4 numbers (x,y,w,h), got {coords!r}")
    try:
        x, y, w, h = (int(p.strip()) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--aoi coordinates must be integers, got {coords!r}")
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"--aoi width/height must be positive, got {w}x{h}")
    return name.strip(), (x, y, w, h)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analyze_cli.py",
        description="Predict where users look first on a web page (DeepGaze IIE), as JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples:", 1)[1] if "Examples:" in __doc__ else None,
    )
    p.add_argument(
        "target",
        help="URL (http/https/file), path to a local .html file, or path to an existing screenshot.",
    )
    p.add_argument("--out", default="./saliency_out", help="Directory for the rendered PNGs (default: %(default)s)")

    g = p.add_argument_group("hotspots & heatmap")
    g.add_argument("--top-k", type=int, default=5, help="How many hotspots to report (default: %(default)s)")
    g.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Hotspot cutoff as a fraction of peak saliency; higher = only the strongest blobs (default: %(default)s)",
    )
    g.add_argument("--alpha", type=float, default=0.55, help="Heatmap overlay opacity (default: %(default)s)")
    g.add_argument(
        "--max-side",
        type=int,
        default=1024,
        help="Longest edge used for inference; lower is faster on CPU (default: %(default)s)",
    )
    g.add_argument("--device", default=None, help="Force a torch device, e.g. cpu or cuda (default: auto)")

    g = p.add_argument_group("AOI layout")
    g.add_argument(
        "--layout",
        choices=["bands", "grid", "custom"],
        default="bands",
        help="bands = Header/Hero/Body/Footer, grid = 3x3, custom = only the --aoi boxes (default: %(default)s)",
    )
    g.add_argument(
        "--aoi",
        type=parse_aoi,
        action="append",
        default=[],
        metavar="NAME:X,Y,W,H",
        help="Extra named region to measure; repeatable. Added on top of --layout unless --layout custom.",
    )

    g = p.add_argument_group("screenshot (URL / .html targets only)")
    g.add_argument("--viewport", type=parse_viewport, default=(1440, 900), metavar="WxH", help="(default: 1440x900)")
    g.add_argument(
        "--first-screen",
        action="store_true",
        help="Capture only the viewport (above the fold) instead of the whole scrollable page.",
    )
    g.add_argument("--no-auto-scroll", action="store_true", help="Skip the pre-screenshot scroll that triggers lazy-loading.")
    g.add_argument("--timeout", type=int, default=45, help="Page load timeout in seconds (default: %(default)s)")

    return p


def resolve_target(target: str):
    """Return ``(PIL image, source description)`` for a URL, .html file, or image file."""
    if target.startswith(("http://", "https://", "file://")):
        return None, target  # screenshot path, handled by caller

    path = Path(target).expanduser()
    if not path.exists():
        raise SystemExit(f"error: no such file: {path} (did you mean a URL?)")

    suffix = path.suffix.lower()
    if suffix in PAGE_SUFFIXES:
        return None, path.resolve().as_uri()
    if suffix in IMAGE_SUFFIXES:
        from PIL import Image

        return Image.open(path), str(path)
    raise SystemExit(
        f"error: don't know how to handle {path.name!r}; expected a URL, an .html file, or an image "
        f"({', '.join(sorted(IMAGE_SUFFIXES))})"
    )


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Fail on bad input before paying for the screenshot and a ~15s inference.
    if args.layout == "custom" and not args.aoi:
        raise SystemExit("error: --layout custom needs at least one --aoi NAME:X,Y,W,H")
    image, source = resolve_target(args.target)

    # Heavy imports stay inside main so --help works without torch installed.
    from src.aoi import analyze as analyze_aoi, grid_layout, vertical_band_layout
    from src.saliency import DeepGazePredictor
    from src.screenshot import capture_url
    from src.visualization import draw_hotspots, find_hotspots, overlay_heatmap

    if image is None:
        log(f"[1/3] capturing {source} …")
        image = capture_url(
            source,
            viewport=args.viewport,
            full_page=not args.first_screen,
            auto_scroll=not args.no_auto_scroll,
            timeout_ms=args.timeout * 1000,
        )
    width, height = image.size
    log(f"      page is {width}x{height} px")

    log("[2/3] running DeepGaze IIE (first run downloads weights; CPU takes 5-15s per image) …")
    predictor = DeepGazePredictor(device=args.device)
    t0 = time.time()
    saliency = predictor.predict(image, max_side=args.max_side)
    infer_ms = (time.time() - t0) * 1000
    log(f"      done in {infer_ms / 1000:.1f}s on {predictor.device}")

    log("[3/3] rendering …")
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    hotspots = find_hotspots(saliency, top_k=args.top_k, threshold_pct=args.threshold)
    original_png = out_dir / "original.png"
    heatmap_png = out_dir / "heatmap.png"
    hotspots_png = out_dir / "hotspots.png"
    image.save(original_png)
    overlay_heatmap(image, saliency, alpha=args.alpha).save(heatmap_png)
    draw_hotspots(image, hotspots).save(hotspots_png)

    if args.layout == "custom":
        regions = list(args.aoi)
    elif args.layout == "grid":
        regions = grid_layout(width, height, rows=3, cols=3) + list(args.aoi)
    else:
        regions = vertical_band_layout(width, height) + list(args.aoi)
    aoi_results = analyze_aoi(saliency, regions)

    report = {
        "source": source,
        "size": {"width": width, "height": height},
        "capture": {
            "full_page": not args.first_screen,
            "viewport": list(args.viewport),
        },
        "inference": {
            "device": predictor.device,
            "max_side": args.max_side,
            "elapsed_ms": round(infer_ms),
        },
        "images": {
            "original": str(original_png),
            "heatmap": str(heatmap_png),
            "hotspots": str(hotspots_png),
        },
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
            for r in aoi_results
        ],
    }
    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
from typing import List, Tuple

from src.pipeline import TargetError, analyze_target, parse_aoi


def log(msg: str) -> None:
    """Progress goes to stderr so stdout stays pure JSON."""
    print(f"  {msg}", file=sys.stderr, flush=True)


def _viewport(value: str) -> Tuple[int, int]:
    try:
        w, h = value.lower().split("x", 1)
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError(f"viewport must look like 1440x900, got {value!r}")


def _aoi(value: str):
    try:
        return parse_aoi(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))


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
        type=_aoi,
        action="append",
        default=[],
        metavar="NAME:X,Y,W,H",
        help="Extra named region to measure; repeatable. Added on top of --layout unless --layout custom.",
    )

    g = p.add_argument_group("screenshot (URL / .html targets only)")
    g.add_argument("--viewport", type=_viewport, default=(1440, 900), metavar="WxH", help="(default: 1440x900)")
    g.add_argument(
        "--first-screen",
        action="store_true",
        help="Capture only the viewport (above the fold) instead of the whole scrollable page.",
    )
    g.add_argument("--no-auto-scroll", action="store_true", help="Skip the pre-screenshot scroll that triggers lazy-loading.")
    g.add_argument("--timeout", type=int, default=45, help="Page load timeout in seconds (default: %(default)s)")

    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.layout == "custom" and not args.aoi:
        raise SystemExit("error: --layout custom needs at least one --aoi NAME:X,Y,W,H")

    try:
        report = analyze_target(
            args.target,
            args.out,
            top_k=args.top_k,
            threshold=args.threshold,
            alpha=args.alpha,
            max_side=args.max_side,
            device=args.device,
            layout=args.layout,
            aoi=args.aoi,
            viewport=args.viewport,
            full_page=not args.first_screen,
            auto_scroll=not args.no_auto_scroll,
            timeout_s=args.timeout,
            progress=log,
        )
    except TargetError as exc:
        raise SystemExit(f"error: {exc}")

    json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

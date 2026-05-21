"""Heatmap overlay and top-K hotspot detection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class Hotspot:
    rank: int
    cx: int
    cy: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area_px: int
    attention_share: float  # fraction of total saliency mass in this region


def overlay_heatmap(image: Image.Image, saliency: np.ndarray, alpha: float = 0.55) -> Image.Image:
    """Blend a JET colormap of ``saliency`` on top of ``image``."""
    base = np.asarray(image.convert("RGB"))
    h, w = base.shape[:2]
    if saliency.shape != (h, w):
        saliency = cv2.resize(saliency, (w, h), interpolation=cv2.INTER_LINEAR)

    norm = np.clip(saliency, 0.0, 1.0)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    blended = (base.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def find_hotspots(saliency: np.ndarray, top_k: int = 5, threshold_pct: float = 0.6) -> List[Hotspot]:
    """Identify the top-K most salient connected regions.

    ``threshold_pct`` keeps pixels whose saliency >= (max * pct) before grouping.
    """
    h, w = saliency.shape
    if saliency.max() <= 0:
        return []

    norm = saliency / saliency.max()
    binary = (norm >= threshold_pct).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    total_mass = float(saliency.sum())

    candidates = []
    for label_id in range(1, num_labels):
        x, y, bw, bh, area = stats[label_id]
        mask = labels == label_id
        mass = float(saliency[mask].sum())
        share = mass / total_mass if total_mass > 0 else 0.0
        cx, cy = centroids[label_id]
        candidates.append((mass, share, int(cx), int(cy), (int(x), int(y), int(bw), int(bh)), int(area)))

    candidates.sort(key=lambda t: t[0], reverse=True)
    hotspots: List[Hotspot] = []
    for rank, (_, share, cx, cy, bbox, area) in enumerate(candidates[:top_k], start=1):
        hotspots.append(Hotspot(rank=rank, cx=cx, cy=cy, bbox=bbox, area_px=area, attention_share=share))
    return hotspots


def draw_hotspots(image: Image.Image, hotspots: List[Hotspot]) -> Image.Image:
    """Annotate top-K bounding boxes with rank labels."""
    canvas = np.asarray(image.convert("RGB")).copy()
    for hs in hotspots:
        x, y, w, h = hs.bbox
        color = (255, 220, 0)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 3)
        label = f"#{hs.rank}  {hs.attention_share * 100:.1f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(canvas, (x, y - th - 10), (x + tw + 10, y), color, -1)
        cv2.putText(canvas, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.circle(canvas, (hs.cx, hs.cy), 6, (255, 0, 0), -1)
    return Image.fromarray(canvas)

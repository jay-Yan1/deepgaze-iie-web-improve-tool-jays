"""Area-of-interest analysis: split the page into named regions and measure attention share."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class AOIResult:
    name: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    attention_share: float           # fraction of total saliency mass
    area_share: float                # fraction of total pixel area
    intensity_ratio: float           # attention_share / area_share (>1 = over-attended)


def vertical_band_layout(width: int, height: int) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """Split a page into Header / Hero / Body / Footer vertical bands."""
    header_h = int(height * 0.12)
    hero_h = int(height * 0.30)
    footer_h = int(height * 0.12)
    body_h = height - header_h - hero_h - footer_h
    bands = [
        ("Header", (0, 0, width, header_h)),
        ("Hero", (0, header_h, width, hero_h)),
        ("Body", (0, header_h + hero_h, width, body_h)),
        ("Footer", (0, height - footer_h, width, footer_h)),
    ]
    return bands


def grid_layout(width: int, height: int, rows: int = 3, cols: int = 3) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """Split the page into a rows x cols grid (e.g. Top-Left, Top-Center ...)."""
    row_names = ["Top", "Middle", "Bottom"] if rows == 3 else [f"R{i+1}" for i in range(rows)]
    col_names = ["Left", "Center", "Right"] if cols == 3 else [f"C{i+1}" for i in range(cols)]

    cell_h = height // rows
    cell_w = width // cols
    cells = []
    for r in range(rows):
        for c in range(cols):
            x = c * cell_w
            y = r * cell_h
            w = width - x if c == cols - 1 else cell_w
            h = height - y if r == rows - 1 else cell_h
            cells.append((f"{row_names[r]}-{col_names[c]}", (x, y, w, h)))
    return cells


def analyze(saliency: np.ndarray, regions: List[Tuple[str, Tuple[int, int, int, int]]]) -> List[AOIResult]:
    """Compute attention share per region."""
    total_mass = float(saliency.sum())
    total_area = float(saliency.shape[0] * saliency.shape[1])
    results: List[AOIResult] = []
    for name, (x, y, w, h) in regions:
        region = saliency[y : y + h, x : x + w]
        mass = float(region.sum())
        share = mass / total_mass if total_mass > 0 else 0.0
        area_share = (w * h) / total_area if total_area > 0 else 0.0
        intensity = share / area_share if area_share > 0 else 0.0
        results.append(
            AOIResult(
                name=name,
                bbox=(x, y, w, h),
                attention_share=share,
                area_share=area_share,
                intensity_ratio=intensity,
            )
        )
    return results


def summarize(results: List[AOIResult]) -> Dict[str, float]:
    """Flatten AOI list to a {name: attention_share} dict, useful for the LLM prompt."""
    return {r.name: round(r.attention_share, 4) for r in results}

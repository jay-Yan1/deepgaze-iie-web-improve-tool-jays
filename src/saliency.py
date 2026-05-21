"""DeepGaze IIE saliency prediction wrapper."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
from PIL import Image
from scipy.ndimage import zoom
from scipy.special import logsumexp

CENTERBIAS_URL = (
    "https://github.com/matthias-k/DeepGaze/raw/main/centerbias_mit1003.npy"
)
CENTERBIAS_PATH = Path(__file__).resolve().parent.parent / "assets" / "centerbias_mit1003.npy"


def _ensure_centerbias() -> np.ndarray:
    """Download MIT1003 centerbias if not already cached, then load it."""
    if not CENTERBIAS_PATH.exists():
        CENTERBIAS_PATH.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(CENTERBIAS_URL, timeout=60)
        resp.raise_for_status()
        CENTERBIAS_PATH.write_bytes(resp.content)
    return np.load(CENTERBIAS_PATH)


class DeepGazePredictor:
    """Thin wrapper around deepgaze_pytorch.DeepGazeIIE."""

    def __init__(self, device: Optional[str] = None) -> None:
        import deepgaze_pytorch  # imported lazily so the app can boot without GPU

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(self.device)
        self.model.eval()
        self.centerbias_template = _ensure_centerbias()

    @torch.inference_mode()
    def predict(self, image: Image.Image, max_side: int = 1024) -> np.ndarray:
        """Return a normalized saliency map (H, W) in [0, 1] aligned to the input image size.

        ``max_side`` bounds the inference resolution so CPU runs stay tractable.
        """
        rgb = image.convert("RGB")
        orig_w, orig_h = rgb.size

        scale = min(1.0, max_side / max(orig_w, orig_h))
        if scale < 1.0:
            work_w, work_h = int(orig_w * scale), int(orig_h * scale)
            work = rgb.resize((work_w, work_h), Image.BILINEAR)
        else:
            work_w, work_h = orig_w, orig_h
            work = rgb

        arr = np.asarray(work, dtype=np.float32)

        cb = zoom(
            self.centerbias_template,
            (work_h / self.centerbias_template.shape[0], work_w / self.centerbias_template.shape[1]),
            order=0,
            mode="nearest",
        )
        cb -= logsumexp(cb)

        image_tensor = torch.tensor(arr.transpose(2, 0, 1)[None, ...]).to(self.device)
        cb_tensor = torch.tensor(cb[None, ...], dtype=torch.float32).to(self.device)

        log_density = self.model(image_tensor, cb_tensor)
        density = log_density.exp().squeeze().cpu().numpy()

        if (work_w, work_h) != (orig_w, orig_h):
            density = zoom(
                density,
                (orig_h / density.shape[0], orig_w / density.shape[1]),
                order=1,
                mode="nearest",
            )

        density -= density.min()
        denom = density.max()
        if denom > 0:
            density = density / denom
        return density.astype(np.float32)

"""Grad-CAM post-processing (NumPy only, so it can be tested without PyTorch)."""
from __future__ import annotations

import numpy as np

STRONG_PEAK = 1.4   # scale used when the model says "pneumonia" (p >= 0.5)
FAINT_PEAK = 0.4    # scale used otherwise, so the heatmap looks faint on a normal X-ray


def cam_to_grid(cam: np.ndarray, p: float, n: int = 8) -> list[float]:
    """Average-pool a square CAM to an n x n grid (row-major) for the website's heatmap.

    The grid is normalised to a maximum of 1 and then scaled: strong when p >= 0.5,
    faint otherwise. Raises ValueError for input that cannot be pooled.
    """
    cam = np.asarray(cam, dtype=float)
    if cam.ndim != 2 or cam.shape[0] != cam.shape[1]:
        raise ValueError("cam must be a square 2-D array")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be a probability between 0 and 1")
    if n < 1 or cam.shape[0] < n:
        raise ValueError("cam is smaller than the requested grid")
    k = cam.shape[0] // n
    grid = cam[: k * n, : k * n].reshape(n, k, n, k).mean(axis=(1, 3))
    grid = grid / (grid.max() + 1e-8)
    peak = STRONG_PEAK if p >= 0.5 else FAINT_PEAK
    return (grid * peak).flatten().round(4).tolist()

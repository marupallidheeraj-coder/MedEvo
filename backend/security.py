"""Request-safety helpers for the API (no PyTorch or FastAPI imports, easy to unit test)."""
from __future__ import annotations

import io
import time
import warnings
from collections import deque

from PIL import Image, UnidentifiedImageError

MAX_BYTES = 8 * 1024 * 1024          # upload size limit
MAX_PIXELS = 25_000_000              # guards against decompression bombs
MIN_SIDE = 32                        # a real X-ray is far larger than this
ALLOWED_FORMATS = {"JPEG", "PNG"}

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class ImageRejected(Exception):
    """Raised when an upload should be refused. Carries the HTTP status and a safe message."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def validate_image_bytes(raw: bytes) -> Image.Image:
    """Return a fully decoded PIL image, or raise ImageRejected."""
    if not raw:
        raise ImageRejected(400, "Empty file")
    if len(raw) > MAX_BYTES:
        raise ImageRejected(413, "Image too large (max 8 MB)")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(raw))
            fmt = img.format
            if fmt not in ALLOWED_FORMATS:
                raise ImageRejected(415, "Unsupported image type (use JPG or PNG)")
            if min(img.size) < MIN_SIDE:
                raise ImageRejected(400, f"Image too small (each side must be at least {MIN_SIDE} px)")
            img.load()
    except ImageRejected:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageRejected(413, "Image dimensions too large") from None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise ImageRejected(400, "Not a valid image file") from None
    return img


class RateLimiter:
    """Sliding-window limiter: at most `limit` requests per `window` seconds per key."""

    def __init__(self, limit: int = 30, window: float = 60.0):
        if limit < 1 or window <= 0:
            raise ValueError("limit must be >= 1 and window > 0")
        self.limit, self.window = limit, window
        self._hits: dict[str, deque] = {}

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        q = self._hits.setdefault(key, deque())
        while q and now - q[0] >= self.window:
            q.popleft()
        if len(q) >= self.limit:
            return False
        q.append(now)
        if len(self._hits) > 10_000:                     # keep memory bounded
            self._hits = {k: v for k, v in self._hits.items() if v and now - v[-1] < self.window}
        return True


def parse_origins(value: str) -> list[str]:
    """Turn a comma-separated ALLOWED_ORIGINS string into a clean list."""
    return [o.strip().rstrip("/") for o in value.split(",") if o.strip()]

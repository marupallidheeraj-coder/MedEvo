import io
import unittest

from PIL import Image

import _paths  # noqa: F401
from security import (MAX_BYTES, ImageRejected, RateLimiter, parse_origins,
                      validate_image_bytes)


def encode(fmt: str, size=(64, 64), mode="L") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, 128).save(buf, format=fmt)
    return buf.getvalue()


class ImageValidationTests(unittest.TestCase):
    def status_of(self, raw: bytes) -> int:
        with self.assertRaises(ImageRejected) as cm:
            validate_image_bytes(raw)
        return cm.exception.status

    def test_accepts_png_and_jpeg(self):
        self.assertEqual(validate_image_bytes(encode("PNG")).size, (64, 64))
        self.assertEqual(validate_image_bytes(encode("JPEG")).size, (64, 64))

    def test_rejects_empty_upload(self):
        self.assertEqual(self.status_of(b""), 400)

    def test_rejects_oversized_upload_before_decoding(self):
        self.assertEqual(self.status_of(b"\0" * (MAX_BYTES + 1)), 413)

    def test_rejects_non_image_bytes(self):
        self.assertEqual(self.status_of(b"this is not an image" * 50), 400)

    def test_rejects_other_image_formats(self):
        self.assertEqual(self.status_of(encode("GIF")), 415)
        self.assertEqual(self.status_of(encode("BMP")), 415)

    def test_rejects_tiny_images(self):
        self.assertEqual(self.status_of(encode("PNG", size=(8, 8))), 400)

    def test_rejects_truncated_image(self):
        raw = encode("PNG", size=(200, 200), mode="RGB")
        self.assertEqual(self.status_of(raw[: len(raw) // 2]), 400)

    def test_rejects_decompression_bomb(self):
        # ~36 megapixels of one colour compresses to a tiny file but must still be refused
        self.assertEqual(self.status_of(encode("PNG", size=(6000, 6000))), 413)

    def test_error_messages_do_not_leak_internals(self):
        with self.assertRaises(ImageRejected) as cm:
            validate_image_bytes(b"garbage" * 30)
        self.assertNotIn("Traceback", cm.exception.detail)
        self.assertNotIn("PIL", cm.exception.detail)


class RateLimiterTests(unittest.TestCase):
    def test_blocks_after_limit_and_recovers(self):
        rl = RateLimiter(limit=3, window=60)
        self.assertTrue(all(rl.allow("a", now=t) for t in (0, 1, 2)))
        self.assertFalse(rl.allow("a", now=3))
        self.assertTrue(rl.allow("a", now=61))       # first hit has expired

    def test_clients_are_independent(self):
        rl = RateLimiter(limit=1, window=60)
        self.assertTrue(rl.allow("a", now=0))
        self.assertTrue(rl.allow("b", now=0))
        self.assertFalse(rl.allow("a", now=1))

    def test_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            RateLimiter(limit=0)
        with self.assertRaises(ValueError):
            RateLimiter(window=0)


class ParseOriginsTests(unittest.TestCase):
    def test_splits_trims_and_drops_empty(self):
        self.assertEqual(parse_origins(" https://a.com/ , ,http://b.com,"), ["https://a.com", "http://b.com"])

    def test_wildcard_passes_through(self):
        self.assertEqual(parse_origins("*"), ["*"])

    def test_empty_string_gives_empty_list(self):
        self.assertEqual(parse_origins(""), [])


if __name__ == "__main__":
    unittest.main()

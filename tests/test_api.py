import io
import os
import unittest

from PIL import Image

import _paths  # noqa: F401

try:
    import fastapi  # noqa: F401
    import httpx  # noqa: F401
    import torch  # noqa: F401
    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False


def png_bytes(size=(96, 96)) -> bytes:
    buf = io.BytesIO()
    Image.new("L", size, 90).save(buf, format="PNG")
    return buf.getvalue()


@unittest.skipUnless(HAVE_STACK, "needs fastapi, httpx and torch")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["MODEL_PATH"] = "/nonexistent/model.pt"       # untrained weights are fine for API tests
        os.environ["RATE_LIMIT_PER_MIN"] = "1000"
        from fastapi.testclient import TestClient
        import app as app_module
        cls.mod = app_module
        cls.client = TestClient(app_module.app)

    def post(self, data: bytes, name="xray.png", ctype="image/png"):
        return self.client.post("/predict-image", files={"file": (name, data, ctype)})

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok", "trained": False})

    def test_predict_returns_documented_shape(self):
        r = self.post(png_bytes())
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(0.0 <= body["p"] <= 1.0)
        self.assertEqual(len(body["con"]), 64)
        self.assertFalse(body["trained"])
        self.assertIn("not a medical diagnosis", body["disclaimer"])

    def test_bad_uploads_are_rejected_with_clear_codes(self):
        self.assertEqual(self.post(b"").status_code, 400)
        self.assertEqual(self.post(b"not an image" * 40).status_code, 400)
        gif = io.BytesIO()
        Image.new("L", (64, 64)).save(gif, format="GIF")
        self.assertEqual(self.post(gif.getvalue(), "x.gif", "image/gif").status_code, 415)
        self.assertEqual(self.post(b"\0" * (8 * 1024 * 1024 + 10)).status_code, 413)

    def test_missing_file_field_is_a_validation_error(self):
        self.assertEqual(self.client.post("/predict-image").status_code, 422)

    def test_responses_are_not_cacheable_and_are_nosniff(self):
        r = self.client.get("/health")
        self.assertEqual(r.headers["cache-control"], "no-store")
        self.assertEqual(r.headers["x-content-type-options"], "nosniff")

    def test_rate_limit_returns_429_with_retry_after(self):
        from security import RateLimiter
        original = self.mod.limiter
        self.mod.limiter = RateLimiter(limit=2, window=60)
        try:
            codes = [self.post(png_bytes()).status_code for _ in range(3)]
        finally:
            self.mod.limiter = original
        self.assertEqual(codes, [200, 200, 429])

    def test_cors_only_allows_configured_origins(self):
        ok = self.client.get("/health", headers={"Origin": "http://localhost:8000"})
        bad = self.client.get("/health", headers={"Origin": "https://evil.example"})
        self.assertEqual(ok.headers.get("access-control-allow-origin"), "http://localhost:8000")
        self.assertIsNone(bad.headers.get("access-control-allow-origin"))


if __name__ == "__main__":
    unittest.main()

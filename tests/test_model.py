import os
import tempfile
import unittest

from PIL import Image

import _paths  # noqa: F401

try:
    import numpy as np
    import torch
    import torchvision  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "needs torch and torchvision")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import model
        cls.model_mod = model
        torch.manual_seed(0)
        cls.net, cls.trained = model.load_model("/nonexistent/model.pt")
        cls.img = Image.fromarray((np.random.default_rng(0).random((256, 256)) * 255).astype("uint8"))

    def test_missing_weights_fall_back_to_untrained(self):
        self.assertFalse(self.trained)

    def test_output_has_two_classes(self):
        out = self.net(torch.zeros(1, 3, 224, 224))
        self.assertEqual(tuple(out.shape), (1, 2))

    def test_predict_returns_probability_and_cam(self):
        p, cam = self.model_mod.predict(self.net, self.img)
        self.assertTrue(0.0 <= p <= 1.0)
        self.assertEqual(cam.shape, (224, 224))

    def test_gradcam_is_normalised(self):
        _, cam = self.model_mod.predict(self.net, self.img)
        self.assertGreaterEqual(float(cam.min()), 0.0)
        self.assertLessEqual(float(cam.max()), 1.0 + 1e-6)

    def test_predict_accepts_rgb_and_grayscale(self):
        for mode in ("L", "RGB"):
            p, _ = self.model_mod.predict(self.net, self.img.convert(mode))
            self.assertTrue(0.0 <= p <= 1.0)

    def test_saved_weights_round_trip_with_safe_loading(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            torch.save(self.net.state_dict(), path)
            net2, trained = self.model_mod.load_model(path)
        self.assertTrue(trained)
        x = torch.rand(1, 3, 224, 224)
        self.assertTrue(torch.allclose(self.net(x), net2(x), atol=1e-5))


if __name__ == "__main__":
    unittest.main()

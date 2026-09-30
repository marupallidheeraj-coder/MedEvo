import unittest

import numpy as np

import _paths  # noqa: F401
from heatmap import cam_to_grid


class CamToGridTests(unittest.TestCase):
    def test_returns_64_cells_for_default_grid(self):
        g = cam_to_grid(np.random.default_rng(0).random((224, 224)), 0.9)
        self.assertEqual(len(g), 64)

    def test_strong_peak_when_pneumonia_likely(self):
        g = cam_to_grid(np.random.default_rng(1).random((224, 224)), 0.9)
        self.assertAlmostEqual(max(g), 1.4, places=3)   # contract with the website's heat() renderer

    def test_faint_peak_when_pneumonia_unlikely(self):
        g = cam_to_grid(np.random.default_rng(1).random((224, 224)), 0.1)
        self.assertAlmostEqual(max(g), 0.4, places=3)

    def test_threshold_is_inclusive_at_half(self):
        g = cam_to_grid(np.ones((224, 224)), 0.5)
        self.assertAlmostEqual(max(g), 1.4, places=3)   # contract with the website's heat() renderer

    def test_hot_spot_lands_in_the_right_cell(self):
        cam = np.zeros((224, 224))
        cam[0:28, 0:28] = 1.0                      # top-left cell
        g = cam_to_grid(cam, 0.9)
        self.assertEqual(int(np.argmax(g)), 0)
        cam = np.zeros((224, 224))
        cam[196:224, 196:224] = 1.0                # bottom-right cell
        self.assertEqual(int(np.argmax(cam_to_grid(cam, 0.9))), 63)

    def test_all_zero_cam_gives_zeros_not_nan(self):
        g = cam_to_grid(np.zeros((224, 224)), 0.9)
        self.assertTrue(all(v == 0 for v in g))

    def test_values_are_never_negative(self):
        g = cam_to_grid(np.random.default_rng(2).random((224, 224)), 0.7)
        self.assertGreaterEqual(min(g), 0)

    def test_non_multiple_size_is_cropped(self):
        self.assertEqual(len(cam_to_grid(np.random.default_rng(3).random((230, 230)), 0.9)), 64)

    def test_custom_grid_size(self):
        self.assertEqual(len(cam_to_grid(np.ones((224, 224)), 0.9, n=4)), 16)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            cam_to_grid(np.ones((224, 100)), 0.5)          # not square
        with self.assertRaises(ValueError):
            cam_to_grid(np.ones(224), 0.5)                 # not 2-D
        with self.assertRaises(ValueError):
            cam_to_grid(np.ones((224, 224)), 1.5)          # not a probability
        with self.assertRaises(ValueError):
            cam_to_grid(np.ones((4, 4)), 0.5)              # smaller than the grid


if __name__ == "__main__":
    unittest.main()

import importlib.util
import unittest

import numpy as np
import pandas as pd

from _paths import DATA_DIR

spec = importlib.util.spec_from_file_location("make_clinical", DATA_DIR / "make_clinical.py")
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)

FEATURES = [n for n, *_ in mc.NUM] + [n for n, _ in mc.BIN]


class CommittedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = pd.read_csv(DATA_DIR / "clinical_train_800.csv")
        cls.demo = pd.read_csv(DATA_DIR / "patients_50.csv")

    def test_shapes_and_columns(self):
        self.assertEqual(self.train.shape, (800, 19))
        self.assertEqual(self.demo.shape, (50, 20))
        self.assertEqual(list(self.train.columns), FEATURES + ["diagnosis"])
        self.assertEqual(list(self.demo.columns), ["patient_id"] + FEATURES + ["diagnosis"])

    def test_no_missing_values(self):
        self.assertFalse(self.train.isna().any().any())
        self.assertFalse(self.demo.isna().any().any())

    def test_labels_and_flags_are_binary(self):
        binary = [n for n, _ in mc.BIN] + ["diagnosis"]
        for df in (self.train, self.demo):
            for col in binary:
                self.assertTrue(set(df[col].unique()) <= {0, 1}, col)

    def test_numeric_values_stay_in_physiological_bounds(self):
        for name, _m, _s, lo, hi in mc.NUM:
            for df in (self.train, self.demo):
                self.assertGreaterEqual(df[name].min(), lo, name)
                self.assertLessEqual(df[name].max(), hi, name)

    def test_patient_ids_unique_and_ordered(self):
        self.assertEqual(self.demo.patient_id.tolist(), [f"P{1001 + i}" for i in range(50)])

    def test_prevalence_is_plausible(self):
        self.assertAlmostEqual(self.train.diagnosis.mean(), 0.40, delta=0.06)
        self.assertGreater(self.demo.diagnosis.sum(), 10)
        self.assertLess(self.demo.diagnosis.sum(), 40)

    def test_clinical_signal_points_the_right_way(self):
        g = self.train.groupby("diagnosis").mean()
        self.assertLess(g.loc[1, "spo2"], g.loc[0, "spo2"])          # lower oxygen in pneumonia
        for col in ("temp", "crp", "wbc", "rr", "fever", "dyspnea"):
            self.assertGreater(g.loc[1, col], g.loc[0, col], col)


class GeneratorTests(unittest.TestCase):
    def test_same_seed_gives_identical_data(self):
        pd.testing.assert_frame_equal(mc.make(100, 0.4, 5), mc.make(100, 0.4, 5))

    def test_different_seed_gives_different_data(self):
        self.assertFalse(mc.make(100, 0.4, 5).equals(mc.make(100, 0.4, 6)))

    def test_requested_size_and_prevalence(self):
        df = mc.make(2000, 0.3, 1)
        self.assertEqual(len(df), 2000)
        self.assertAlmostEqual(df.diagnosis.mean(), 0.3, delta=0.04)

    def test_label_is_last_column(self):
        self.assertEqual(mc.make(10, 0.5, 1).columns[-1], "diagnosis")

    def test_values_respect_bounds(self):
        df = mc.make(500, 0.5, 3)
        for name, _m, _s, lo, hi in mc.NUM:
            self.assertTrue(df[name].between(lo, hi).all(), name)
        self.assertTrue(np.isfinite(df.to_numpy()).all())


if __name__ == "__main__":
    unittest.main()

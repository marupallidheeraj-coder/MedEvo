import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from _paths import DATA_DIR
from clinical_pipeline import (bootstrap_auc_ci, evaluate, expected_calibration_error,
                               ga_select, linear_contributions, load_clinical)

STRONG_SIGNALS = {"temp", "spo2", "rr", "wbc", "crp", "fever", "dyspnea", "sputum"}


class LoadTests(unittest.TestCase):
    def write(self, df):
        f = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
        df.to_csv(f.name, index=False)
        self.addCleanup(Path(f.name).unlink)
        return f.name

    def test_loads_committed_dataset(self):
        X, y = load_clinical(DATA_DIR / "clinical_train_800.csv")
        self.assertEqual(X.shape, (800, 18))
        self.assertEqual(len(y), 800)

    def test_drops_patient_id(self):
        X, _ = load_clinical(DATA_DIR / "patients_50.csv")
        self.assertNotIn("patient_id", X.columns)

    def test_rejects_missing_target(self):
        with self.assertRaisesRegex(ValueError, "diagnosis"):
            load_clinical(self.write(pd.DataFrame({"a": [1, 2]})))

    def test_rejects_missing_values(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            load_clinical(self.write(pd.DataFrame({"a": [1, None], "diagnosis": [0, 1]})))

    def test_rejects_non_binary_labels(self):
        with self.assertRaisesRegex(ValueError, "0 and 1"):
            load_clinical(self.write(pd.DataFrame({"a": [1, 2], "diagnosis": [0, 2]})))


class GeneticAlgorithmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = load_clinical(DATA_DIR / "clinical_train_800.csv")
        cls.cols = list(X.columns)
        cls.S = StandardScaler().fit_transform(X)
        cls.y = y

    def run_ga(self, seed=0):
        return ga_select(self.S, self.y, pop=10, gens=6, seed=seed)

    def test_returns_valid_non_empty_mask(self):
        mask, _ = self.run_ga()
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(len(mask), self.S.shape[1])
        self.assertTrue(mask.any())

    def test_same_seed_is_reproducible(self):
        m1, h1 = self.run_ga(3)
        m2, h2 = self.run_ga(3)
        self.assertTrue((m1 == m2).all())
        self.assertEqual(h1, h2)

    def test_elitism_means_best_fitness_never_drops(self):
        _, history = self.run_ga()
        self.assertEqual(len(history), 6)
        self.assertTrue(all(b >= a - 1e-12 for a, b in zip(history, history[1:])))

    def test_selects_real_clinical_signals_and_a_smaller_subset(self):
        mask, _ = self.run_ga()
        chosen = {c for c, m in zip(self.cols, mask) if m}
        self.assertTrue(chosen & STRONG_SIGNALS)
        self.assertLess(mask.sum(), len(mask))

    def test_penalty_shrinks_the_subset(self):
        big, _ = ga_select(self.S, self.y, pop=10, gens=6, seed=1, penalty=0.0)
        small, _ = ga_select(self.S, self.y, pop=10, gens=6, seed=1, penalty=0.05)
        self.assertLessEqual(small.sum(), big.sum())


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = load_clinical(DATA_DIR / "clinical_train_800.csv")
        cls.report = evaluate(X, y, seed=0, pop=10, gens=6, n_boot=100)

    def test_report_has_expected_fields(self):
        for key in ("auc", "auc_ci95", "sensitivity", "specificity", "brier", "ece", "selected_features"):
            self.assertIn(key, self.report)

    def test_held_out_performance_is_good_but_not_perfect(self):
        self.assertGreater(self.report["auc"], 0.85)
        self.assertLess(self.report["auc"], 1.0)

    def test_ga_subset_is_close_to_using_every_feature(self):
        self.assertGreater(self.report["auc"], self.report["auc_all_features"] - 0.03)

    def test_confidence_interval_brackets_the_estimate(self):
        lo, hi = self.report["auc_ci95"]
        self.assertLessEqual(lo, self.report["auc"])
        self.assertGreaterEqual(hi, self.report["auc"])

    def test_confusion_matrix_adds_up(self):
        cm = self.report["confusion_matrix"]
        self.assertEqual(sum(cm.values()), self.report["n_test"])
        self.assertEqual(self.report["n_train"] + self.report["n_test"], 800)

    def test_rates_are_probabilities(self):
        for k in ("sensitivity", "specificity", "brier", "ece", "accuracy"):
            self.assertTrue(0.0 <= self.report[k] <= 1.0, k)


class ExplanationTests(unittest.TestCase):
    def test_contributions_add_up_to_the_models_log_odds(self):
        X, y = load_clinical(DATA_DIR / "clinical_train_800.csv")
        S = StandardScaler().fit_transform(X)
        clf = LogisticRegression(max_iter=1000).fit(S, y)
        for i in (0, 10, 500):
            base, contrib = linear_contributions(clf, S[i])
            self.assertAlmostEqual(base + contrib.sum(), clf.decision_function(S[i:i + 1])[0], places=8)

    def test_zero_input_gives_only_the_baseline(self):
        clf = LogisticRegression().fit(np.array([[0.0], [1.0], [2.0], [3.0]]), [0, 0, 1, 1])
        base, contrib = linear_contributions(clf, np.array([0.0]))
        self.assertEqual(contrib.sum(), 0.0)
        self.assertEqual(base, float(clf.intercept_[0]))


class MetricTests(unittest.TestCase):
    def test_ece_is_zero_for_perfectly_calibrated_extremes(self):
        self.assertEqual(expected_calibration_error([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0]), 0.0)

    def test_ece_detects_overconfidence(self):
        self.assertGreater(expected_calibration_error([0, 0, 0, 0], [0.9, 0.9, 0.9, 0.9]), 0.8)

    def test_bootstrap_interval_is_ordered_and_reproducible(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        p = np.clip(y * 0.5 + rng.random(200) * 0.5, 0, 1)
        lo, hi = bootstrap_auc_ci(y, p, n_boot=200, seed=1)
        self.assertLess(lo, hi)
        self.assertEqual((lo, hi), bootstrap_auc_ci(y, p, n_boot=200, seed=1))


if __name__ == "__main__":
    unittest.main()

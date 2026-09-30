"""Clinical model, Genetic-Algorithm feature selection and honest validation.

Needs only NumPy, pandas and scikit-learn (no GPU, no X-rays), so it runs anywhere and is
covered by the test-suite. It mirrors what the website does in the browser.

  python clinical_pipeline.py --data ../data/clinical_train_800.csv --out ../docs/validation_report.json

Validation protocol
  1. Stratified 75/25 split. The test quarter is never used for feature selection or fitting.
  2. The GA scores candidate feature subsets with cross-validated AUC on the TRAIN part only.
  3. The final model is scored once on the held-out test part: AUC with a bootstrap 95% CI,
     sensitivity, specificity, Brier score and expected calibration error.
  4. The GA-selected model is compared with a model that uses every feature.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "diagnosis"
ID_COL = "patient_id"


def load_clinical(path) -> tuple[pd.DataFrame, np.ndarray]:
    """Read a clinical CSV and check it is usable. Returns (features, labels)."""
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"column '{TARGET}' is missing")
    if df.isna().any().any():
        raise ValueError("the data contain missing values")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError(f"'{TARGET}' must contain only 0 and 1")
    X = df.drop(columns=[c for c in (TARGET, ID_COL) if c in df.columns])
    if X.shape[1] == 0:
        raise ValueError("no feature columns")
    return X, df[TARGET].to_numpy()


def ga_select(X, y, pop: int = 20, gens: int = 15, seed: int = 0, penalty: float = 0.006,
              mutation: float = 0.04, cv: int = 3):
    """Genetic algorithm over feature subsets.

    Fitness = cross-validated AUC of a logistic regression, minus `penalty` per feature (prefers
    small subsets). Uses tournament selection, uniform crossover, bit-flip mutation and elitism.
    Returns (boolean mask, list with the best fitness of every generation).
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    n_feat = X.shape[1]
    folds = StratifiedKFold(cv, shuffle=True, random_state=seed)
    cache: dict[bytes, float] = {}

    def fitness(mask: np.ndarray) -> float:
        key = mask.tobytes()
        if key not in cache:
            auc = cross_val_score(LogisticRegression(max_iter=500), X[:, mask], y, cv=folds,
                                  scoring="roc_auc").mean()
            cache[key] = float(auc - penalty * mask.sum())
        return cache[key]

    def fix(mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            mask[rng.integers(n_feat)] = True
        return mask

    population = [fix(rng.random(n_feat) < 0.3) for _ in range(pop)]
    history: list[float] = []
    for g in range(gens):
        ranked = sorted(population, key=fitness, reverse=True)
        history.append(fitness(ranked[0]))
        if g == gens - 1:
            break
        nxt = [ranked[0].copy(), ranked[1].copy()] if pop > 1 else [ranked[0].copy()]

        def tournament() -> np.ndarray:
            a, b = ranked[rng.integers(pop)], ranked[rng.integers(pop)]
            return a if fitness(a) >= fitness(b) else b

        while len(nxt) < pop:
            p1, p2 = tournament(), tournament()
            child = np.where(rng.random(n_feat) < 0.5, p1, p2)
            child = np.where(rng.random(n_feat) < mutation, ~child, child)
            nxt.append(fix(child.astype(bool)))
        population = nxt
    best = max(population, key=fitness)
    return best.astype(bool), history


def linear_contributions(clf: LogisticRegression, x_scaled: np.ndarray):
    """Exact SHAP values for a logistic regression on standardised features.

    With a zero-mean background the contribution of feature i is coef_i * x_i, so that
    base + sum(contributions) equals the model's log-odds for the patient.
    """
    x_scaled = np.asarray(x_scaled, dtype=float).ravel()
    contrib = clf.coef_.ravel() * x_scaled
    return float(clf.intercept_[0]), contrib


def bootstrap_auc_ci(y, p, n_boot: int = 500, seed: int = 0, level: float = 0.95):
    """Percentile bootstrap confidence interval for the AUC."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(aucs, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def expected_calibration_error(y, p, bins: int = 10) -> float:
    """Weighted mean gap between predicted probability and observed frequency, 0 = perfect."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = which == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def evaluate(X: pd.DataFrame, y, seed: int = 0, pop: int = 20, gens: int = 15, n_boot: int = 500) -> dict:
    """Run the full validation protocol and return a JSON-friendly report."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    scaler = StandardScaler().fit(Xtr)
    Str, Ste = scaler.transform(Xtr), scaler.transform(Xte)

    mask, history = ga_select(Str, ytr, pop=pop, gens=gens, seed=seed)
    clf = LogisticRegression(max_iter=1000).fit(Str[:, mask], ytr)
    p = clf.predict_proba(Ste[:, mask])[:, 1]
    base = LogisticRegression(max_iter=1000).fit(Str, ytr)
    p_all = base.predict_proba(Ste)[:, 1]

    tn, fp, fn, tp = confusion_matrix(yte, p >= 0.5, labels=[0, 1]).ravel()
    lo, hi = bootstrap_auc_ci(yte, p, n_boot=n_boot, seed=seed)
    return {
        "n_train": int(len(ytr)), "n_test": int(len(yte)), "prevalence": float(np.mean(y)),
        "selected_features": [c for c, m in zip(X.columns, mask) if m],
        "n_selected": int(mask.sum()), "n_available": int(len(mask)),
        "ga_best_fitness_per_generation": [round(h, 4) for h in history],
        "auc": float(roc_auc_score(yte, p)), "auc_ci95": [lo, hi],
        "auc_all_features": float(roc_auc_score(yte, p_all)),
        "accuracy": float(((p >= 0.5).astype(int) == np.asarray(yte)).mean()),
        "sensitivity": float(tp / max(tp + fn, 1)), "specificity": float(tn / max(tn + fp, 1)),
        "brier": float(brier_score_loss(yte, p)),
        "ece": expected_calibration_error(yte, p),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "seed": seed,
        "caveat": "Clinical data are synthetic and generated from the label, so these numbers are optimistic "
                  "and say nothing about real-world performance.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="../data/clinical_train_800.csv")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    X, y = load_clinical(a.data)
    report = evaluate(X, y, seed=a.seed)
    print(json.dumps(report, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2))
        print("saved", a.out)


if __name__ == "__main__":
    main()

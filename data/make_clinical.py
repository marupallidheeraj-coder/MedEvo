"""Generate the SYNTHETIC clinical datasets used by MedFusion-Evo.

Mirrors the generator inside frontend/index.html (same 18 variables and class
distributions). No public dataset links chest X-rays to real patient vitals, so
these values are simulated from the diagnosis label. Demo use only.

  python make_clinical.py          -> clinical_train_800.csv, patients_50.csv
"""
import numpy as np
import pandas as pd

# name, kind, (mean_normal, mean_pneumonia) or (p_normal, p_pneumonia), sd, min, max
NUM = [
    ("age", (46, 56), 17, 18, 95), ("temp", (36.9, 38.1), .7, 35, 41.5),
    ("spo2", (97, 92), 2.6, 75, 100), ("hr", (78, 92), 13, 45, 170),
    ("rr", (16, 22), 3.5, 8, 45), ("cough_days", (2, 8), 5, 0, 40),
    ("wbc", (7.5, 12), 3, 2, 30), ("crp", (12, 70), 35, 0, 300),
    ("bmi", (26, 26), 5, 14, 50), ("sbp", (126, 124), 17, 70, 210),
]
BIN = [
    ("fever", (.15, .85)), ("dyspnea", (.12, .70)), ("sputum", (.12, .65)),
    ("smoker", (.28, .45)), ("chest_pain", (.12, .40)), ("male", (.5, .5)),
    ("headache", (.3, .3)), ("diabetes", (.15, .15)),
]


def make(n, p_pneumonia, seed):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < p_pneumonia).astype(int)
    df = pd.DataFrame({"diagnosis": y})
    for name, m, s, lo, hi in NUM:
        v = np.array([m[d] for d in y]) + rng.standard_normal(n) * s * 1.9
        v = np.clip(v, lo, hi)
        df[name] = np.round(v, 1) if s < 1 else np.round(v).astype(int)
    for name, p in BIN:
        prob = np.array([0.5 + (p[d] - 0.5) * 0.75 for d in y])
        df[name] = (rng.random(n) < prob).astype(int)
    return df[[c for c in df.columns if c != "diagnosis"] + ["diagnosis"]]


if __name__ == "__main__":
    make(800, 0.40, 7).to_csv("clinical_train_800.csv", index=False)
    p = make(50, 0.50, 2024)
    p.insert(0, "patient_id", [f"P{1001 + i}" for i in range(50)])
    p.to_csv("patients_50.csv", index=False)
    print("wrote clinical_train_800.csv and patients_50.csv")

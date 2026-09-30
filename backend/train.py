"""Train the ResNet18 chest X-ray classifier and save backend/model.pt.

Usage (after downloading the dataset, see data/README.md):
  python train.py --data ../chest_xray --epochs 3 --n-train 2000
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from model import TF, build_model

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../chest_xray")
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--n-train", type=int, default=2000, help="max training images (use fewer on CPU)")
ap.add_argument("--n-test", type=int, default=624)
ap.add_argument("--out", default="model.pt")
ap.add_argument("--metrics-out", default="metrics.json", help="held-out test metrics (JSON)")
a = ap.parse_args()

root = a.data
if os.path.isdir(os.path.join(root, "chest_xray")):      # handle nested zip folder
    root = os.path.join(root, "chest_xray")

random.seed(42); np.random.seed(42); torch.manual_seed(42)
dev = "cuda" if torch.cuda.is_available() else "cpu"
train_ds = datasets.ImageFolder(os.path.join(root, "train"), TF)   # NORMAL=0, PNEUMONIA=1
test_ds = datasets.ImageFolder(os.path.join(root, "test"), TF)
rng = np.random.default_rng(42)
tr = rng.permutation(len(train_ds))[: a.n_train]
te = rng.permutation(len(test_ds))[: a.n_test]
ytr = np.array([train_ds.targets[i] for i in tr])

model = build_model(pretrained=True).to(dev)
w = torch.tensor(len(ytr) / (2 * np.bincount(ytr)), dtype=torch.float32).to(dev)   # class balance
lossf = nn.CrossEntropyLoss(weight=w)
opt = torch.optim.Adam(model.parameters(), 1e-4)
loader = DataLoader(Subset(train_ds, tr), batch_size=32, shuffle=True)

for ep in range(a.epochs):
    model.train(); tot = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(); loss = lossf(model(x), y); loss.backward(); opt.step()
        tot += loss.item() * len(y)
    print(f"epoch {ep + 1}/{a.epochs}  loss {tot / len(tr):.4f}")

model.eval(); probs, ys = [], []
with torch.no_grad():
    for x, y in DataLoader(Subset(test_ds, te), batch_size=64):
        probs += torch.softmax(model(x.to(dev)), 1)[:, 1].cpu().tolist(); ys += y.tolist()
probs, ys = np.array(probs), np.array(ys)
pred = probs > .5
tn, fp, fn, tp = confusion_matrix(ys, pred, labels=[0, 1]).ravel()
metrics = {
    "n_test": int(len(ys)), "accuracy": float(accuracy_score(ys, pred)), "auc": float(roc_auc_score(ys, probs)),
    "sensitivity": float(tp / max(tp + fn, 1)), "specificity": float(tn / max(tn + fp, 1)),
    "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "epochs": a.epochs, "n_train": int(len(tr)),
}
print(f"TEST accuracy {metrics['accuracy']:.3f}  AUC {metrics['auc']:.3f}  "
      f"sensitivity {metrics['sensitivity']:.3f}  specificity {metrics['specificity']:.3f}")
with open(a.metrics_out, "w") as fh:
    json.dump(metrics, fh, indent=2)
print("saved", a.metrics_out)

torch.save(model.cpu().state_dict(), a.out)
print("saved", a.out)

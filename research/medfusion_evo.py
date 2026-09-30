"""
MedFusion-Evo: explainable multimodal pneumonia screening (demo)
CNN (ResNet18) + Genetic-Algorithm feature selection + fusion MLP + Grad-CAM + SHAP

SETUP
  pip install torch torchvision scikit-learn shap matplotlib pandas pillow kaggle
  Dataset (X-rays): https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
     kaggle datasets download -d paultimothymooney/chest-xray-pneumonia --unzip -p .
     (needs kaggle.json from kaggle.com > Settings > API > Create New Token,
      placed in ~/.kaggle/ or C:\\Users\\<you>\\.kaggle\\)
  Result: folder  chest_xray/train/{NORMAL,PNEUMONIA}  and  chest_xray/test/{NORMAL,PNEUMONIA}

RUN
  python medfusion_evo.py      -> prints metrics, saves report.png

NOTE: no public dataset links X-rays to real patient vitals, so the clinical
data below is SYNTHETIC (generated from the label). Demo only, not validated.
"""
import os, random
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import shap

# ---------------- config ----------------
SEED, N_TRAIN, N_TEST, EPOCHS = 42, 1200, 400, 3   # lower N_TRAIN/EPOCHS on slow CPUs
ROOT = "chest_xray"
if os.path.isdir(os.path.join(ROOT, "chest_xray")):
    ROOT = os.path.join(ROOT, "chest_xray")
dev = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

# ---------------- images ----------------
tf = transforms.Compose([
    transforms.Grayscale(3), transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train_ds = datasets.ImageFolder(os.path.join(ROOT, "train"), tf)   # NORMAL=0, PNEUMONIA=1
test_ds = datasets.ImageFolder(os.path.join(ROOT, "test"), tf)
tr_idx = rng.permutation(len(train_ds))[:N_TRAIN]
te_idx = rng.permutation(len(test_ds))[:N_TEST]
ytr = np.array([train_ds.targets[i] for i in tr_idx])
yte = np.array([test_ds.targets[i] for i in te_idx])

# ---------------- synthetic clinical data ----------------
def make_clinical(y):
    n = len(y)
    return pd.DataFrame({
        "Age": rng.normal(45 + 10 * y, 15).clip(1, 95),
        "Sex(1=M)": rng.integers(0, 2, n),
        "Temperature": rng.normal(36.9 + 1.3 * y, 0.7),
        "HeartRate": rng.normal(80 + 14 * y, 12),
        "RespRate": rng.normal(16 + 7 * y, 3.5),
        "SpO2": rng.normal(97 - 4 * y, 2.2).clip(70, 100),
        "Cough": rng.binomial(1, 0.25 + 0.5 * y),
        "BreathingDiff": rng.binomial(1, 0.10 + 0.5 * y),
        "Diabetes": rng.binomial(1, 0.15 + 0.05 * y),
        "Smoking": rng.binomial(1, 0.20 + 0.05 * y)})
clin_tr, clin_te = make_clinical(ytr), make_clinical(yte)
cols = list(clin_tr.columns)
sc_c = StandardScaler().fit(clin_tr)
Ctr, Cte = sc_c.transform(clin_tr), sc_c.transform(clin_te)

# ---------------- 1. train CNN ----------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, 2)
model.to(dev)
cw = torch.tensor(len(ytr) / (2 * np.bincount(ytr)), dtype=torch.float32).to(dev)
lossf = nn.CrossEntropyLoss(weight=cw)
opt = torch.optim.Adam(model.parameters(), 1e-4)
loader = DataLoader(Subset(train_ds, tr_idx), batch_size=32, shuffle=True)
for ep in range(EPOCHS):
    model.train(); tot = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(); loss = lossf(model(x), y); loss.backward(); opt.step()
        tot += loss.item() * len(y)
    print(f"CNN epoch {ep+1}/{EPOCHS}  loss {tot/len(tr_idx):.4f}")

@torch.no_grad()
def extract(ds, idx):
    model.eval()
    backbone = nn.Sequential(*list(model.children())[:-1])
    feats, probs = [], []
    for x, _ in DataLoader(Subset(ds, idx), batch_size=64):
        f = backbone(x.to(dev)).flatten(1)
        feats.append(f.cpu()); probs.append(F.softmax(model.fc(f), 1)[:, 1].cpu())
    return torch.cat(feats).numpy(), torch.cat(probs).numpy()

Ftr, _ = extract(train_ds, tr_idx)
Fte, p_img = extract(test_ds, te_idx)
sc_f = StandardScaler().fit(Ftr)
Ftr, Fte = sc_f.transform(Ftr), sc_f.transform(Fte)

# ---------------- 2. Genetic Algorithm feature selection ----------------
def ga_select(X, y, pop=20, gens=15):
    nf = X.shape[1]
    def fit(m):
        if m.sum() == 0: return 0
        auc = cross_val_score(LogisticRegression(max_iter=500), X[:, m.astype(bool)], y,
                              cv=3, scoring="roc_auc").mean()
        return auc - 0.002 * m.sum()
    P = rng.integers(0, 2, (pop, nf))
    for g in range(gens):
        f = np.array([fit(m) for m in P]); P = P[np.argsort(-f)]
        new = [P[0], P[1]]                                   # elitism
        while len(new) < pop:
            a, b = P[rng.integers(0, pop // 2, 2)]
            cut = rng.integers(1, nf)
            c = np.concatenate([a[:cut], b[cut:]])           # crossover
            c = np.where(rng.random(nf) < 0.1, 1 - c, c)     # mutation
            new.append(c)
        P = np.array(new)
    return P[int(np.argmax([fit(m) for m in P]))].astype(bool)

mask = ga_select(Ctr, ytr)
sel = [c for c, m in zip(cols, mask) if m]
print("GA-selected clinical features:", sel)
Ctr_s, Cte_s = Ctr[:, mask], Cte[:, mask]

# ---------------- 3. clinical model (for SHAP) ----------------
clin_clf = LogisticRegression(max_iter=1000).fit(Ctr_s, ytr)
p_clin = clin_clf.predict_proba(Cte_s)[:, 1]

# ---------------- 4. fusion network ----------------
Xtr = torch.tensor(np.hstack([Ftr, Ctr_s]), dtype=torch.float32)
Xte = torch.tensor(np.hstack([Fte, Cte_s]), dtype=torch.float32)
fusion = nn.Sequential(nn.Linear(Xtr.shape[1], 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))
opt2 = torch.optim.Adam(fusion.parameters(), 1e-3, weight_decay=1e-3)
cw2 = cw.cpu()
for ep in range(80):
    fusion.train(); opt2.zero_grad()
    nn.CrossEntropyLoss(weight=cw2)(fusion(Xtr), torch.tensor(ytr)).backward(); opt2.step()
fusion.eval()
with torch.no_grad():
    p_fus = F.softmax(fusion(Xte), 1)[:, 1].numpy()

print("\n=== TEST METRICS (synthetic clinical data!) ===")
for name, p in [("Image only", p_img), ("Clinical only", p_clin), ("Fusion", p_fus)]:
    print(f"{name:14s} acc {accuracy_score(yte, p > .5):.3f}  AUC {roc_auc_score(yte, p):.3f}")

# ---------------- 5. Grad-CAM ----------------
def gradcam(x, cls=1):
    acts, grads = [], []
    def fwd(m, i, o):
        acts.append(o); o.register_hook(lambda g: grads.append(g))
    h = model.layer4.register_forward_hook(fwd)
    model.eval(); model.zero_grad()
    out = model(x.unsqueeze(0).to(dev)); out[0, cls].backward()
    h.remove()
    w = grads[0].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * acts[0]).sum(1, keepdim=True)).detach()
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear")[0, 0].cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

# ---------------- 6. report for one test patient ----------------
k = int(np.argmax(p_fus))                       # most confident pneumonia case
img_tensor, _ = test_ds[te_idx[k]]
path = test_ds.samples[te_idx[k]][0]
raw = np.array(Image.open(path).convert("L").resize((224, 224)))
cam = gradcam(img_tensor)

sv = shap.Explainer(clin_clf, Ctr_s, feature_names=sel)(Cte_s[k:k + 1])
vals = sv.values[0]
if vals.ndim > 1: vals = vals[:, -1]
order = np.argsort(np.abs(vals))

risk = p_fus[k]
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("MEDFUSION-EVO  |  Explainable Multimodal Pneumonia AI (AI-assisted screening)",
             fontsize=15, fontweight="bold")
ax[0, 0].imshow(raw, cmap="gray"); ax[0, 0].set_title("Original X-ray"); ax[0, 0].axis("off")
ax[0, 1].imshow(raw, cmap="gray"); ax[0, 1].imshow(cam, cmap="jet", alpha=0.4)
ax[0, 1].set_title("Grad-CAM (image model attention)"); ax[0, 1].axis("off")
ax[0, 2].axis("off")
ax[0, 2].text(0, 0.9, "OVERALL AI-ASSISTED SCREENING", fontsize=13, fontweight="bold")
ax[0, 2].text(0, 0.68, f"Pneumonia risk: {'ELEVATED' if risk >= .5 else 'LOW'}  ({risk*100:.0f}%)",
              fontsize=15, color="crimson" if risk >= .5 else "green")
ax[0, 2].text(0, 0.48, f"Image model confidence: {p_img[k]*100:.0f}%", fontsize=12)
ax[0, 2].text(0, 0.34, f"Clinical model probability: {p_clin[k]*100:.0f}%", fontsize=12)
ax[0, 2].text(0, 0.14, "Final prediction uses both image\nand clinical representations.", fontsize=10)
ax[1, 0].axis("off"); ax[1, 0].set_title("Patient clinical data (synthetic)")
row = clin_te.iloc[k]
for i, c in enumerate(cols):
    v = row[c]
    ax[1, 0].text(0.02, 0.95 - i * 0.09, f"{c:15s} {v:.1f}" if c in ("Age", "Temperature", "HeartRate", "RespRate", "SpO2")
                  else f"{c:15s} {'Yes' if v else 'No'}", fontsize=11, family="monospace")
ax[1, 1].barh([sel[i] for i in order], vals[order], color=["crimson" if v > 0 else "steelblue" for v in vals[order]])
ax[1, 1].set_title("SHAP: clinical factors (red = raises risk)")
ax[1, 2].axis("off"); ax[1, 2].set_title("Evolutionary optimisation & metrics")
ax[1, 2].text(0, 0.85, f"GA-selected features: {len(sel)}/{len(cols)}", fontsize=12)
ax[1, 2].text(0, 0.72, ", ".join(sel), fontsize=9, wrap=True)
ax[1, 2].text(0, 0.45, f"Test AUC  image {roc_auc_score(yte, p_img):.2f} | "
              f"clinical {roc_auc_score(yte, p_clin):.2f} | fusion {roc_auc_score(yte, p_fus):.2f}", fontsize=10)
fig.text(0.5, 0.01, "AI-assisted screening - not a medical diagnosis. Clinical data are synthetic (demo).",
         ha="center", fontsize=10, style="italic")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("report.png", dpi=150)
print("\nSaved report.png")

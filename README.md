# MedFusion-Evo

Explainable multimodal AI for **pneumonia screening**: chest X-ray + patient clinical data.

- **Image model:** ResNet18 CNN with **Grad-CAM** heatmaps (where in the X-ray the model looked)
- **Clinical model:** logistic regression on features chosen by a **Genetic Algorithm**, explained with **SHAP-style** per-feature effects
- **Fusion:** the X-ray score and clinical risk are combined into one final risk score

> **AI-assisted screening demo, not a medical diagnosis.** Clinical data are synthetic
> (no public dataset links X-rays to real patient vitals). Not clinically validated.

## The problem

Pneumonia is the leading infectious cause of death in young children, with about 740,000 deaths under age five in
2019 ([Karim et al., Cureus 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC9928894)). Where a radiologist is not
close by, the diagnosis leans on clinical signs that are not specific: in a multicentre Pakistani study of children
diagnosed with non-severe pneumonia on the WHO clinical definition, only about 14% had radiological evidence of it
([BMJ editorial, 2006](https://pmc.ncbi.nlm.nih.gov/articles/PMC1570801)).
A good screening aid therefore has to (1) use **both** the X-ray and the bedside data, (2) ask for as **little data entry as
possible**, (3) **explain** itself so a clinician can disagree with it, and (4) **never let a confident-looking score
hide a dangerous patient**.

## How MedFusion-Evo answers it

| Need | What the project does |
|---|---|
| Use image and bedside data together | ResNet18 X-ray score and a clinical risk model are fused into one risk score |
| Less data entry | A Genetic Algorithm keeps 8 of 18 variables for almost the same accuracy (AUC 0.961 vs 0.967 with all 18) |
| Explain the result | Grad-CAM shows *where* in the X-ray the model looked. Exact linear SHAP values show *why* on the clinical side, in plain language |
| Do not hide danger | Rule-based **red flags** (SpO₂ < 90%, breathing rate ≥ 30/min, systolic BP < 90) force an **Urgent** action whatever the AI score says |
| Know when it is unsure | A warning appears when the X-ray model and the clinical model disagree |
| Let people probe it | A **what-if explorer** shows how the risk moves when a value changes |
| Protect privacy | The clinical model runs in the browser. The optional backend never stores or logs images |
| Be honest | Model card, validation report with confidence intervals, and clear "synthetic data" labelling |

Users: clinicians and health workers deciding who to assess first, plus students and reviewers exploring explainable AI.

## Project structure

```
medfusion-evo/
├── frontend/index.html          Website (single file, no build step)
├── backend/                     FastAPI server: real ResNet18 + Grad-CAM for uploaded X-rays
│   ├── app.py  security.py  model.py  heatmap.py  train.py  requirements.txt  Dockerfile
├── data/                        Datasets + generators
│   ├── clinical_train_800.csv   Synthetic clinical training set (800 rows)
│   ├── patients_50.csv          Synthetic 50-patient set
│   ├── make_clinical.py         Regenerates the CSVs
│   └── download_data.py         Downloads the Kaggle chest X-ray dataset
├── research/                    Full Python pipeline + Colab notebook (CNN + GA + fusion + Grad-CAM + SHAP)
│   └── clinical_pipeline.py     GA + clinical model + validation (no GPU needed)
├── tests/                       Python + JavaScript tests (see "Testing")
├── docs/                        MODEL_CARD.md  ARCHITECTURE.md  VALIDATION.md  validation_report.json
├── .github/workflows/ci.yml     Lint, tests and Docker build on every push
├── SECURITY.md  CONTRIBUTING.md  LICENSE  Makefile  pyproject.toml  .env.example
└── vercel.json                  Serves ./frontend with security headers
```

## How the website works

| Mode | What runs |
|---|---|
| **Select randomly from dataset** | Picks one of 50 patients. The X-ray is one of 8 built-in sample X-rays (4 normal, 4 pneumonia). The clinical model and Genetic Algorithm run in your browser. The sample X-rays use **preset** scores and heatmap regions (illustrative). |
| **Upload data** | Upload your own X-ray and enter the GA-selected clinical values. With no backend, an in-browser demo image model scores it (unreliable on real X-rays). With the backend connected, the trained ResNet18 + real Grad-CAM score it. |

Every result also shows a suggested action (Urgent / Evaluate soon / Clinical review / Routine), the red flags found,
a warning if the two models disagree, a what-if explorer, and a JSON report you can download or print. In upload mode you
can add optional SpO₂, breathing rate and blood pressure so the red-flag check works even when the GA did not select them.

## Run locally

**Frontend only:** open `frontend/index.html` in a browser.

**With the backend:**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```
Then set `var API_URL='http://localhost:8000';` near the top of the script in `frontend/index.html`.
Without `model.pt` the server runs with untrained weights (`/health` shows `"trained": false`).

## Testing

```bash
make install     # dev dependencies
make test        # 71 Python tests + 22 JavaScript tests
make lint        # ruff
make validate    # clinical model + GA validation report
```

- **Python:** data checks, GA behaviour, exact-SHAP additivity, upload security, rate limiting, Grad-CAM pooling, and
  (when PyTorch/FastAPI are installed) the model and the API.
- **JavaScript:** unit tests for the fusion, red-flag, triage and validation rules, plus a smoke test that runs the
  whole page on a fake DOM.
- **CI** runs all of it on every push, plus `ruff` and a Docker build.

Results and method: [docs/VALIDATION.md](docs/VALIDATION.md). Intended use and limits: [docs/MODEL_CARD.md](docs/MODEL_CARD.md).
Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Accessibility

Skip link, labelled inputs with plain-language error messages, table captions, descriptive image text, a status message
that announces each result to screen readers, focus moved to the result after a user action, no colour-only meaning,
dark mode, respect for "reduce motion", and a print view.

## Security

Upload limits and content checks, rate limiting, CORS allow-list, safe model loading, no image storage, non-root
container, security headers. Details and known limits: [SECURITY.md](SECURITY.md).

## Get the datasets

1. **X-rays (real):** Kaggle "Chest X-Ray Images (Pneumonia)". Create a Kaggle API token, put `kaggle.json` in `~/.kaggle/`, then:
   ```bash
   pip install kaggle
   cd data && python download_data.py       # creates ./chest_xray/{train,test,val}
   ```
2. **Clinical data (synthetic):** already in `data/`. Regenerate with `python make_clinical.py`.
3. **Real linked data (later):** MIMIC-CXR-JPG + MIMIC-IV (PhysioNet credentialed access) joined on `subject_id`.

## Train the image model

Easiest on **Google Colab** (free T4 GPU): upload `research/MedFusion_Evo_Colab.ipynb` and run all cells.
To train the backend model:
```bash
cd backend
python train.py --data ../chest_xray --epochs 3 --n-train 2000     # writes backend/model.pt
```
This also writes `metrics.json` (accuracy, AUC, sensitivity, specificity, confusion matrix on the held-out test set).
Commit `model.pt` (about 45 MB, under GitHub's 100 MB limit) or use Git LFS.

## Deploy

### 1. GitHub
```bash
git init
git add .
git commit -m "MedFusion-Evo"
git branch -M main
git remote add origin https://github.com/<your-username>/medfusion-evo.git
git push -u origin main
```

### 2. Frontend on Vercel
1. vercel.com → sign in with GitHub → **Add New → Project** → import the repo.
2. Framework Preset **Other**, leave Build Command empty. `vercel.json` already points Output Directory at `frontend`.
3. Deploy. You get a `*.vercel.app` link.

### 3. Backend (optional, for real image scoring)
Vercel cannot run PyTorch, so host `backend/` elsewhere:
- **Render:** New → Web Service → connect repo → Root Directory `backend` → Runtime **Docker** → Deploy.
- **Hugging Face Spaces:** New Space → SDK **Docker** → upload the `backend/` files (set `PORT=7860`).

Set the environment variable `ALLOWED_ORIGINS` to your Vercel URL (see `.env.example`), otherwise the browser will block
the request. Then put the backend `https://…` URL in `API_URL` in `frontend/index.html`, commit, push. Vercel redeploys.

## API

`POST /predict-image` (multipart field `file`, JPG or PNG, max 8 MB) returns
`{"p": 0.91, "con": [64 Grad-CAM cell values, 8x8 row-major], "trained": true, "model_version": "...", "disclaimer": "..."}`.
Errors: `400` invalid or empty image, `413` too large, `415` wrong type, `429` rate limit, `500` prediction failed.
`GET /health` returns `{"status":"ok","trained":true}`.

## Explaining it to judges (30 seconds)

"MedFusion-Evo is an explainable multimodal AI system for pneumonia screening. A CNN reads the chest X-ray and Grad-CAM shows which lung regions drove the score. A Genetic Algorithm selects the most useful clinical variables, and a clinical model explains each patient's risk factor by factor. Both scores are fused into one risk value."

Then add: "If a vital sign is dangerous, a rule-based safety net overrides the AI, and the app warns when its two models disagree."

Be upfront about limits: clinical data and sample X-ray findings are synthetic or preset, and this is a screening aid, not a diagnosis.

## Licence and images
The code is MIT-licensed (see `LICENSE`). Datasets and X-ray images keep their own licences.

Check the licence of every X-ray you publish. Watermarked stock images should not be hosted in a public repo.

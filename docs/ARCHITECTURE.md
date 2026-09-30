# Architecture

```mermaid
flowchart LR
  subgraph Browser["Website (single HTML file, no build)"]
    IN[Input: sample patient or upload] --> VAL[Validate file + values]
    VAL --> IMG[Image score]
    VAL --> CLIN[Clinical model on GA-selected features]
    CLIN --> SHAP[Per-feature contributions]
    IMG --> FUSE[Fusion]
    CLIN --> FUSE
    VAL --> RF[Red-flag rules]
    FUSE --> TRI[Triage action]
    RF --> TRI
    IMG --> AGR[Agreement check]
    CLIN --> AGR
    TRI --> OUT[Report + what-if explorer + JSON export]
    SHAP --> OUT
    AGR --> OUT
  end
  IMG -. optional, HTTPS .-> API
  subgraph Backend["FastAPI backend (optional)"]
    API[/predict-image/] --> SEC[security.py: size, type, bomb, rate limit]
    SEC --> RES[ResNet18 + Grad-CAM]
    RES --> HM[heatmap.py: 8x8 grid]
  end
  HM -. p + 64 cells .-> IMG
```

## Design decisions
| Decision | Why |
|---|---|
| Clinical model runs in the browser | Patient values never leave the device. Works offline and needs no server |
| Backend is optional | The demo works on a static host. Real Grad-CAM needs PyTorch, which Vercel cannot run |
| Logistic regression for the clinical model | Linear, so SHAP values are exact and easy to check (`tests/test_clinical_pipeline.py`) |
| GA for feature selection | Fewer inputs means less data entry. 8 of 18 features reach AUC 0.961 against 0.967 for all 18 |
| Red flags sit **outside** the model | A safety rule must not depend on a learned score. It fires even when the AI says "low risk" |
| Agreement check | When the X-ray and the clinical data disagree the fused number is least trustworthy, so say so |
| Pure logic kept separate from DOM code | The rules can be unit-tested in Node. `heatmap.py` and `security.py` avoid heavy imports so they test without PyTorch |
| Requests handled one at a time | Grad-CAM hooks the shared model. Scale with more workers instead of threads |

## Files
| Path | Role |
|---|---|
| `frontend/index.html` | Website: models, GA, explanations, what-if, accessibility |
| `backend/app.py` | API routes, CORS, rate limit, headers |
| `backend/security.py` | Upload validation, `RateLimiter`, origin parsing |
| `backend/model.py`, `heatmap.py` | ResNet18 + Grad-CAM, grid pooling |
| `backend/train.py` | Trains the image model, writes `model.pt` and `metrics.json` |
| `research/clinical_pipeline.py` | GA + clinical model + validation protocol |
| `research/medfusion_evo.py`, Colab notebook | Full multimodal pipeline with SHAP and a report figure |
| `tests/` | Python (`unittest`/pytest) and JavaScript (`node --test`) suites |

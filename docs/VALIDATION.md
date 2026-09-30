# Validation

## Protocol (`research/clinical_pipeline.py`)
1. Stratified 75/25 split (600 train, 200 test). The test set is never used for selection or fitting.
2. The Genetic Algorithm (population 20, 15 generations) scores feature subsets by 3-fold cross-validated AUC on
   the **training part only**, minus 0.006 per feature.
3. The chosen model is scored once on the held-out test set. AUC has a 500-sample bootstrap 95% interval.
4. It is compared with a logistic regression that uses all 18 features.

Reproduce: `make validate` (a few seconds, no GPU). The raw output is in `validation_report.json`.

## Results (seed 0)
| Metric | Value |
|---|---|
| Selected features | temp, spo2, rr, wbc, crp, fever, dyspnea, sputum (8 of 18) |
| AUC, GA-selected | **0.961** (95% CI 0.932-0.985) |
| AUC, all 18 features | 0.967 |
| Accuracy at 50% cut-off | 93.0% |
| Sensitivity / specificity | 88.6% (70 of 79) / 95.9% (116 of 121) |
| Brier score / calibration error | 0.067 / 0.039 |

The GA-selected model gives up 0.006 AUC to cut the inputs by more than half. The best fitness rises from 0.889 to
0.918 over 15 generations and never falls (elitism, checked by a test).

The website trains its own copy in the browser with a different random stream, so its selected variables and
numbers differ slightly from this table.

## Read these numbers carefully
The clinical data are **synthetic and generated from the label**, so the model is being asked to recover the recipe
that made the data. High scores show the pipeline works, not that it would work on patients. Sensitivity of 88.6%
means 9 of 79 pneumonia cases fall below the 50% line, which is why the interface uses a 30% review band and
rule-based red flags.

## Test suite
| Suite | Tests | What it covers |
|---|---|---|
| `tests/test_data.py` | 12 | CSV schema, bounds, binary flags, prevalence, clinical direction, generator determinism |
| `tests/test_clinical_pipeline.py` | 21 | Loader checks, GA validity/reproducibility/elitism/penalty, evaluation report, SHAP additivity, ECE, bootstrap CI |
| `tests/test_security.py` | 15 | Empty, oversized, non-image, wrong-format, tiny, truncated and bomb uploads. Rate limiter. CORS parsing |
| `tests/test_heatmap.py` | 10 | Grid size, scaling contract with the website, hot-spot placement, bad input |
| `tests/test_model.py` | 6 | ResNet18 output, Grad-CAM range, safe weight loading (needs PyTorch) |
| `tests/test_api.py` | 7 | Endpoints, error codes, headers, 429 rate limit, CORS (needs FastAPI, httpx, PyTorch) |
| `tests/frontend/logic.test.js` | 14 | Sigmoid, AUC, logistic regression, seeded cohort, fusion, red flags, triage, agreement, input/file/API validation |
| `tests/frontend/smoke.test.js` | 8 | Runs the whole page script on a fake DOM: GA completes, report renders, focus handling, what-if, safety net override, disagreement warning, upload validation |

Run everything with `make test`. CI (`.github/workflows/ci.yml`) also lints with ruff and builds the Docker image.
The PyTorch and API tests skip themselves when those packages are missing, so the rest still run on a light install.

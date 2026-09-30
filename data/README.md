# Data

| File | What | Source |
|---|---|---|
| `chest_xray/` (not committed) | ~5,800 labelled chest X-rays (NORMAL / PNEUMONIA) | Kaggle: paultimothymooney/chest-xray-pneumonia (`download_data.py`) |
| `clinical_train_800.csv` | 800 synthetic patients, 18 clinical variables + `diagnosis` | `make_clinical.py` (synthetic) |
| `patients_50.csv` | 50 synthetic patients for the demo | `make_clinical.py` (synthetic) |

Clinical values are simulated from the diagnosis label using clinically plausible distributions
(e.g. lower SpO2, higher temperature, CRP and WBC in pneumonia). They are **not real patients**.

Columns: age, temp (°C), spo2 (%), hr (bpm), rr (/min), cough_days, wbc (×10⁹/L), crp (mg/L), bmi, sbp (mmHg),
fever, dyspnea, sputum, smoker, chest_pain, male, headache, diabetes (0/1), diagnosis (0 = normal, 1 = pneumonia).

## Note on the demo cohort
`patients_50.csv` (made by `make_clinical.py` with NumPy) and the 50 patients inside `frontend/index.html` (made by a
seeded JavaScript generator) follow the same distributions but are **not row-for-row identical**, because the two
random-number generators differ. Use the CSVs for analysis and the website's cohort for the demo.

The validation in `research/clinical_pipeline.py` reads `clinical_train_800.csv` and checks the schema (target column,
no missing values, binary labels) before it runs.

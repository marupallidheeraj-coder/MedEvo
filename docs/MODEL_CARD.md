# Model card: MedFusion-Evo

## Intended use
An **explainable screening aid** that helps a clinician or trained health worker decide *who should be assessed
first* for possible pneumonia, by combining a chest X-ray with routine vitals and labs. It is a teaching and research
demo.

**Users:** clinicians, health workers, students and reviewers exploring explainable multimodal AI.
**Not for:** diagnosis, treatment decisions, use with real patient data, or use without a clinician in the loop.

## Components
| Part | Method | Output |
|---|---|---|
| Image model | ResNet18 (ImageNet weights, fine-tuned) on Kaggle "Chest X-Ray Images (Pneumonia)" | Pneumonia probability + Grad-CAM heatmap |
| Feature selection | Genetic Algorithm (tournament selection, uniform crossover, mutation, elitism). Fitness is cross-validated AUC minus a small penalty per feature | The smallest useful set of clinical variables |
| Clinical model | Logistic regression on the selected variables | Clinical risk + exact per-feature contributions (linear SHAP) |
| Fusion | `final = sigmoid(0.65 * clinical_logit + 0.35 * image_logit)`, image score clamped to 2-98% | One risk score |
| Safety net | Rule-based red flags: SpO₂ < 90%, respiratory rate ≥ 30/min, systolic BP < 90 mmHg | Forces an **Urgent** action whatever the AI score says |
| Agreement check | Warns when the two models fall on opposite sides of 50% and differ by 40 points or more | "Review both findings" notice |

The fusion weights are fixed by hand, not learned. See limitations.

## Data
- **X-rays:** Kaggle chest X-ray dataset, about 5,800 labelled images of paediatric patients from one centre.
- **Clinical variables:** 18 variables (age, temperature, SpO₂, heart rate, respiratory rate, cough days, WBC, CRP,
  BMI, systolic BP, fever, dyspnoea, sputum, smoking, chest pain, sex, headache, diabetes). **Synthetic**: generated
  from the diagnosis label, because no public dataset links X-rays to real vitals. See `data/README.md`.

## Performance
See [VALIDATION.md](VALIDATION.md). On the held-out synthetic test set the clinical model reaches AUC 0.96
(95% CI 0.93-0.99). **These numbers are optimistic by construction** because the data were generated from the label.
Image-model metrics are produced by `backend/train.py` (`metrics.json`) when you train on the Kaggle data.

## Limitations and risks
- **Synthetic clinical data.** Real-world performance is unknown. Fusion has never been tested on real, linked
  X-ray and clinical records.
- **Sample X-rays on the website carry preset findings.** Their scores and heatmaps are illustrative. The
  in-browser image model is trained on generated pictures and is unreliable on real X-rays.
- **Population mismatch.** The X-ray dataset is paediatric and single-centre, while the demo's clinical ranges
  start at age 18. A real system would need age-specific ranges, external validation and site calibration.
- **Sensitivity.** At a 50% cut-off the clinical model misses about 1 in 9 pneumonia cases in the synthetic test
  set. For screening, use the "Clinical review" band (above 30%) and the red-flag rules rather than the 50% line.
- **Hand-set fusion weights** and thresholds are not clinically validated.
- **Grad-CAM shows where the model looked, not proof of disease.** Heatmaps can highlight artefacts such as
  markers or borders.
- **Not evaluated for fairness** across age, sex, device or hospital.

## Human oversight
Every result carries the disclaimer "AI-assisted screening aid, not a medical diagnosis". The tool never
recommends a treatment. Its strongest action is "Urgent: needs prompt clinical assessment".

## Ethical and privacy notes
No real patient data are used or stored. The optional backend processes images in memory only and never saves or
logs them. Check the licence of every X-ray you publish.

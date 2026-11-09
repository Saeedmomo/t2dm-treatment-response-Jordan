# Predicting HbA1c trajectory in type 2 diabetes (Jordan EHR)

Predicting the **direction of a patient's next HbA1c** from routine electronic health record (EHR)
data of a Jordanian health system, and comparing three model architectures on an identical,
deployment-like split.

The outcome is a 3-class label describing the change from an index HbA1c to the next HbA1c measured
90 to 450 days later:

```
improved  = fall of at least 0.5 percentage points
worsened  = rise of at least 0.5 percentage points
stable    = anything in between
```

Every HbA1c measurement in an eligible patient's record is a candidate index, so the task is issued
at the moment of a routine test rather than at a drug initiation or a fixed threshold.

## Models compared

| Model | Representation | Role |
|---|---|---|
| **LightGBM** (best) | flattened tabular features (see below) | gradient-boosted tree |
| Transformer (BEHRT-style) | tokenised patient timeline | sequence model with masked-language pretraining |
| Logistic regression | current HbA1c, age, sex | regression-to-the-mean baseline |

All three were trained and evaluated on the **same temporal, patient-disjoint split**: training
index before 2022, validation 2022, test 2023, with no patient in more than one partition. The
first two are built on one clinical event vocabulary, so the comparison varies the architecture
while holding the recorded information fixed.

## Headline result

On the temporal test set (30,894 instances from 22,908 patients), **LightGBM was the
best-performing model**:

| Metric | LightGBM | Transformer | Logistic regression |
|---|---|---|---|
| AUROC macro (95% CI) | **0.761 (0.757-0.766)** | 0.726 (0.721-0.732) | 0.681 (0.676-0.686) |
| AUROC worsened | **0.702** | 0.641 | 0.518 |
| Brier score | **0.499** | 0.527 | 0.552 |

All pairwise AUROC differences are statistically significant (paired patient-clustered bootstrap;
DeLong with Holm correction for the per-class tests). Confidence intervals are patient-clustered,
because a patient can contribute several prediction instances. Full comparison, calibration,
decision-curve and sensitivity results are in
[`docs/MODEL_COMPARISON.md`](docs/MODEL_COMPARISON.md).

The transformer was trained on a single NVIDIA RTX 5060 laptop GPU in single precision. Training
the identical model on CPU moves these three metrics by about 0.003 macro AUROC.

## The ranking depends on the information each model receives

The headline ordering is not an architectural result. Given **identical binned histories** on the
same 96-token window, the transformer reaches 0.726 macro AUROC against 0.712 for LightGBM, an
advantage of **0.014 (95% CI 0.012 to 0.017)**, and 0.026 (0.020 to 0.032) on the worsened class.

Adding the same five numeric covariates to both, including the exact index HbA1c, raises LightGBM
by 0.050 and the transformer by 0.028, leaving LightGBM ahead by only **0.008 (0.006 to 0.009)**.
Matching the numeric information therefore reverses which architecture appears superior, which is
the central methodological finding of the study.

## Class weighting and calibration

Class weighting moves the probability scale and the decision boundary, not the ranking. Training
the transformer without class weights over three seeds gives:

| | Raw macro AUROC | Calibrated macro AUROC | Raw ECE | Argmax changed by recalibration |
|---|---|---|---|---|
| Transformer, weighted | 0.7264 | 0.7314 | 0.0996 | 0.101 |
| Transformer, unweighted | 0.7312 | 0.7320 | 0.0351 | 0.025 |

Paired patient-clustered differences (2000 resamples, seed 20260819), unweighted minus weighted:
raw +0.0048 (+0.0038 to +0.0058); calibrated +0.0005 (-0.0001 to +0.0011). The ordering
LightGBM > transformer > baseline is unchanged. Aggregate outputs are in
`results/class_weighting/`, produced by
[`src/gpu/train_transformer_unweighted_gpu.py`](src/gpu/train_transformer_unweighted_gpu.py) and
[`src/revision/compare_transformer_weighting.py`](src/revision/compare_transformer_weighting.py).

**Recalibration.** The final pipeline applies vector scaling, `q = softmax(a * log(p) + b)`, fitted
on the 2022 validation partition and applied unchanged to test. After it, expected calibration
error is 0.008 for LightGBM, 0.012 for the transformer and 0.028 for the baseline, and recalibrated
LightGBM gives the greatest net benefit across threshold probabilities of 0.15 to 0.35.

**Patient-clustered confidence intervals.** `src/metrics_lib.py::bootstrap_ci` is the row-level
bootstrap reported for comparison. The patient-clustered intervals reported throughout resample
patients with replacement and keep all of a sampled patient's instances.

## Manuscript figures and reporting checks

The scripts in [`src/figures/`](src/figures) rebuild the manuscript figures from the same data or
from the stored aggregate outputs in `results/`; no model is retrained and no estimate recomputed.
The scripts in [`src/revision/`](src/revision) build the submission documents and fill the official
TRIPOD+AI checklist against the rendered pagination. Their verification outputs, including the
figure data checks and the checklist audit, are in `results/reporting_checks/`.

These scripts load plotting and PDF dependencies from a task-local folder that is not part of this
repository. Install the packages in `requirements.txt` and remove the two `sys.path` lines to run
them against a normal environment.

## Try the model

An interactive demo of the LightGBM model is in [`app/`](app): enter a patient's values and it
returns the calibrated probability of the next HbA1c being improved / stable / worsened. It runs on
your own computer, no account or server needed.

1. Download this repository: click the green **`< > Code`** button on the repo page, then
   **Download ZIP**, and unzip it.
2. Install Python 3.9+ and the packages: `pip install flask lightgbm numpy`
3. Open the `app` folder and run `python app.py` (or on Windows, double-click `app/run.bat`).
4. Open **http://127.0.0.1:5000** in your browser.

Exactly which files to download and full step-by-step instructions are in
[`app/README.md`](app/README.md).

## LightGBM input features (the main model)

The best model, LightGBM, uses **180 features** built from each patient's pre-index history:

- **5 numeric/derived features:** `baseline_hba1c`, `age`, `sex_code`, `n_visits`, `seq_len`.
- **175 token-frequency features:** counts of clinical tokens (binned lab values, lab recency,
  medications on, therapy complexity, diagnosis burden, comorbidities, demographics, inter-visit
  gaps).

The current HbA1c value (`baseline_hba1c`) is by far the most influential feature, followed by the
recent HbA1c history, age and fasting glucose. The full feature list, definitions and importance
ranking are in **[`docs/LIGHTGBM_FEATURES.md`](docs/LIGHTGBM_FEATURES.md)**.

## Interpreting the predictions

This is a **direction-of-change model, not an absolute risk score**. `improved` means the next
HbA1c is predicted to **fall** by at least 0.5 percentage points, `worsened` that it will **rise**
by at least 0.5, and `stable` little change, all relative to the current value.

Because very high values tend to come down and near-target values have little room to fall, a
**high current HbA1c usually predicts "improved" (regression to the mean)**. This is expected and
clinically real, and it does not mean the patient reaches target: a patient can improve from 12% to
10% and remain poorly controlled. To read the effect of risk factors, compare patients at the
**same baseline HbA1c**; holding the baseline fixed, adding risk features (higher fasting glucose,
insulin use, comorbidities) increases the predicted probability of `worsened`.

## Repository structure

```
src/                  analysis and modelling pipeline
src/gpu/              GPU transformer training
src/figures/          manuscript figure rebuild scripts
src/revision/         document build, checklist refill, class-weighting comparison
docs/                 feature documentation and model comparison
results/              aggregate result tables, no patient-level data
results/class_weighting/    weighted versus unweighted transformer
results/reporting_checks/   figure data checks and the TRIPOD+AI checklist audit
figures/              ROC, calibration, decision curve, subgroup and interpretability figures
app/                  interactive demo of the LightGBM model
LICENSE               MIT licence
CITATION.cff          how to cite this repository
```

Pipeline (in `src/`): `explore_data.py` -> `build_cohort.py` -> `prep_features.py` ->
`train_baselines.py` (LightGBM + logistic regression) and `train_transformer.py` -> `evaluate.py`,
`sensitivity.py`, `subgroup.py`, `interpret_ig.py`, `reporting.py`.

## Data availability and ethics

The underlying EHR data are **not included** in this repository and cannot be shared, as they
contain routinely collected patient records. Formal ethical approval was granted by the Jordanian
Ministry of Health Research Ethics Committee (Approval No. MOH/REC/2025/487). Only code, aggregate
result tables and figures are published here. The `.gitignore` is configured to prevent any raw
data or patient-level derived files from being committed.

## Reproducing

The code expects a local copy of the source EHR extract (not provided). With that in place and the
dependencies in `requirements.txt`, run the pipeline scripts in `src/` in the order listed above.

## Status

Research prototype. The models require external validation before any clinical use.

## Licence

MIT. See `LICENSE`. Copyright (c) 2026 Dania Rahhal, Said Moshawih, Suha Telfah.

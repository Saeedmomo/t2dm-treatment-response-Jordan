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

## Models compared

| Model | Representation | Role |
|---|---|---|
| **LightGBM** (best) | flattened tabular features (see below) | gradient-boosted tree |
| Transformer (BEHRT-style) | tokenised patient timeline | sequence model with masked-language pretraining |
| Logistic regression | current HbA1c, age, sex | regression-to-the-mean baseline |

All three were trained and evaluated on the **same temporal, patient-disjoint split** (training index
before 2022, validation 2022, test 2023; no patient in more than one partition).

## Headline result

On the temporal test set (30,894 instances / 22,908 patients), **LightGBM was the best-performing
model**:

| Metric | LightGBM | Transformer | Logistic regression |
|---|---|---|---|
| AUROC macro (95% CI) | **0.761 (0.757-0.766)** | 0.726 (0.721-0.732) | 0.681 (0.676-0.686) |
| AUROC worsened | **0.702** | 0.641 | 0.518 |
| Brier score | **0.499** | 0.527 | 0.552 |

All pairwise AUROC differences are statistically significant (paired patient-clustered bootstrap;
DeLong with Holm correction for the per-class tests). Confidence intervals are patient-clustered,
because a patient can contribute several prediction instances. Full comparison, calibration,
decision-curve, and sensitivity results are in
[`docs/MODEL_COMPARISON.md`](docs/MODEL_COMPARISON.md).

The transformer was trained on a single NVIDIA RTX 5060 laptop GPU in single precision. Training the
identical model on CPU moves these three metrics by about 0.003 macro AUROC, and the agreement
between the two computing environments is reported in the supplementary material.

## Try the model

An interactive demo of the LightGBM model is in [`app/`](app): enter a patient's values and it returns
the predicted probability of the next HbA1c being improved / stable / worsened. It runs on your own
computer, no account or server needed.

1. Download this repository: click the green **`< > Code`** button on the repo page, then **Download
   ZIP**, and unzip it.
2. Install Python 3.9+ and the packages: `pip install flask lightgbm numpy`
3. Open the `app` folder and run `python app.py` (or on Windows, double-click `app/run.bat`).
4. Open **http://127.0.0.1:5000** in your browser.

Exactly which files to download and full step-by-step instructions are in
[`app/README.md`](app/README.md).

## LightGBM input features (the main model)

The best model, LightGBM, uses **180 features** built from each patient's pre-index history:

- **5 numeric/derived features:** `baseline_hba1c`, `age`, `sex_code`, `n_visits`, `seq_len`.
- **175 token-frequency features:** counts of clinical tokens (binned lab values, lab recency,
  medications on, therapy complexity, diagnosis burden, comorbidities, demographics, inter-visit gaps).

The current HbA1c value (`baseline_hba1c`) is by far the most influential feature, followed by the
recent HbA1c history, age, and fasting glucose. The full feature list, definitions, and importance
ranking are in **[`docs/LIGHTGBM_FEATURES.md`](docs/LIGHTGBM_FEATURES.md)**.

## Interpreting the predictions

This is a **direction-of-change model, not an absolute risk score**. `improved` means the next HbA1c is
predicted to **fall** by at least 0.5 percentage points, `worsened` that it will **rise** by at least
0.5, and `stable` little change, all relative to the current value.

Because very high values tend to come down and near-target values have little room to fall, a **high
current HbA1c usually predicts "improved" (regression to the mean)**. This is expected and clinically
real, and it does not mean the patient reaches target: a patient can improve from 12% to 10% and remain
poorly controlled. To read the effect of risk factors, compare patients at the **same baseline HbA1c**;
holding the baseline fixed, adding risk features (higher fasting glucose, insulin use, comorbidities)
increases the predicted probability of `worsened`.

## Repository structure

```
src/        analysis and modelling code
docs/       feature documentation and model comparison
results/    aggregate result tables (no patient-level data)
figures/    figures (ROC, calibration, decision curve, subgroup, interpretability)
```

Pipeline (in `src/`): `explore_data.py` -> `build_cohort.py` -> `prep_features.py` ->
`train_baselines.py` (LightGBM + logistic regression) and `train_transformer.py` -> `evaluate.py`,
`sensitivity.py`, `subgroup.py`, `interpret_ig.py`, `reporting.py`.

## Data availability and ethics

The underlying EHR data are **not included** in this repository and cannot be shared, as they contain
routinely collected patient records. The study was approved by the ethics committee of the Ministry of
Health of Jordan. Only code, aggregate result tables, and figures are published here. The `.gitignore`
is configured to prevent any raw data or patient-level derived files from being committed.

## Reproducing

The code expects a local copy of the source EHR extract (not provided). With that in place and the
dependencies in `requirements.txt`, run the pipeline scripts in `src/` in the order listed above.

## Status

Research prototype. The models require external validation before any clinical use.

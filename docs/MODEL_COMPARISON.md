# Three-model comparison

Task: predict the direction of a patient's next HbA1c (**improved / stable / worsened**), 90 to 450
days after an index measurement. Improved means a fall of at least 0.5 percentage points, worsened a
rise of at least 0.5, and stable anything in between.

All three models were trained and evaluated on the **identical temporal, patient-disjoint split**
(training index before 2022, validation in 2022, test in 2023; no patient appears in more than one
partition). The test set contains 30,894 prediction instances from 22,908 patients.

## Headline result

**LightGBM was the best-performing model** on every headline measure. The BEHRT-style transformer was
second, and the logistic-regression baseline (current HbA1c, age, sex) was third.

| Metric | LightGBM | Transformer | Logistic regression |
|---|---|---|---|
| AUROC macro (95% CI) | **0.761 (0.757-0.766)** | 0.726 (0.721-0.732) | 0.681 (0.676-0.686) |
| AUROC micro | **0.799** | 0.777 | 0.763 |
| AUROC worsened | **0.702** | 0.641 | 0.518 |
| AUROC stable | **0.772** | 0.750 | 0.733 |
| AUROC improved | **0.810** | 0.788 | 0.792 |
| AUPRC worsened | **0.375** | 0.313 | 0.210 |
| Brier score (lower is better) | **0.499** | 0.527 | 0.552 |
| Expected calibration error (mean) | **0.084** | 0.100 | 0.112 |
| Balanced accuracy | **0.568** | 0.542 | 0.505 |

The transformer is the mean of three fine-tuning seeds (macro AUROC 0.726, standard deviation 0.005
over five seeds), trained on GPU in single precision. Training the identical model on CPU gives
0.729 macro / 0.644 worsened / 0.520 Brier, so the two computing environments agree to within about
0.003 macro AUROC.

### A note on the probabilities

All three models are trained with balanced class weights, which helps the minority classes but leaves
the predicted probabilities on a reweighted rather than an observed-risk scale. Untouched, the mean
predicted probability of worsening is 0.284 for LightGBM, 0.308 for the transformer and 0.341 for the
baseline, against an observed prevalence of 0.195. The paper therefore applies a per-class affine
correction (vector scaling) fitted on the validation partition and applied unchanged to the test
partition, which brings the mean predicted risk to 0.192 / 0.203 / 0.182 and the mean expected
calibration error to 0.008 / 0.012 / 0.028. Discrimination is unaffected (macro AUROC moves by at most
0.005) and the ranking is unchanged. The calibration and decision-curve figures in the paper are drawn
on the corrected probabilities; the raw and temperature-scaled values are reported alongside them.

The demonstration app in [`app/`](../app) returns the model's raw probabilities and does not apply this
correction, so its outputs should be read as relative risk ordering rather than as calibrated absolute
risk.

## Statistical comparison

Pairwise differences in AUROC by paired bootstrap (2,000 resamples), from
`results/model_comparison_auroc.csv`. Every difference is significant.

| Comparison | Metric | Difference (95% CI) | p |
|---|---|---|---|
| LightGBM vs transformer | macro | 0.033 (0.029-0.036) | < 0.001 |
| LightGBM vs transformer | worsened | 0.057 (0.051-0.064) | < 0.001 |
| Transformer vs logistic regression | macro | 0.048 (0.044-0.051) | < 0.001 |
| LightGBM vs logistic regression | macro | 0.080 (0.076-0.085) | < 0.001 |

## Calibration and clinical utility

- LightGBM had the lowest Brier score and expected calibration error. Temperature scaling improved all
  three models (LightGBM ECE 0.084 to 0.077). See `results/calibration_metrics.csv`,
  `figures/reliability_diagrams.png`, and `figures/calibration_temp_scaling_worsened.png`.
- In decision curve analysis for the worsened class, LightGBM gave the highest net benefit across the
  clinically relevant threshold range and no model beat it (`figures/decision_curve_worsened.png`).

## Robustness (sensitivity analyses)

From `results/sensitivity_auroc.csv` (AUROC macro / worsened):

- Secondary patient-grouped random split: LightGBM 0.767 / 0.728 (consistent with the temporal split).
- Outcome threshold 0.3: LightGBM 0.720 / 0.684. Threshold 1.0: LightGBM 0.836 / 0.743.
- Follow-up window 90-365 days: LightGBM 0.762 / 0.702 (unchanged).
- Transformer with vs without masked-language pretraining: 0.726 / 0.641 vs 0.726 / 0.640 when both are averaged over the same three seeds. Comparing a single unpretrained seed against the three-seed pretrained ensemble would suggest a larger benefit, but that difference is an ensembling effect rather than a pretraining effect.

The worsened class was the hardest to predict for every model, reflecting genuine clinical
unpredictability rather than a modelling shortcoming.

## Why the tree beat the transformer

The predictive signal is concentrated in a few tabular quantities (chiefly the current HbA1c, then
fasting glucose and age). Gradient-boosted trees are strong on exactly this kind of tabular EHR feature
set, and masked-language pretraining added little. A gradient-boosted tree should be included as a
serious comparator whenever an EHR transformer is proposed.

## Key figures

- `figures/roc_ovr.png` - ROC curves per class and model
- `figures/confusion_matrices.png` - confusion matrices
- `figures/reliability_diagrams.png` - calibration
- `figures/decision_curve_worsened.png` - clinical utility
- `figures/subgroup_auroc.png` - subgroup discrimination
- `figures/ig_global_importance.png` - transformer token attributions

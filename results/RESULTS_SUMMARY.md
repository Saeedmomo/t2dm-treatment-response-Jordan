# HbA1c-Trajectory Modelling — Results Summary

Prediction task: given a patient's history **strictly before** an index HbA1c draw, predict whether
their next HbA1c (90–450 days later) will be **improved** (Δ ≤ −0.5), **stable**, or **worsened**
(Δ ≥ +0.5). Three models trained and evaluated on the **identical PRIMARY temporal + patient-disjoint
split** (train index < 2022 / validation 2022 / test 2023). Code in `src/`, tables in `results/`,
figures in `figures/`.

**Compute note.** The transformer was trained on a single NVIDIA GeForce RTX 5060 laptop GPU (8 GB) in
single precision. Architecture: BERT, hidden 192, 4 layers, 4 heads, ~1.95M parameters, sequence length
capped at 96 (keeping `[CLS]` + most-recent history), batch 256, MLM pretraining 2 epochs + 2-epoch
fine-tuning × 3 seeds. Training the identical model on CPU moves macro AUROC by about 0.003.

**Confidence intervals.** A patient can contribute several prediction instances (1.35 per patient in the
test partition), so intervals are **patient-clustered** bootstrap (2,000 resamples): patients, not rows,
are resampled. Clustered intervals are on average **1.062×** wider than row-level ones, and no conclusion
depends on the choice (`robust_clustered_ci.csv`).

---

## 1. Main results — PRIMARY temporal test set (n = 30,894 pairs / 22,908 patients)

Transformer = ensemble mean of 3 seeds.

| Model | AUROC macro | AUROC micro | AUROC worsened | AUPRC worsened | Brier | ECE (mean) | Balanced acc |
|---|---|---|---|---|---|---|---|
| **LightGBM** | **0.761 [0.757, 0.766]** | **0.799** | **0.702 [0.694, 0.709]** | **0.375** | **0.499** | **0.084** | **0.568** |
| Transformer (BEHRT, GPU) | 0.726 [0.721, 0.732] | 0.777 | 0.641 [0.633, 0.649] | 0.313 | 0.527 | 0.100 | 0.542 |
| LogReg (baseline HbA1c+age+sex) | 0.681 [0.676, 0.686] | 0.763 | 0.518 [0.509, 0.526] | 0.210 | 0.552 | 0.112 | 0.505 |

Per-class AUROC, AUPRC, and full sens/spec/PPV/NPV/F1 tables: `main_results_test.csv`,
`per_class_operating_metrics.csv`. Confusion matrices: `confusion_*.csv` + `confusion_matrices.png`.
Note that `main_results_test.csv` carries row-level intervals; the clustered intervals quoted above are
in `robust_clustered_ci.csv`.

**Headline finding.** The **LightGBM gradient-boosted comparator is the best model**, the transformer is
second, and the logistic baseline third. The baseline captures the "improved" class reasonably
(AUROC 0.792 — a high starting HbA1c regresses to the mean) but is essentially **chance on "worsened"**
(0.518), confirming that predicting deterioration needs more than the current value. Section 6 shows
that most of the transformer's deficit is a **representation** difference rather than an architectural
one.

**Transformer seed stability** (`transformer_seed_variability.csv`): over five fine-tuning seeds,
AUROC macro **0.724 ± 0.005**, worsened **0.636 ± 0.011**.

### Operating threshold

Decision uses **argmax** for the confusion matrices. For a deployable **"worsened" alert**, the threshold
was tuned on the validation set by Youden's J and applied unchanged to test (`worsened_operating_point.csv`):

| Model | worsened threshold | sensitivity | specificity | PPV | NPV | F1 |
|---|---|---|---|---|---|---|
| LightGBM | 0.278 | 0.686 | 0.607 | 0.297 | 0.889 | 0.414 |
| Transformer | 0.309 | 0.615 | 0.583 | 0.263 | 0.862 | 0.368 |
| LogReg | 0.366 | 0.336 | 0.718 | 0.223 | 0.817 | 0.268 |

## 2. Model comparison

Two methods, agreeing to the third decimal: a **paired patient-clustered bootstrap**
(`model_comparison_auroc.csv`, 2,000 resamples) and the **DeLong test** for correlated ROC curves with
Holm–Bonferroni correction (`robust_delong.csv`).

- Transformer − LightGBM (macro): **−0.035 [−0.038, −0.032]** → LightGBM better.
- Transformer − LogReg (macro): **+0.046 [+0.042, +0.050]** → transformer beats baseline.
- LightGBM − LogReg (macro): **+0.080 [+0.076, +0.085]**.
- On "worsened": LightGBM − LogReg **+0.184**, Transformer − LogReg **+0.123**.

Every macro and worsened-class difference is significant after Holm correction (DeLong p ≤ 3.5e−76 for
LightGBM vs transformer). The bootstrap p values print as `0.0` in the CSV because they fall below the
resolution of 2,000 resamples, i.e. p < 0.0005; `robust_delong.csv` carries exact analytic p values.

**One exception to the ordering:** on the *improved* class the transformer falls marginally **below** the
logistic baseline, by 0.004 (DeLong z = −2.00, Holm-adjusted **p = 0.046**). The effect is small and sits
at the significance boundary, and is reported as a marginal difference. Its interest is that a
three-variable regression is not bettered by a sequence transformer on the one class that regression to
the mean already explains.

## 3. Calibration (`calibration_metrics.csv`, figures `reliability_diagrams.png`, `calibration_temp_scaling_worsened.png`)

- Multiclass Brier: LightGBM 0.499 < Transformer 0.527 < LogReg 0.552.
- Raw ECE (mean over classes): LightGBM 0.084, Transformer 0.100, LogReg 0.112.
- **Temperature scaling** (single T fit on validation) improved every model — LightGBM 0.084 → 0.077
  (T = 0.85), Transformer 0.100 → 0.088 (T = 0.75), LogReg 0.112 → 0.098 (T = 0.64). Ordering unchanged.
- Calibration slopes are closest to unity for LightGBM on every class (1.02 / 1.08 / 0.97 for improved /
  stable / worsened). The transformer is mildly under-confident on "stable" (1.23) and the logistic
  baseline markedly over-confident on "worsened" (0.565).

## 4. Clinical utility — decision curve (`decision_curve_worsened.png`)

For the "worsened" one-vs-rest decision, **LightGBM has the highest net benefit across the clinically
relevant threshold range (~0.15–0.35)** and stays above both treat-all and treat-none. The transformer
tracks and at times falls below treat-all in the mid-range; the logistic baseline falls below treat-none
once thresholds exceed prevalence.

## 5. Sensitivity analyses (`sensitivity_auroc.csv`)

All three models were refitted under **every** condition, so the grid is complete.

| Condition | Transformer | LightGBM | LogReg |
|---|---|---|---|
| primary (±0.5, 90–450, temporal) | 0.726 / 0.641 | 0.761 / 0.702 | 0.681 / 0.518 |
| secondary grouped-random split | 0.746 / 0.695 | 0.767 / 0.728 | 0.681 / 0.546 |
| label threshold ±0.3 | 0.681 / 0.625 | 0.720 / 0.684 | 0.639 / 0.503 |
| label threshold ±1.0 | 0.814 / 0.699 | 0.836 / 0.743 | 0.772 / 0.590 |
| follow-up window 90–365 | 0.732 / 0.652 | 0.762 / 0.702 | 0.683 / 0.519 |

*(cells are AUROC macro / AUROC worsened)*

- **Ordering is stable:** LightGBM ranks first in all five conditions, with a margin over the transformer
  of between **0.021 and 0.038** macro AUROC.
- **Split agreement:** the random split reproduces the temporal ordering and magnitudes closely
  (LightGBM 0.767 vs 0.761) — the temporal split is not creating an artefact. All models score slightly
  higher on the random split, as expected once distribution shift is removed.
- **Threshold:** a bigger required change is easier to predict (±1.0 → 0.836) and a tighter one harder
  (±0.3 → 0.720), monotone and sensible.
- **Window:** 90–365 ≈ 90–450 (0.762 vs 0.761) — robust to the follow-up horizon.

## 6. Why the tree wins: tuning, data volume, and representation

Three candidate explanations for the transformer's second place were tested directly rather than assumed
(`transformer_optimisation.csv`, `learning_curve.csv`, `pretraining_ablation_fair.csv`).

**Not under-tuning.** Varying one factor at a time moved macro AUROC by between −0.003 and +0.005,
against a deficit of 0.035:

| Variant | AUROC macro | Δ vs primary |
|---|---|---|
| primary (1.95M params, 96 tokens, 2 epochs, lr 3e-4) | 0.726 | — |
| sequence length 128 | 0.727 | +0.001 |
| 4 fine-tuning epochs | 0.730 | +0.003 |
| learning rate 1e-4 | 0.723 | −0.003 |
| learning rate 5e-4 | 0.728 | +0.002 |
| 4 MLM pretraining epochs | 0.724 | −0.003 |
| capacity 11.1M params (hidden 384, 6 layers) | 0.732 | **+0.005** |

**Not data volume.** On nested patient-level subsets, both models improve slowly and in parallel; the gap
stays between 0.032 and 0.037 with no narrowing trend:

| Training instances | Transformer | LightGBM | gap |
|---|---|---|---|
| 29,865 (10%) | 0.714 | 0.746 | 0.032 |
| 74,862 (25%) | 0.720 | 0.757 | 0.037 |
| 149,582 (50%) | 0.728 | 0.761 | 0.033 |
| 298,918 (100%) | 0.728 | 0.761 | 0.034 |

**Largely representation.** The transformer saw laboratory values only as binned tokens, while the
tabular models also received continuous features. Giving the transformer the **identical five continuous
features** LightGBM receives raises macro AUROC 0.726 → **0.754** and worsened 0.641 → **0.691**,
recovering about **80%** of the gap. This model has strictly more information than LightGBM (the same
continuous features *plus* the full token sequence) and still loses on all three classes, by 0.003–0.010
AUROC (DeLong p ≤ 6.4e−04).

**Pretraining ablation.** Compared like-for-like (three seeds against three seeds), MLM pretraining gives
**+0.0005 macro / +0.0016 worsened** — an order of magnitude below the seed-to-seed SD of 0.005, i.e.
**no detectable benefit**. Comparing a single unpretrained seed against a three-seed pretrained ensemble
would instead suggest +0.004; that difference is an ensembling effect, not a pretraining one.

## 7. Subgroup / fairness (`subgroup_metrics.csv`, `subgroup_gaps.csv`, `subgroup_auroc.png`)

Differences are reported with intervals **on the difference itself**, which is the quantity a fairness
claim rests on.

- **Sex:** no evidence of disparity for the transformer, gap 0.002 [−0.013, 0.019] (spans zero). For
  LightGBM, 0.015 [0.000, 0.032], only marginally excluding zero.
- **Baseline HbA1c** is the largest and most consistent source of variation: both models are weakest in
  the **8–9% "hard middle"** (transformer 0.598, LightGBM 0.597), where the direction of change is
  genuinely ambiguous — a gap of 0.080 [0.045, 0.112] and 0.112 [0.080, 0.141] respectively.
- **Age:** the models differ. The transformer is weakest under 40 (worsened AUROC 0.602) whereas LightGBM
  is *best* in that band (0.726).
- **History length:** more than 3 visits improves worsened AUROC for both (transformer 0.633 → 0.664,
  LightGBM 0.697 → 0.720).
- LightGBM equalled or exceeded the transformer on macro AUROC in **every** subgroup examined.

## 8. Interpretability — Layer Integrated Gradients (`ig_*` tables + `ig_global_importance.png`)

Global token importance for "worsened" is **clinically coherent**: **HbA1c value bins** occupy seven of
the top ten positions, followed by **fasting glucose** (FBS_GT_400 strongly positive) and **age bands**.
Signed attributions run in the expected direction — the 7–8% and 8–9% HbA1c bins push toward worsening
(+0.164, +0.128), while the highest bin (>12%) pushes **away** from it (−0.059), which is regression to
the mean expressed in the model's own attributions.

**Faithfulness, against a control.** Removing the top-12 attributed tokens drops mean P(worsened) from
**0.318 to 0.260**, whereas removing an equal number of **randomly chosen** tokens from the same
sequences drops it only to **0.308**. The separation widens monotonically with the number removed (0.012
at one token, 0.052 at eight), so the attributions identify positions the model actually relies on rather
than merely reflecting loss of information in general.

## 9. Reporting artifacts

- **Cohort flow** (`cohort_flow.png`): 382,537 patients → 382,472 with any HbA1c → cleaning removals →
  381,374 with a clean draw → **151,750 patients / 569,201 pairs** with a 90–450-day follow-up.
- **Table 1** by split (`table1_by_split.csv`) and by outcome class (`table1_by_outcome.csv`). Note the
  **temporal drift**: train baseline HbA1c 7.38 ± 2.07 vs test 6.68 ± 1.93, and more "stable" cases in
  2022–23 — a realistic covariate shift the temporal split deliberately preserves.

## 10. Limitations

- Tree ensembles are a strong baseline on this task. Tuning the transformer across capacity, sequence
  length, epochs, learning rate, and pretraining did not close the gap, and the residual deficit after
  representation parity, while small, is statistically significant on all three classes.
- The shared clinical vocabulary still represents laboratory values as bins inside the token sequence; a
  tokenisation designed around continuous values might narrow the residual gap further.
- The optimisation search varied one factor at a time from a sensible default rather than exploring the
  joint space.
- Vitals carry no timestamp and were emitted as `*_MISSING` (no leakage), so the models use labs, meds,
  diagnoses and demographics only.
- Argmax detection of "worsened" is poor for every model, and even at tuned thresholds precision for that
  class is limited, which constrains the immediate clinical applicability of a worsening alert.

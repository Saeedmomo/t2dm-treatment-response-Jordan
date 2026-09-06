# LightGBM input features (HbA1c-trajectory model)

This document describes exactly what the **best-performing model (LightGBM)** receives as input.
The model predicts the 3-class direction of a patient's next HbA1c (improved / stable / worsened),
90 to 450 days after an index HbA1c draw.

Every model in this study is trained and evaluated on the **same underlying pre-index history**, so
the LightGBM feature vector and the transformer token sequence are derived from identical information.
The LightGBM vector is a **flattened tabular summary** of that history and has **180 features** in two
groups.

Feature construction lives in [`src/prep_features.py`](../src/prep_features.py) (builds the vectors)
and [`src/train_baselines.py`](../src/train_baselines.py) (assembles the LightGBM design matrix).

---

## Group 1 — Numeric / derived tabular features (5)

| Feature | Type | Definition |
|---|---|---|
| `baseline_hba1c` | continuous | The index HbA1c value in percent (the value at the prediction point). |
| `age` | continuous | Patient age in years at the index draw, from date of birth. |
| `sex_code` | categorical | 0 = female, 1 = male, -1 = missing. |
| `n_visits` | count | Number of distinct lab-visit panels in the patient's pre-index history. |
| `seq_len` | count | Length of the patient's tokenised timeline (number of tokens). |

## Group 2 — Token-frequency features (175)

Each patient's pre-index history is encoded as a sequence of clinical tokens drawn from a fixed
vocabulary of **175 tokens** (`vocab.json`). For the tabular models, that sequence is summarised as a
**bag-of-tokens count vector**: one feature per vocabulary token, whose value is how many times the
token appears in that patient's history. The vocabulary is grouped into the following families.

| Family | Example tokens | What it encodes |
|---|---|---|
| Binned lab values | `NUM_HBA1C_7_0_8_0`, `NUM_FBS_200_400`, `NUM_CREATININE_0_9_1_2`, `NUM_LDL_100_130` | The value of each of 7 tracked analytes (HbA1c, fasting blood sugar, creatinine, blood urea nitrogen, LDL, triglycerides, total cholesterol), discretised into clinical bands; `..._MISSING` when never measured. |
| Lab recency | `LABAGE_HBA1C_0_30D`, `LABAGE_CREATININE_91_180D`, `LABAGE_FBS_GT180D` | How recently each analyte was last measured relative to the visit (0-30, 31-90, 91-180, >180 days, or missing). |
| Medications on | `MED_ON_METFORMIN`, `MED_ON_SULFONYLUREA`, `MED_ON_DPP4`, `MED_ON_SGLT2`, `MED_ON_GLP1`, `MED_ON_INSULIN_BASAL`, `MED_ON_INSULIN_BOLUS` | Glucose-lowering drug classes active at the visit (dispensed within a 180-day look-back). |
| Therapy complexity | `THERAPY_NAGENTS_1` … `THERAPY_NAGENTS_6`, `THERAPY_MONOCOMBO_0/1` | Number of active drug classes and monotherapy vs combination. |
| Therapy changes | `ACT_ADD_METFORMIN`, `ACT_REM_SGLT2`, `EVTYPE_ADD_ON`, `EVTYPE_SWITCH_OR_COMPLEX` | Regimen add/remove actions and event types between visits. |
| Diagnosis burden | `DXCOUNT_0`, `DXCOUNT_1_2`, `DXCOUNT_3_5`, `DXCOUNT_GT5` | Cumulative count of recorded diagnoses as of the visit. |
| Comorbidities | `COM_HAS_HTN`, `COM_HAS_DYSLIPIDEMIA`, `COM_HAS_CKD`, `COM_HAS_ASCVD`, `COM_HAS_HF`, `COM_HAS_MASLD` | Recognised comorbidities present as of the visit (mapped from diagnosis text). |
| Demographics | `DEM_AGE_50_60`, `DEM_SEX_FEMALE`, `DEM_SEX_MALE` | Age band and sex as static context tokens. |
| Inter-visit gaps | `GAP_0_7D`, `GAP_1_3M`, `GAP_3_6M`, `GAP_6_12M`, `GAP_GT1Y` | Time between consecutive visits. |
| Vitals | `VITAL_SBP_MISSING`, `VITAL_BMI_MISSING`, … | Vital signs. In this dataset vitals carry **no timestamp** and cannot be placed on the timeline without leakage, so they are always encoded as missing. |
| Structural | `[CLS]`, `[EV]`, `[CTX]`, `[SEP]` | Sequence-structure markers. |

**Total: 175 token features + 5 numeric features = 180 features.**

### Notes

- **Leakage guard.** The interval to the follow-up draw (`days_between`) is a property of the outcome,
  not known at the prediction point, and is **excluded from every model**.
- **Class imbalance** is handled with balanced sample weights during training.
- **Feature names** containing special characters (for example `[CLS]`) are sanitised for LightGBM
  (`[CLS]` becomes `SPECIAL_CLS`, etc.).
- The continuous `baseline_hba1c` is only available to the tabular models; the transformer sees the
  index HbA1c only through its binned token. This is discussed as a contributor to the performance gap
  in the accompanying manuscript.

---

## Feature importance (top 25 by gain)

From [`results/lgbm_feature_importance.csv`](../results/lgbm_feature_importance.csv). Gain is the total
reduction in loss attributed to a feature across all splits; a higher value means a more influential
feature.

| Rank | Feature | Gain | Interpretation |
|---:|---|---:|---|
| 1 | `baseline_hba1c` | 584,394 | Current HbA1c value (by far the dominant predictor). |
| 2 | `NUM_HBA1C_LE_6_0` | 90,970 | History includes a normal/near-normal HbA1c. |
| 3 | `age` | 68,938 | Patient age. |
| 4 | `NUM_HBA1C_6_0_7_0` | 56,714 | HbA1c in the 6-7% band. |
| 5 | `NUM_HBA1C_7_0_8_0` | 41,934 | HbA1c in the 7-8% band. |
| 6 | `NUM_HBA1C_10_0_12_0` | 33,362 | HbA1c in the 10-12% band. |
| 7 | `NUM_HBA1C_8_0_9_0` | 29,200 | HbA1c in the 8-9% band. |
| 8 | `THERAPY_MONOCOMBO_1` | 24,914 | Combination therapy. |
| 9 | `NUM_FBS_LE_110` | 24,585 | Fasting glucose at or below 110 mg/dL. |
| 10 | `NUM_HBA1C_9_0_10_0` | 23,326 | HbA1c in the 9-10% band. |
| 11 | `seq_len` | 21,472 | Amount of recorded history. |
| 12 | `NUM_HBA1C_GT_12_0` | 18,873 | Very high HbA1c (>12%). |
| 13 | `LABAGE_HBA1C_0_30D` | 18,716 | A recent HbA1c (within 30 days). |
| 14 | `MED_ON_METFORMIN` | 16,251 | On metformin. |
| 15 | `NUM_FBS_200_400` | 14,568 | High fasting glucose (200-400 mg/dL). |
| 16 | `GAP_3_6M` | 14,485 | A 3-6 month gap between visits. |
| 17 | `NUM_CREATININE_MISSING` | 14,418 | No creatinine on record. |
| 18 | `LABAGE_CREATININE_0_30D` | 13,441 | A recent creatinine. |
| 19 | `MED_ON_SULFONYLUREA` | 12,918 | On a sulfonylurea. |
| 20 | `THERAPY_NAGENTS_1` | 12,254 | Single-agent therapy. |
| 21 | `NUM_FBS_MISSING` | 11,303 | No fasting glucose on record. |
| 22 | `DXCOUNT_0` | 10,985 | No recorded diagnoses. |
| 23 | `NUM_FBS_130_160` | 10,698 | Fasting glucose 130-160 mg/dL. |
| 24 | `NUM_FBS_110_130` | 10,545 | Fasting glucose 110-130 mg/dL. |
| 25 | `LABAGE_FBS_0_30D` | 10,388 | A recent fasting glucose. |

The pattern is clinically coherent: the current HbA1c value dominates, followed by the recent HbA1c
history, age, fasting glucose, and the treatment context. The full ranking of all 180 features is in
`results/lgbm_feature_importance.csv`.

## A note on reading the output (regression to the mean)

Because `baseline_hba1c` dominates, the model's output is driven mainly by where the patient starts.
A **high** current HbA1c usually predicts **improved** (the value tends to fall), and a near-target
value predicts **stable**. This is regression to the mean, not a defect, and "improved" does not imply
the patient reaches target. The effect of the other, weaker features (labs, medications, comorbidities)
is seen by comparing patients at the **same** baseline HbA1c: with the baseline held fixed, adding
risk-increasing features raises the predicted probability of **worsened**.

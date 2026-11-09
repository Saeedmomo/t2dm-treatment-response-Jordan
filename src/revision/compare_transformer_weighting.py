"""V8 revision, item 7: compare the class-weighted and unweighted BEHRT-style transformer.

Run after src/gpu/train_transformer_unweighted_gpu.py has written
GPU_trans/preds/transformer_unweighted__temporal_{validation,test}__seed{0,1,2}.parquet.

Protocol (identical to the V7 LightGBM and logistic-regression class-weighting analysis):
  * three-seed mean probabilities, validation-fitted vector scaling, Youden threshold from validation;
  * patient-clustered 95% CIs and paired differences, 2000 resamples, seed 20260819;
  * the test partition is never used for fitting.

Writes to REVISION_OUT (default <HAKEEM_BASE>/revision_V7):
  transformer_weighting_results.csv     one row per model, full metric set (evaluate_runs.evaluate)
  transformer_weighting_contrasts.csv   paired clustered differences
  transformer_weighting_summary.md      plain-language verdict and whether the manuscript needs updating
"""
import glob, json, numpy as np, pandas as pd
from revlib import *
os.chdir(OUT)
from evaluate_runs import evaluate

W = [f"{GPRED}/transformer__temporal_validation__seed*.parquet", f"{GPRED}/transformer__temporal_test__seed*.parquet"]
U = [f"{GPRED}/transformer_unweighted__temporal_validation__seed*.parquet", f"{GPRED}/transformer_unweighted__temporal_test__seed*.parquet"]
LG = [f"{PREDS}/lgbm__temporal_validation__seed0.parquet", f"{PREDS}/lgbm__temporal_test__seed0.parquet"]
LR = [f"{PREDS}/logreg__temporal_validation__seed0.parquet", f"{PREDS}/logreg__temporal_test__seed0.parquet"]
LRU = ["preds/logreg_E4_unweighted__validation.parquet", "preds/logreg_E4_unweighted__test.parquet"]
for pat in U:
    if not glob.glob(pat):
        raise SystemExit(f"Missing {pat}. Run src/gpu/train_transformer_unweighted_gpu.py on the GPU first.")

runs = {"Transformer weighted (published, 3 seeds)": W, "Transformer unweighted (3 seeds)": U,
        "LightGBM weighted (published)": LG, "Logistic regression weighted (published)": LR}
if glob.glob(LRU[1]):
    runs["Logistic regression unweighted"] = LRU
res = pd.DataFrame([evaluate(k, *v) for k, v in runs.items()])
res.to_csv("transformer_weighting_results.csv", index=False)

# ---- paired contrasts on a common, aligned test set
def load(globpat):
    b, p, _ = load_pred(globpat); return b, p
bt, PW = load(W[1]); y = bt.label_id.to_numpy(); g = bt.patient_id.astype(str).to_numpy()
key = (bt.patient_id.astype(str) + "|" + bt.index_date.astype(str)).values
def aligned(globpat):
    b, p = load(globpat); k = (b.patient_id.astype(str) + "|" + b.index_date.astype(str)).values
    idx = pd.Series(range(len(k)), index=k).reindex(key).to_numpy()
    assert not np.isnan(idx).any(), f"{globpat} does not cover the same test instances"
    return p[idx.astype(int)]
def calibrated(val_glob, P_test):
    bv, Pv, _ = load_pred(val_glob); th = fit_vector_scaling(Pv, bv.label_id.to_numpy())
    return apply_vector_scaling(th, P_test)
PU = aligned(U[1]); PL = aligned(LG[1]); PR = aligned(LR[1])
M = {"raw": {"TW": PW, "TU": PU, "LGB": PL, "LR": PR},
     "cal": {"TW": calibrated(W[0], PW), "TU": calibrated(U[0], PU), "LGB": calibrated(LG[0], PL), "LR": calibrated(LR[0], PR)}}
if "Logistic regression unweighted" in runs:
    PRU = aligned(LRU[1]); M["raw"]["LRU"] = PRU; M["cal"]["LRU"] = calibrated(LRU[0], PRU)
S = ClusterSampler(g)
wors = lambda yy, P: auroc_k(yy, P, 2)
rows = []
pairs = [("TU", "TW"), ("LGB", "TU"), ("TU", "LR")] + ([("TU", "LRU")] if "LRU" in M["raw"] else [])
for scale in ("raw", "cal"):
    for a, b in pairs:
        for mname, fn in (("macro AUROC", auroc_macro), ("worsened AUROC", wors)):
            d, lo, hi, _p = paired_clustered(fn, y, M[scale][a], M[scale][b], g, sampler=S)
            rows.append({"scale": scale, "contrast": f"{a} minus {b}", "metric": mname, "diff": d, "lo": lo, "hi": hi})
C = pd.DataFrame(rows); C.to_csv("transformer_weighting_contrasts.csv", index=False)

# ---- verdict
r = res.set_index("model")
tw, tu = r.loc["Transformer weighted (published, 3 seeds)"], r.loc["Transformer unweighted (3 seeds)"]
c = C.set_index(["scale", "contrast", "metric"])
d_raw = c.loc[("raw", "TU minus TW", "macro AUROC")]; d_cal = c.loc[("cal", "TU minus TW", "macro AUROC")]
lgb_vs_tu = c.loc[("cal", "LGB minus TU", "macro AUROC")]; tu_vs_lr = c.loc[("cal", "TU minus LR", "macro AUROC")]
ranking_kept = lgb_vs_tu["lo"] > 0 and tu_vs_lr["lo"] > 0
material = abs(d_cal["diff"]) >= 0.005 or not ranking_kept
fmt = lambda s: f"{s['diff']:+.4f} ({s['lo']:+.4f} to {s['hi']:+.4f})"
lines = ["# Class-weighted versus unweighted transformer (V8 item 7)", "",
         f"Weighted transformer, macro AUROC raw {tw.raw_auroc_macro:.4f}, calibrated {tw.cal_auroc_macro:.4f}; worsened-class ECE after calibration {tw.cal_ece_worsened:.4f}.",
         f"Unweighted transformer, macro AUROC raw {tu.raw_auroc_macro:.4f}, calibrated {tu.cal_auroc_macro:.4f}; worsened-class ECE after calibration {tu.cal_ece_worsened:.4f}.",
         "", f"Unweighted minus weighted, macro AUROC: raw {fmt(d_raw)}; calibrated {fmt(d_cal)}.",
         f"LightGBM minus unweighted transformer (calibrated): {fmt(lgb_vs_tu)}.",
         f"Unweighted transformer minus weighted logistic regression (calibrated): {fmt(tu_vs_lr)}.", "",
         f"Model ranking LightGBM > transformer > baseline retained: {'YES' if ranking_kept else 'NO'}.",
         f"Manuscript update required: {'YES' if material else 'NO'} "
         "(rule: calibrated macro AUROC changes by 0.005 or more, or the ranking changes).", "",
         "If NO: add one sentence to Section 4.5 and a row to Supplementary Table S21 reporting these values, "
         "and replace 'the corresponding transformer run is provided as a script' in Section 2.12.",
         "If YES: send these three output files back for a full revision of Sections 2.12, 4.5, 4.7 and Table S21."]
open("transformer_weighting_summary.md", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))

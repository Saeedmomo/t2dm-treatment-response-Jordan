"""
Subgroup / fairness: discrimination (AUROC macro + worsened) and calibration (ECE mean, Brier)
on the temporal test set, stratified by sex, age band, baseline-HbA1c band, and history length
(short vs long sequence). Uses the transformer ensemble as the primary model, and reports LightGBM
alongside for reference. Output: Create_results/subgroup_metrics.csv + a figure.
"""
import os, sys, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from common import load_meta, PREDS, RES, FIG
import metrics_lib as M

meta = load_meta()
test_mask = (meta["split_temporal"]=="test").to_numpy()
mt = meta[test_mask].sort_values(["patient_id","index_date"]).reset_index(drop=True)

def load_pred(model):
    files = sorted(glob.glob(os.path.join(PREDS, f"{model}__temporal_test__seed*.parquet")))
    dfs = [pd.read_parquet(f).sort_values(["patient_id","index_date"]).reset_index(drop=True) for f in files]
    y = dfs[0]["label_id"].to_numpy()
    P = np.mean([d[["p_improved","p_stable","p_worsened"]].to_numpy() for d in dfs], axis=0)
    return y, P

def bh_band(v):
    if v < 7: return "<7"
    if v < 8: return "7-8"
    if v < 9: return "8-9"
    return ">=9"
def age_group(a):
    if a < 40: return "<40"
    if a < 55: return "40-55"
    if a < 70: return "55-70"
    return ">=70"

mt["bh_band"] = mt["baseline_hba1c"].map(bh_band)
mt["age_group"] = mt["age"].map(age_group)
mt["hist_len"] = np.where(mt["n_visits"] <= 3, "short (<=3 visits)", "long (>3 visits)")

def metrics_for(y, P):
    if len(np.unique(y)) < 3 or len(y) < 50:
        return dict(n=len(y), auroc_macro=np.nan, auroc_worsened=np.nan, ece_mean=np.nan, brier=np.nan,
                    pct_worsened=round((y==2).mean()*100,1))
    a = M.auroc_all(y, P)
    return dict(n=len(y), auroc_macro=round(a["auroc_macro"],4), auroc_worsened=round(a["auroc_worsened"],4),
                ece_mean=round(M.ece_ovr(y,P)["ece_mean"],4), brier=round(M.brier_multiclass(y,P),4),
                pct_worsened=round((y==2).mean()*100,1))

rows = []
for model_key, disp in [("transformer","Transformer"), ("lgbm","LightGBM")]:
    y, P = load_pred(model_key)
    for var, col in [("sex","sex"),("age_group","age_group"),("baseline_hba1c_band","bh_band"),("history_length","hist_len")]:
        for level in mt[col].dropna().unique():
            m = (mt[col]==level).to_numpy()
            r = metrics_for(y[m], P[m])
            rows.append({"model": disp, "variable": var, "subgroup": str(level), **r})
sg = pd.DataFrame(rows)
sg.to_csv(os.path.join(RES, "subgroup_metrics.csv"), index=False)
print(sg.to_string(index=False))

# figure: AUROC worsened by subgroup (transformer)
sgt = sg[sg["model"]=="Transformer"]
fig, axes = plt.subplots(1, 4, figsize=(17,4))
for ax,(var,title) in zip(axes, [("sex","Sex"),("age_group","Age band"),
                                 ("baseline_hba1c_band","Baseline HbA1c"),("history_length","History length")]):
    d = sgt[sgt["variable"]==var]
    ax.bar(d["subgroup"].astype(str), d["auroc_worsened"], color="#3b6ea5")
    ax.set_ylim(0.5, 0.9); ax.set_title(title); ax.set_ylabel("AUROC (worsened)")
    ax.axhline(sgt["auroc_worsened"].mean(), color="red", ls="--", lw=.8)
    for x,v,n in zip(range(len(d)), d["auroc_worsened"], d["n"]):
        ax.text(x, v+0.005, f"{v:.3f}\nn={n}", ha="center", fontsize=7)
    ax.tick_params(axis="x", rotation=20)
plt.suptitle("Transformer discrimination by subgroup — 'worsened' AUROC (temporal test)")
plt.tight_layout(); plt.savefig(os.path.join(FIG,"subgroup_auroc.png"), dpi=140); plt.close()
print("\nsubgroup.py DONE")

"""
Sensitivity analyses -> compact AUROC table (Create_results/sensitivity_auroc.csv).
Conditions:
  - primary (label +/-0.5, window 90-450, temporal split)         [all 3 models, from preds]
  - secondary grouped random split                                 [all 3 models, from preds]
  - label threshold +/-0.3 and +/-1.0 (vs primary +/-0.5)          [lgbm + logreg retrained]
  - follow-up window 90-365 (vs primary 90-450)                    [lgbm + logreg retrained]
  - pretraining ablation: transformer with vs without MLM          [from preds]

The strong comparator (LightGBM) and the logistic baseline are cheap to retrain, so they are
re-fit for every threshold/window variant; the transformer (expensive on CPU) is reported for the
conditions where trained predictions already exist.
"""
import os, sys, glob, numpy as np, pandas as pd
from scipy import sparse
sys.path.insert(0, os.path.dirname(__file__))
from common import load_meta, load_token_counts, PREDS
import metrics_lib as M
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

meta = load_meta()
Xtok, feat_names = load_token_counts()
meta["sex_code"] = meta["sex"].map({"FEMALE":0,"MALE":1}).fillna(-1).astype(int)
Xextra = sparse.csr_matrix(meta[["baseline_hba1c","age","sex_code","n_visits","seq_len"]].to_numpy(np.float32))
X_all = sparse.hstack([Xtok, Xextra], format="csr")
Xlr = meta[["baseline_hba1c","age","sex_code"]].to_numpy(np.float32)

def label_from_change(change, d):
    y = np.ones(len(change), dtype=int)  # stable
    y[change <= -d] = 0                  # improved
    y[change >=  d] = 2                  # worsened
    return y

def auroc_pair(y_true, P):
    a = M.auroc_all(y_true, P)
    return a["auroc_macro"], a["auroc_worsened"]

def fit_lgbm(tr, te, y):
    sw = compute_sample_weight("balanced", y[tr])
    dtr = lgb.Dataset(X_all[tr], label=y[tr], weight=sw)
    params = dict(objective="multiclass", num_class=3, metric="multi_logloss", learning_rate=0.05,
                  num_leaves=64, min_child_samples=100, feature_fraction=0.7, bagging_fraction=0.8,
                  bagging_freq=1, verbose=-1, num_threads=0, seed=42)
    model = lgb.train(params, dtr, num_boost_round=400)
    return model.predict(X_all[te])

def fit_logreg(tr, te, y):
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))])
    pipe.fit(Xlr[tr], y[tr]); return pipe.predict_proba(Xlr[te])

def load_pred(model, eval_name):
    files = sorted(glob.glob(os.path.join(PREDS, f"{model}__{eval_name}__seed*.parquet")))
    if not files: return None, None
    dfs = [pd.read_parquet(f).sort_values(["patient_id","index_date"]).reset_index(drop=True) for f in files]
    y = dfs[0]["label_id"].to_numpy()
    P = np.mean([d[["p_improved","p_stable","p_worsened"]].to_numpy() for d in dfs], axis=0)
    return y, P

rows = []
def add(cond, model, y, P):
    if y is None:
        rows.append({"condition": cond, "model": model, "auroc_macro": np.nan, "auroc_worsened": np.nan}); return
    ma, wo = auroc_pair(y, P)
    rows.append({"condition": cond, "model": model, "auroc_macro": round(ma,4), "auroc_worsened": round(wo,4)})

tr_t = (meta["split_temporal"]=="train").to_numpy()
te_t = (meta["split_temporal"]=="test").to_numpy()
tr_r = (meta["split_random"]=="train").to_numpy()
te_r = (meta["split_random"]=="test").to_numpy()

# ---- primary (from preds) ----
for m,disp in [("transformer","Transformer"),("lgbm","LightGBM"),("logreg","LogReg")]:
    y,P = load_pred(m,"temporal_test"); add("primary (+/-0.5, 90-450, temporal)", disp, y, P)

# ---- random split (from preds) ----
for m,disp in [("transformer","Transformer"),("lgbm","LightGBM"),("logreg","LogReg")]:
    y,P = load_pred(m,"random_test"); add("secondary grouped-random split", disp, y, P)

# ---- threshold variants (retrain fast models on temporal split) ----
for d in [0.3, 1.0]:
    yv = label_from_change(meta["change"].to_numpy(), d)
    add(f"label threshold +/-{d} (temporal)", "LightGBM", yv[te_t], fit_lgbm(tr_t, te_t, yv))
    add(f"label threshold +/-{d} (temporal)", "LogReg",   yv[te_t], fit_logreg(tr_t, te_t, yv))

# ---- window 90-365 (retrain fast models on the subset, temporal split) ----
sub = (meta["days_between"] <= 365).to_numpy()
y05 = label_from_change(meta["change"].to_numpy(), 0.5)
tr_w = tr_t & sub; te_w = te_t & sub
add("follow-up window 90-365 (temporal)", "LightGBM", y05[te_w], fit_lgbm(tr_w, te_w, y05))
add("follow-up window 90-365 (temporal)", "LogReg",   y05[te_w], fit_logreg(tr_w, te_w, y05))

# ---- pretraining ablation (from preds) ----
for m,disp in [("transformer","Transformer WITH MLM pretrain"),
               ("transformer_nopretrain","Transformer WITHOUT MLM pretrain")]:
    y,P = load_pred(m,"temporal_test"); add("pretraining ablation (temporal)", disp, y, P)

tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(os.path.dirname(PREDS), "sensitivity_auroc.csv"), index=False)
print(tab.to_string(index=False))
print("\nsensitivity.py DONE")

"""
Model 2 (LightGBM on flattened tabular features) and Model 3 (logistic regression on
baseline HbA1c + age + sex only). Trains on the PRIMARY temporal train split, predicts on
temporal validation + test. Also fits on the SECONDARY random split for the sensitivity check.
Saves 3-class predicted probabilities to Create_results/preds/.

Leakage guard: days_between (timing of the follow-up draw) is an OUTCOME property, not known at
the index draw, so it is NOT used as a predictor by any model.
"""
import os, sys, json, numpy as np, pandas as pd
from scipy import sparse
sys.path.insert(0, os.path.dirname(__file__))
from common import load_meta, load_token_counts, save_preds, PREP, RES, CLASSES
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

meta = load_meta()
Xtok, feat_names = load_token_counts()

# extra numeric/categorical tabular features known at index time
meta["sex_code"] = meta["sex"].map({"FEMALE":0,"MALE":1}).fillna(-1).astype(int)
extra_cols = ["baseline_hba1c","age","sex_code","n_visits","seq_len"]
Xextra = sparse.csr_matrix(meta[extra_cols].to_numpy(dtype=np.float32))
X_all = sparse.hstack([Xtok, Xextra], format="csr")
# LightGBM forbids special JSON chars in feature names -> sanitise [CLS]->SPECIAL_CLS etc.
def sanitize(n):
    return (n.replace("[","SPECIAL_").replace("]","").replace('"',"").replace(":","_")
             .replace(",","_").replace("{","").replace("}",""))
all_feat_names = [sanitize(n) for n in (feat_names + extra_cols)]
y = meta["label_id"].to_numpy()

def masks(split_col):
    return (meta[split_col]=="train").to_numpy(), (meta[split_col]=="validation").to_numpy(), (meta[split_col]=="test").to_numpy()

# ---------------- LightGBM ----------------
def run_lgbm(split_col, tag):
    tr, va, te = masks(split_col)
    sw = compute_sample_weight("balanced", y[tr])
    dtr = lgb.Dataset(X_all[tr], label=y[tr], weight=sw, feature_name=all_feat_names)
    dva = lgb.Dataset(X_all[va], label=y[va], reference=dtr)
    params = dict(objective="multiclass", num_class=3, metric="multi_logloss",
                  learning_rate=0.05, num_leaves=64, min_child_samples=100,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
                  max_depth=-1, verbose=-1, num_threads=0, seed=42)
    model = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)])
    for name, m in [("validation", va), ("test", te)]:
        p = model.predict(X_all[m])
        save_preds("lgbm", f"{tag}_{name}", meta[m], p, seed=0)
    if tag == "temporal":
        imp = pd.DataFrame({"feature": all_feat_names,
                            "gain": model.feature_importance("gain"),
                            "split": model.feature_importance("split")}).sort_values("gain", ascending=False)
        imp.to_csv(os.path.join(RES, "lgbm_feature_importance.csv"), index=False)
        model.save_model(os.path.join(RES, "lgbm_temporal.txt"))
    print(f"  LGBM [{tag}] done (best_iter={model.best_iteration})")

# ---------------- Logistic regression (baseline HbA1c + age + sex only) ----------------
def run_logreg(split_col, tag):
    tr, va, te = masks(split_col)
    feats = meta[["baseline_hba1c","age","sex_code"]].to_numpy(dtype=np.float32)
    pipe = Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))])
    pipe.fit(feats[tr], y[tr])
    for name, m in [("validation", va), ("test", te)]:
        p = pipe.predict_proba(feats[m])
        save_preds("logreg", f"{tag}_{name}", meta[m], p, seed=0)
    print(f"  LogReg [{tag}] done")

if __name__ == "__main__":
    print("Training LightGBM + Logistic Regression on PRIMARY temporal split ...")
    run_lgbm("split_temporal", "temporal")
    run_logreg("split_temporal", "temporal")
    print("Training on SECONDARY random split (sensitivity) ...")
    run_lgbm("split_random", "random")
    run_logreg("split_random", "random")
    print("DONE baselines.")

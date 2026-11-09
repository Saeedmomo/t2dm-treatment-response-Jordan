"""Shared helpers: paths, loading prepped data, saving predictions."""
import os, json, numpy as np, pandas as pd
from scipy import sparse

BASE = r"D:\Hakeem_compitition\new_code"
RES  = os.path.join(BASE, "Create_results")
PREP = os.path.join(RES, "prep")
PREDS= os.path.join(RES, "preds"); os.makedirs(PREDS, exist_ok=True)
FIG  = os.path.join(BASE, "Create_figures")
CLASSES = ["improved", "stable", "worsened"]
ID2LAB = {0:"improved", 1:"stable", 2:"worsened"}

def load_meta():
    return pd.read_parquet(os.path.join(PREP, "meta.parquet"))

def load_token_counts():
    X = sparse.load_npz(os.path.join(PREP, "token_counts.npz"))
    names = json.load(open(os.path.join(PREP, "token_feature_names.json")))
    return X, names

def load_transformer_arrays():
    ids = np.load(os.path.join(PREP, "input_ids.npy"))
    mask = np.load(os.path.join(PREP, "attn_mask.npy"))
    lab = np.load(os.path.join(PREP, "labels.npy"))
    return ids, mask, lab

def save_preds(model, eval_name, meta_sub, proba, seed=0):
    """proba: (n,3). meta_sub aligned rows with patient_id/index_date/label_id."""
    df = meta_sub[["patient_id","index_date","label_id"]].reset_index(drop=True).copy()
    df["p_improved"] = proba[:,0]; df["p_stable"] = proba[:,1]; df["p_worsened"] = proba[:,2]
    df["model"] = model; df["eval"] = eval_name; df["seed"] = seed
    path = os.path.join(PREDS, f"{model}__{eval_name}__seed{seed}.parquet")
    df.to_parquet(path)
    return path

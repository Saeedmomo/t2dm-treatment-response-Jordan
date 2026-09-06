"""
Shared feature/data preparation so ALL three models train & evaluate on the identical rows.

Reads:  Create_results/sequences.jsonl        (tokens + label + numeric targets + temporal split)
        Create_results/splits_random_grouped.jsonl
        Dataset/Sex_birth/date of birth.csv    (numeric age)
Writes (Create_results/prep/):
        meta.parquet        one row per pair: ids, both splits, label, change, numerics, static feats
        token_counts.npz    sparse CSR (n_pairs x 175) token-count matrix + feature names (LGBM)
        input_ids.npy       int16 (n_pairs x MAX_LEN) padded token ids (transformer)
        attn_mask.npy       int8  (n_pairs x MAX_LEN)
        labels.npy          int8  (n_pairs,)  0=improved 1=stable 2=worsened
        row_index.parquet   patient_id, index_date  (row order key, aligned with all arrays)
Row order is IDENTICAL across every array and meta.parquet.
"""
import os, json, numpy as np, pandas as pd
from scipy import sparse

BASE = r"D:\Hakeem_compitition\new_code"
RES  = os.path.join(BASE, "Create_results")
OUT  = os.path.join(RES, "prep"); os.makedirs(OUT, exist_ok=True)
VOCAB = json.load(open(os.path.join(BASE, "Example_transformer_model", "vocab.json"), encoding="utf-8"))
MAX_LEN = 128
LABEL2ID = {"improved":0, "stable":1, "worsened":2}

# random-split lookup keyed by (patient_id, index_date)
rand = {}
with open(os.path.join(RES, "splits_random_grouped.jsonl")) as f:
    for line in f:
        o = json.loads(line); rand[(o["patient_id"], o["index_date"])] = o["split"]

# DOB for numeric age
dob = pd.read_csv(os.path.join(BASE, "Dataset", "Sex_birth", "date of birth.csv"), encoding="utf-8-sig")
dob = dob.rename(columns={"PATIENT_ID":"pid","DATE_OF_BIRTH":"dob"})
dob["dob"] = pd.to_datetime(dob["dob"], errors="coerce")
dob_map = dict(zip(dob["pid"].astype(str), dob["dob"]))

n_vocab = len(VOCAB)
rows_meta = []
data, indices, indptr = [], [], [0]        # CSR accumulation for token counts
ids_arr, mask_arr, lab_arr = [], [], []

def age_band_from_token(tokens):
    for t in tokens:
        if t.startswith("DEM_AGE_"): return t.replace("DEM_AGE_","")
    return "MISSING"

N = 0
with open(os.path.join(RES, "sequences.jsonl"), encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        toks = o["tokens"]
        pid, idate = o["patient_id"], o["index_date"]
        # ---- token-count row (sparse) ----
        counts = {}
        for t in toks:
            tid = VOCAB.get(t)
            if tid is not None:
                counts[tid] = counts.get(tid, 0) + 1
        for k in sorted(counts):
            indices.append(k); data.append(counts[k])
        indptr.append(len(indices))
        # ---- transformer input ids (truncate keeping [CLS] + most-recent tail) ----
        tids = [VOCAB.get(t, 1) for t in toks]     # 1 = [UNK]
        if len(tids) > MAX_LEN:
            tids = [tids[0]] + tids[-(MAX_LEN-1):]
        att = [1]*len(tids)
        if len(tids) < MAX_LEN:
            pad = MAX_LEN - len(tids)
            tids = tids + [0]*pad; att = att + [0]*pad
        ids_arr.append(tids); mask_arr.append(att)
        lab_arr.append(LABEL2ID[o["label"]])
        # ---- static / meta ----
        sex = "MISSING"
        for t in toks:
            if t == "DEM_SEX_FEMALE": sex = "FEMALE"
            elif t == "DEM_SEX_MALE": sex = "MALE"
        dobv = dob_map.get(pid)
        age = (pd.Timestamp(idate) - dobv).days/365.25 if pd.notna(dobv) else np.nan
        n_visits = toks.count("[EV]")
        rows_meta.append({
            "patient_id": pid, "index_date": idate,
            "split_temporal": o.get("split"),
            "split_random": rand.get((pid, idate)),
            "label": o["label"], "label_id": LABEL2ID[o["label"]],
            "change": o["change"], "days_between": o["days_between"],
            "baseline_hba1c": o["baseline_hba1c"], "target_hba1c": o["target_hba1c"],
            "age": age, "sex": sex, "age_band": age_band_from_token(toks),
            "n_visits": n_visits, "seq_len": len(toks),
        })
        N += 1
        if N % 100000 == 0: print(f"  processed {N:,}")

meta = pd.DataFrame(rows_meta)
meta.to_parquet(os.path.join(OUT, "meta.parquet"))
meta[["patient_id","index_date"]].to_parquet(os.path.join(OUT, "row_index.parquet"))

X = sparse.csr_matrix((data, indices, indptr), shape=(N, n_vocab), dtype=np.int32)
feat_names = [None]*n_vocab
for t, i in VOCAB.items(): feat_names[i] = t
sparse.save_npz(os.path.join(OUT, "token_counts.npz"), X)
json.dump(feat_names, open(os.path.join(OUT, "token_feature_names.json"), "w"))

np.save(os.path.join(OUT, "input_ids.npy"), np.asarray(ids_arr, dtype=np.int16))
np.save(os.path.join(OUT, "attn_mask.npy"), np.asarray(mask_arr, dtype=np.int8))
np.save(os.path.join(OUT, "labels.npy"), np.asarray(lab_arr, dtype=np.int8))

print(f"\nDONE. N={N:,}  token-count matrix {X.shape} nnz={X.nnz:,}")
print("temporal split counts:\n", meta["split_temporal"].value_counts(dropna=False))
print("random split counts:\n", meta["split_random"].value_counts(dropna=False))
print("age missing:", int(meta['age'].isna().sum()), "| sex counts:", meta['sex'].value_counts().to_dict())

"""
Jordan EHR -- Step 2 of the pipeline: cleaning, cohort building, label creation,
sequence assembly, and the SPLIT.  NO model is trained here.

Outputs (all under Create_results/):
  cleaned_hba1c.parquet              cleaned + deduped HbA1c draws (date-level)
  cohort_pairs.csv                   labelled (index, target) HbA1c pairs
  sequences.jsonl                    assembled token sequences (reusing vocab.json)
  splits_temporal.jsonl              PRIMARY split: temporal + patient-disjoint
  splits_random_grouped.jsonl        SECONDARY split: patient-grouped stratified 70/15/15
  cleaning_report.txt                counts removed at each cleaning step
  cohort_summary.md                  the summary to review before tokenise/train

Design notes are in cohort_summary.md.
"""
import os, json, glob, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

BASE = r"D:\Hakeem_compitition\new_code"
DATA = os.path.join(BASE, "Dataset")
LAB  = os.path.join(DATA, "lab_tests")
RESULTS = os.path.join(BASE, "Create_results")
VOCAB_PATH = os.path.join(BASE, "Example_transformer_model", "vocab.json")
os.makedirs(RESULTS, exist_ok=True)

VOCAB = json.load(open(VOCAB_PATH, encoding="utf-8"))
VOCAB_SET = set(VOCAB)

rep = []
def log(s=""):
    print(s); rep.append(str(s))

# id column is 'PatientID' in labs, 'PATIENT_ID' elsewhere -> normalise to 'patient_id'
def read_csv(p, **kw):
    return pd.read_csv(p, encoding="utf-8-sig", **kw)

# ============================================================
# STEP 1: combine batch-split analyte files, normalise id, fix "NA"
# ============================================================
log("="*90); log("STEP 1  Combine batch files, normalise id column, fix NA sodium parsing"); log("="*90)

# Only the 7 analytes that exist in vocab.json are needed as sequence features (+ HbA1c is the target).
LAB_FILES = {
    "HBA1C":            ["HBA1C 11.csv", "HBA1C 22.csv"],
    "FBS":              ["FBS 11.csv", "FBS 22.csv"],
    "CREATININE":       ["CRT 11.csv", "CRT 22.csv", "CRT 33.csv"],
    "BUN":              ["BUN 11.csv"],
    "LDL":              ["LDL D .csv"],
    "TRIGLYCERIDES":    ["TRIG 1 D.csv", "TRIG 2 D.csv"],
    "TOTAL_CHOLESTEROL":["CHOL 1 D.xlsx", "CHOL 2 D.xlsx"],
}

def load_analyte(analyte, files):
    frames = []
    for fn in files:
        p = os.path.join(LAB, fn)
        # keep_default_na=False stops pandas turning the sodium code "NA" (and, defensively,
        # any 'NA' text) into a null; we then coerce the numeric result column ourselves.
        if fn.endswith(".xlsx"):
            d = pd.read_excel(p)
        else:
            d = pd.read_csv(p, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
        d.columns = [c.strip() for c in d.columns]
        d = d.rename(columns={"PatientID": "patient_id",
                              "LAB_DATA_CHEMISTRY_RESULTS": "value",
                              "DATE_TIME_SPECIMEN_TAKEN": "ts"})
        d = d[["patient_id", "value", "ts"]].copy()
        d["value"] = pd.to_numeric(d["value"], errors="coerce")
        d["ts"] = pd.to_datetime(d["ts"], errors="coerce")
        d["patient_id"] = pd.to_numeric(d["patient_id"], errors="coerce").astype("Int64")
        d["analyte"] = analyte
        d["source_file"] = fn
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    return out

labs = {a: load_analyte(a, f) for a, f in LAB_FILES.items()}
for a, d in labs.items():
    log(f"  {a:18s} loaded {len(d):>9,} rows from {LAB_FILES[a]}")

# Verify the sodium NA fix on a control file (sodium itself is not a tracked analyte,
# but we confirm the parsing approach works).
na_check = pd.read_csv(os.path.join(LAB, "NA 11.csv"), encoding="utf-8-sig",
                       keep_default_na=False, na_values=[""])
na_check.columns = [c.strip() for c in na_check.columns]
log(f"\n  NA sodium parsing fix: LabTestName unique values now = "
    f"{sorted(na_check['LabTestName'].unique())[:3]} (literal 'NA' preserved, not nulled)")

# ============================================================
# STEP 2: clean HbA1c and (with lab-appropriate ranges) the other feature labs
# ============================================================
log("\n" + "="*90); log("STEP 2  Cleaning (plausible-range filter + exact-duplicate removal)"); log("="*90)

# Creatinine has MIXED units within files (mg/dL and umol/L). Serum creatinine in mg/dL is
# essentially never >15; in umol/L it is routinely 40-1200. Convert values >=15 to mg/dL.
UMOL_TO_MGDL = 1/88.42
def normalise_creatinine(d):
    hi = d["value"] >= 15
    d.loc[hi, "value"] = d.loc[hi, "value"] * UMOL_TO_MGDL
    return d, int(hi.sum())

labs["CREATININE"], n_crt_conv = normalise_creatinine(labs["CREATININE"])
log(f"  Creatinine unit normalisation: {n_crt_conv:,} values >=15 treated as umol/L and /88.42 -> mg/dL")

# Plausible ranges. HbA1c is the [2,20]% specified by the user; the others use the SAME
# idea (drop physiologically impossible values) with lab-appropriate bounds in canonical units.
PLAUSIBLE = {
    "HBA1C":            (2.0, 20.0),     # %
    "FBS":              (20.0, 900.0),   # mg/dL
    "CREATININE":       (0.1, 20.0),     # mg/dL (post unit-normalisation)
    "BUN":              (1.0, 200.0),    # mg/dL
    "LDL":              (10.0, 400.0),   # mg/dL
    "TRIGLYCERIDES":    (10.0, 3000.0),  # mg/dL
    "TOTAL_CHOLESTEROL":(40.0, 700.0),   # mg/dL
}

clean_rows = []
for a, d in labs.items():
    n0 = len(d)
    n_missing_val = int(d["value"].isna().sum())
    n_missing_ts  = int(d["ts"].isna().sum())
    lo, hi = PLAUSIBLE[a]
    d = d.dropna(subset=["value", "ts"])
    n_after_missing = len(d)
    inrange = d["value"].between(lo, hi)
    n_out = int((~inrange).sum())
    d = d[inrange]
    n_after_range = len(d)
    # exact duplicate (patient, timestamp, value)
    n_dupe = int(d.duplicated(subset=["patient_id", "ts", "value"]).sum())
    d = d.drop_duplicates(subset=["patient_id", "ts", "value"])
    labs[a] = d.reset_index(drop=True)
    clean_rows.append({"analyte": a, "raw_rows": n0, "missing_value": n_missing_val,
                       "missing_ts": n_missing_ts, "out_of_range_[{},{}]".format(lo,hi): n_out,
                       "exact_dupes": n_dupe, "kept": len(labs[a])})
    log(f"  {a:18s} raw={n0:>9,}  -missing_val={n_missing_val:>6,}  -missing_ts={n_missing_ts:>5,}  "
        f"-out_of_range[{lo},{hi}]={n_out:>6,}  -dupes={n_dupe:>6,}  ->kept={len(labs[a]):>9,}")

clean_df = pd.DataFrame(clean_rows)
clean_df.to_csv(os.path.join(RESULTS, "cleaning_counts_by_analyte.csv"), index=False)

hba1c = labs["HBA1C"].copy()
log(f"\n  HbA1c after cleaning: {len(hba1c):,} measurements, {hba1c['patient_id'].nunique():,} patients")

# Aggregate HbA1c to date-level (one value per patient-day = mean) so that (patient_id, index_date)
# is unique -- required by the {patient_id, index_date, split} split format.
hba1c["date"] = hba1c["ts"].dt.normalize()
hba1c_day = (hba1c.groupby(["patient_id", "date"], as_index=False)["value"].mean()
                  .rename(columns={"value": "hba1c"}))
log(f"  HbA1c collapsed to date-level draws: {len(hba1c_day):,} patient-day draws "
    f"(same-day values averaged)")
hba1c_day.to_parquet(os.path.join(RESULTS, "cleaned_hba1c.parquet"))

# Collapse the 6 feature labs to date-level (mean) too, for the as-of sequence lookups.
feat_labs = {}
for a in ["FBS", "CREATININE", "BUN", "LDL", "TRIGLYCERIDES", "TOTAL_CHOLESTEROL"]:
    d = labs[a].copy()
    d["date"] = d["ts"].dt.normalize()
    feat_labs[a] = (d.groupby(["patient_id", "date"], as_index=False)["value"].mean())

# ============================================================
# STEP 3: build labelled (index -> next-in-window) HbA1c pairs
# ============================================================
log("\n" + "="*90); log("STEP 3  Build labelled pairs (next HbA1c 90-450 days later)"); log("="*90)

MIN_GAP, MAX_GAP = 90, 450
DELTA = 0.5

pairs = []
for pid, g in hba1c_day.sort_values(["patient_id", "date"]).groupby("patient_id", sort=False):
    dates = g["date"].values.astype("datetime64[D]")
    vals  = g["hba1c"].values
    n = len(dates)
    for i in range(n):
        gaps = (dates[i+1:] - dates[i]).astype("timedelta64[D]").astype(int)
        # earliest subsequent draw whose gap is within [MIN_GAP, MAX_GAP]
        ok = np.where((gaps >= MIN_GAP) & (gaps <= MAX_GAP))[0]
        if len(ok) == 0:
            continue
        j = i + 1 + ok[0]
        change = float(vals[j] - vals[i])
        if change <= -DELTA:  label = "improved"
        elif change >= DELTA: label = "worsened"
        else:                 label = "stable"
        pairs.append((int(pid), pd.Timestamp(dates[i]), pd.Timestamp(dates[j]),
                      float(vals[i]), float(vals[j]), change,
                      int((dates[j]-dates[i]).astype(int)), label))

pairs_df = pd.DataFrame(pairs, columns=["patient_id","index_date","target_date","baseline_hba1c",
                                        "target_hba1c","change","days_between","label"])
log(f"  Labelled pairs: {len(pairs_df):,} from {pairs_df['patient_id'].nunique():,} patients")
log(f"  Class counts (all pairs):")
for k, v in pairs_df["label"].value_counts().items():
    log(f"     {k:9s}: {v:>8,} ({v/len(pairs_df)*100:5.1f}%)")
log(f"  days_between: min={pairs_df.days_between.min()}, median={pairs_df.days_between.median():.0f}, max={pairs_df.days_between.max()}")
log(f"  change: mean={pairs_df.change.mean():.3f}, median={pairs_df.change.median():.3f}")

# ============================================================
# load demographics / medications / diagnoses for sequence assembly
# ============================================================
log("\n" + "="*90); log("STEP 4  Assemble token sequences (strictly-before history; reuse vocab.json)"); log("="*90)

# ---- demographics ----
dob = read_csv(os.path.join(DATA, "Sex_birth", "date of birth.csv")).rename(
        columns={"PATIENT_ID":"patient_id","DATE_OF_BIRTH":"dob"})
dob["dob"] = pd.to_datetime(dob["dob"], errors="coerce")
dob_map = dict(zip(dob["patient_id"], dob["dob"]))
sex_map = {}
for fn, s in [("females patients.csv","FEMALE"), ("males patients.csv","MALE")]:
    d = read_csv(os.path.join(DATA, "Sex_birth", fn))
    for pid in d["PATIENT_ID"].values:
        sex_map[pid] = s

# ---- medications: map to the 7 MED_ON classes in the vocab ----
def med_class(name):
    n = str(name).upper()
    if "METFORMIN" in n: return "METFORMIN"
    if any(s in n for s in ["GLIBENCLAMIDE","GLICLAZIDE","GLIMEPIRIDE"]): return "SULFONYLUREA"
    if "GLIPTIN" in n: return "DPP4"
    if "GLIFLOZIN" in n: return "SGLT2"
    if "GLUTIDE" in n: return "GLP1"
    if "INSULIN" in n:
        if any(s in n for s in ["GLARGINE","DETEMIR","DEGLUDEC","NPH"]): return "INSULIN_BASAL"
        return "INSULIN_BOLUS"   # aspart/lispro/glulisine/regular/premix -> bolus
    return None

med_frames = []
for f in glob.glob(os.path.join(DATA, "Medications", "M*.csv")):
    d = read_csv(f, usecols=["PATIENT_ID","DISPENSED_DATE","GENERIC_NAME"])
    med_frames.append(d)
meds = pd.concat(med_frames, ignore_index=True).rename(
        columns={"PATIENT_ID":"patient_id","DISPENSED_DATE":"date","GENERIC_NAME":"name"})
meds["date"] = pd.to_datetime(meds["date"], errors="coerce")
meds["mclass"] = meds["name"].map(med_class)
meds = meds.dropna(subset=["date","mclass"])
log(f"  Medications mapped to classes: {len(meds):,} dispenses, classes={sorted(meds['mclass'].unique())}")

# ---- diagnoses: dxcount + comorbidity onset (COM_HAS_*) ----
dx = read_csv(os.path.join(DATA, "Diagnosis", "DIAGNOSIS wide.csv")).rename(
        columns={"PATIENT_ID":"patient_id","DATE_ENTERED":"date","DIAGNOSIS":"text","CODE_NUMBER":"code"})
dx["date"] = pd.to_datetime(dx["date"], errors="coerce")
dx = dx.dropna(subset=["date"])
dx["text"] = dx["text"].astype(str).str.upper()

def comorbid(text):
    t = text
    coms = []
    if "HYPERTENSION" in t: coms.append("HTN")
    if any(s in t for s in ["HYPERLIPID","DYSLIPID","HYPERCHOLEST","HYPERTRIGLYCERID","LIPID METAB"]): coms.append("DYSLIPIDEMIA")
    if any(s in t for s in ["CHRONIC KIDNEY","CKD","CHRONIC RENAL","NEPHROPATHY","RENAL FAILURE","ESRD"]): coms.append("CKD")
    if any(s in t for s in ["CORONARY","MYOCARD","ISCHEMIC HEART","ATHEROSCLER","ANGINA","CEREBROVASC","STROKE","PERIPHERAL VASCULAR","CAROTID"]): coms.append("ASCVD")
    if any(s in t for s in ["HEART FAILURE","CHF","CARDIOMYOPATHY"]): coms.append("HF")
    if any(s in t for s in ["FATTY LIVER","STEATOHEP","STEATOSIS","NASH","NAFLD","MASLD"]): coms.append("MASLD")
    return coms

dx["coms"] = dx["text"].map(comorbid)

# ---------------- token binning helpers (ALL must be in vocab.json) ----------------
def bin_hba1c(v):
    if v<=6.0: return "LE_6_0"
    if v<=7.0: return "6_0_7_0"
    if v<=8.0: return "7_0_8_0"
    if v<=9.0: return "8_0_9_0"
    if v<=10.0: return "9_0_10_0"
    if v<=12.0: return "10_0_12_0"
    return "GT_12_0"
def bin_fbs(v):
    if v<=110: return "LE_110"
    if v<=130: return "110_130"
    if v<=160: return "130_160"
    if v<=200: return "160_200"
    if v<=400: return "200_400"
    return "GT_400"
def bin_creatinine(v):
    if v<=0.6: return "LE_0_6"
    if v<=0.9: return "0_6_0_9"
    if v<=1.2: return "0_9_1_2"
    if v<=2.0: return "1_2_2_0"
    return "GT_2_0"
def bin_bun(v):
    if v<=7: return "LE_7"
    if v<=20: return "7_20"
    if v<=30: return "20_30"
    if v<=50: return "30_50"
    return "GT_50"
def bin_ldl(v):
    if v<=70: return "LE_70"
    if v<=100: return "70_100"
    if v<=130: return "100_130"
    if v<=160: return "130_160"
    if v<=190: return "160_190"
    return "GT_190"
def bin_trig(v):
    if v<=150: return "LE_150"
    if v<=200: return "150_200"
    if v<=300: return "200_300"
    if v<=500: return "300_500"
    return "GT_500"
def bin_chol(v):
    if v<=150: return "LE_150"
    if v<=200: return "150_200"
    if v<=240: return "200_240"
    if v<=300: return "240_300"
    return "GT_300"
LAB_BINNER = {"HBA1C":bin_hba1c,"FBS":bin_fbs,"CREATININE":bin_creatinine,"BUN":bin_bun,
              "LDL":bin_ldl,"TRIGLYCERIDES":bin_trig,"TOTAL_CHOLESTEROL":bin_chol}
LAB_ORDER = ["HBA1C","FBS","CREATININE","BUN","LDL","TRIGLYCERIDES","TOTAL_CHOLESTEROL"]

def labage_bin(days):
    if days<=30: return "0_30D"
    if days<=90: return "31_90D"
    if days<=180: return "91_180D"
    return "GT180D"
def gap_bin(days):
    if days<=7: return "0_7D"
    if days<=30: return "8_30D"
    if days<=90: return "1_3M"
    if days<=180: return "3_6M"
    if days<=365: return "6_12M"
    return "GT1Y"
def dxcount_bin(c):
    if c==0: return "DXCOUNT_0"
    if c<=2: return "DXCOUNT_1_2"
    if c<=5: return "DXCOUNT_3_5"
    return "DXCOUNT_GT5"
def age_bin(a):
    if a is None or np.isnan(a): return "DEM_AGE_MISSING"
    if a<=30: return "DEM_AGE_LE_30"
    if a<=40: return "DEM_AGE_30_40"
    if a<=50: return "DEM_AGE_40_50"
    if a<=60: return "DEM_AGE_50_60"
    if a<=70: return "DEM_AGE_60_70"
    if a<=80: return "DEM_AGE_70_80"
    if a<=90: return "DEM_AGE_80_90"
    return "DEM_AGE_GT_90"

MED_LOOKBACK = 180  # days: a class is "on" if dispensed within this window before the visit
MAX_VISITS = 20

# ---- pre-index per-patient structures for the patients that appear in pairs ----
pair_pids = set(pairs_df["patient_id"].unique())

# per-patient sorted feature-lab arrays: {pid: {analyte: (dates[], tokens[])}}
def build_lab_index(feat_labs, hba1c_day):
    idx = {}
    # HbA1c as a feature too (its own history)
    hb = hba1c_day[hba1c_day["patient_id"].isin(pair_pids)].sort_values(["patient_id","date"])
    for pid, g in hb.groupby("patient_id", sort=False):
        idx.setdefault(pid, {})["HBA1C"] = (g["date"].values.astype("datetime64[D]"),
                                            np.array([f"NUM_HBA1C_{bin_hba1c(v)}" for v in g["hba1c"].values]))
    for a in ["FBS","CREATININE","BUN","LDL","TRIGLYCERIDES","TOTAL_CHOLESTEROL"]:
        d = feat_labs[a]
        d = d[d["patient_id"].isin(pair_pids)].sort_values(["patient_id","date"])
        b = LAB_BINNER[a]
        for pid, g in d.groupby("patient_id", sort=False):
            idx.setdefault(pid, {})[a] = (g["date"].values.astype("datetime64[D]"),
                                          np.array([f"NUM_{a}_{b(v)}" for v in g["value"].values]))
    return idx
lab_idx = build_lab_index(feat_labs, hba1c_day)

# per-patient med dispense dates by class
med_idx = {}
mm = meds[meds["patient_id"].isin(pair_pids)].sort_values(["patient_id","date"])
for (pid, mclass), g in mm.groupby(["patient_id","mclass"], sort=False):
    med_idx.setdefault(pid, {})[mclass] = g["date"].values.astype("datetime64[D]")

# per-patient diagnosis dates + comorbidity onset dates
dxx = dx[dx["patient_id"].isin(pair_pids)].sort_values(["patient_id","date"])
dx_dates = {}
com_onset = {}
for pid, g in dxx.groupby("patient_id", sort=False):
    dx_dates[pid] = g["date"].values.astype("datetime64[D]")
    onset = {}
    for dt, coms in zip(g["date"].values.astype("datetime64[D]"), g["coms"].values):
        for c in coms:
            if c not in onset: onset[c] = dt
            elif dt < onset[c]: onset[c] = dt
    com_onset[pid] = onset

# visit dates per patient = union of all tracked-lab dates (sorted unique)
visit_dates = {}
for pid, labmap in lab_idx.items():
    alld = np.concatenate([arr[0] for arr in labmap.values()])
    visit_dates[pid] = np.unique(alld)

MED_CLASSES = ["METFORMIN","SULFONYLUREA","DPP4","SGLT2","GLP1","INSULIN_BASAL","INSULIN_BOLUS"]

def asof_token(arr_dates, arr_tokens, ref):
    # most recent index with date <= ref
    pos = np.searchsorted(arr_dates, ref, side="right") - 1
    if pos < 0: return None, None
    return arr_tokens[pos], arr_dates[pos]

def build_sequence(pid, index_date):
    ref = np.datetime64(index_date, "D")
    labmap = lab_idx.get(pid, {})
    vdates = visit_dates.get(pid, np.array([], dtype="datetime64[D]"))
    vdates = vdates[vdates <= ref]
    if len(vdates) > MAX_VISITS:
        vdates = vdates[-MAX_VISITS:]
    toks = ["[CLS]"]
    meds_p = med_idx.get(pid, {})
    dxd = dx_dates.get(pid, None)
    onset = com_onset.get(pid, {})
    for k, vd in enumerate(vdates):
        toks.append("[EV]")
        # medications active as-of vd (dispense within lookback window)
        active = []
        for c in MED_CLASSES:
            dd = meds_p.get(c)
            if dd is None: continue
            hi = np.searchsorted(dd, vd, side="right")
            if hi > 0 and (vd - dd[hi-1]).astype(int) <= MED_LOOKBACK:
                active.append(c)
        if active:
            n = min(len(active), 6)
            toks.append(f"THERAPY_NAGENTS_{n}")
            toks.append("THERAPY_MONOCOMBO_1" if len(active) >= 2 else "THERAPY_MONOCOMBO_0")
            for c in active:
                toks.append(f"MED_ON_{c}")
        # diagnosis count as-of vd
        if dxd is not None:
            dxc = int(np.searchsorted(dxd, vd, side="right"))
        else:
            dxc = 0
        toks.append(dxcount_bin(dxc))
        for c in ["HTN","DYSLIPIDEMIA","CKD","ASCVD","HF","MASLD"]:
            if c in onset and onset[c] <= vd:
                toks.append(f"COM_HAS_{c}")
        # 7 labs: most-recent value as-of vd + recency
        for a in LAB_ORDER:
            if a in labmap:
                tok, ld = asof_token(labmap[a][0], labmap[a][1], vd)
            else:
                tok, ld = None, None
            if tok is None:
                toks.append(f"NUM_{a}_MISSING"); toks.append(f"LABAGE_{a}_MISSING")
            else:
                toks.append(tok)
                toks.append(f"LABAGE_{a}_{labage_bin(int((vd-ld).astype(int)))}")
        if k < len(vdates)-1:
            toks.append(f"GAP_{gap_bin(int((vdates[k+1]-vd).astype(int)))}")
    # static context
    toks.append("[CTX]")
    dobv = dob_map.get(pid)
    if dobv is not None and pd.notna(dobv):
        age = (pd.Timestamp(index_date) - dobv).days/365.25
    else:
        age = float("nan")
    toks.append(age_bin(age))
    sx = sex_map.get(pid)
    toks.append(f"DEM_SEX_{sx}" if sx in ("FEMALE","MALE") else "DEM_SEX_MISSING")
    # vitals have NO timestamp -> cannot be time-attributed before target -> MISSING (no imputation)
    toks += ["VITAL_SBP_MISSING","VITAL_DBP_MISSING","VITAL_BMI_MISSING",
             "VITAL_WEIGHT_MISSING","VITAL_HEIGHT_MISSING"]
    toks.append("[SEP]")
    return toks

# assemble for all pairs
seqs = []
oov = set()
for row in pairs_df.itertuples(index=False):
    toks = build_sequence(row.patient_id, row.index_date)
    for t in toks:
        if t not in VOCAB_SET: oov.add(t)
    seqs.append(toks)
pairs_df["n_tokens"] = [len(s) for s in seqs]
log(f"  Assembled {len(seqs):,} sequences. Mean length={pairs_df['n_tokens'].mean():.1f}, "
    f"median={pairs_df['n_tokens'].median():.0f}, max={pairs_df['n_tokens'].max()}")
log(f"  Out-of-vocabulary tokens emitted: {len(oov)}  {sorted(oov) if oov else '(none -- all tokens are in vocab.json)'}")

# ============================================================
# STEP 5: SPLITS
# ============================================================
log("\n" + "="*90); log("STEP 5  SPLITS"); log("="*90)

pairs_df = pairs_df.sort_values(["patient_id","index_date"]).reset_index(drop=True)
pairs_df["year"] = pairs_df["index_date"].dt.year

# ---- PRIMARY: temporal + patient-disjoint ----
# assign each patient by the YEAR of their earliest qualifying index draw
earliest = pairs_df.groupby("patient_id")["index_date"].min()
def part_of(year):
    if year < 2022: return "train"
    if year == 2022: return "validation"
    if year == 2023: return "test"
    return None  # 2024/2025 earliest -> no partition
patient_partition = {pid: part_of(dt.year) for pid, dt in earliest.items()}

def in_window(part, dt):
    if part == "train":      return dt < pd.Timestamp("2022-01-01")
    if part == "validation": return pd.Timestamp("2022-01-01") <= dt < pd.Timestamp("2023-01-01")
    if part == "test":       return pd.Timestamp("2023-01-01") <= dt < pd.Timestamp("2024-01-01")
    return False

prim = []
for row in pairs_df.itertuples(index=False):
    part = patient_partition.get(row.patient_id)
    if part is None:                      # patient assigned to no partition
        prim.append(None); continue
    prim.append(part if in_window(part, row.index_date) else None)  # drop out-of-window pairs
pairs_df["split_temporal"] = prim

temporal = pairs_df[pairs_df["split_temporal"].notna()].copy()
log("  PRIMARY temporal + patient-disjoint split:")
# integrity checks
sets = {p: set(temporal.loc[temporal.split_temporal==p, "patient_id"]) for p in ["train","validation","test"]}
overlap_tv = sets["train"] & sets["validation"]
overlap_tt = sets["train"] & sets["test"]
overlap_vt = sets["validation"] & sets["test"]
log(f"    patient overlap train/validation = {len(overlap_tv)}")
log(f"    patient overlap train/test       = {len(overlap_tt)}")
log(f"    patient overlap validation/test  = {len(overlap_vt)}")
n_dropped = int(pairs_df["split_temporal"].isna().sum())
log(f"    pairs dropped (out-of-window or patient-year 2024/2025): {n_dropped:,}")
for p in ["train","validation","test"]:
    sub = temporal[temporal.split_temporal==p]
    log(f"    {p:11s}: {len(sub):>8,} pairs, {sub['patient_id'].nunique():>7,} patients, "
        f"index dates {sub['index_date'].min().date()} .. {sub['index_date'].max().date()}")
    vc = sub['label'].value_counts()
    log(f"                 class balance: " + ", ".join(f"{k}={vc.get(k,0)} ({vc.get(k,0)/max(len(sub),1)*100:.1f}%)" for k in ["improved","stable","worsened"]))

# ---- SECONDARY: patient-grouped random 70/15/15 stratified by class ----
from sklearn.model_selection import train_test_split
# patient-level class = the patient's most frequent label (ties -> first seen)
pat_label = (pairs_df.groupby("patient_id")["label"]
             .agg(lambda s: s.value_counts().idxmax()))
pat_ids = np.asarray(pat_label.index.to_numpy(), dtype=np.int64)
pat_y = np.asarray(pat_label.to_numpy(), dtype=object)
tr_ids, tmp_ids, tr_y, tmp_y = train_test_split(pat_ids, pat_y, test_size=0.30,
                                                random_state=42, stratify=pat_y)
va_ids, te_ids, _, _ = train_test_split(tmp_ids, tmp_y, test_size=0.50,
                                        random_state=42, stratify=tmp_y)
rand_part = {}
for pid in tr_ids: rand_part[pid] = "train"
for pid in va_ids: rand_part[pid] = "validation"
for pid in te_ids: rand_part[pid] = "test"
pairs_df["split_random"] = pairs_df["patient_id"].map(rand_part)
log("\n  SECONDARY patient-grouped random 70/15/15 (stratified by patient class):")
rsets = {p: set(pairs_df.loc[pairs_df.split_random==p,"patient_id"]) for p in ["train","validation","test"]}
log(f"    patient overlap (should all be 0): tr/va={len(rsets['train']&rsets['validation'])}, "
    f"tr/te={len(rsets['train']&rsets['test'])}, va/te={len(rsets['validation']&rsets['test'])}")
for p in ["train","validation","test"]:
    sub = pairs_df[pairs_df.split_random==p]
    vc = sub['label'].value_counts()
    log(f"    {p:11s}: {len(sub):>8,} pairs, {sub['patient_id'].nunique():>7,} patients | "
        + ", ".join(f"{k}={vc.get(k,0)/max(len(sub),1)*100:.1f}%" for k in ["improved","stable","worsened"]))

# ============================================================
# STEP 6: save everything
# ============================================================
log("\n" + "="*90); log("STEP 6  Save artifacts"); log("="*90)

def iso(d): return pd.Timestamp(d).strftime("%Y-%m-%d")

# labelled pairs
out_pairs = pairs_df.copy()
out_pairs["index_date"] = out_pairs["index_date"].map(iso)
out_pairs["target_date"] = out_pairs["target_date"].map(iso)
out_pairs.to_csv(os.path.join(RESULTS, "cohort_pairs.csv"), index=False)

# sequences.jsonl (primary split label attached; also carries numeric targets)
LABEL2ID = {"improved":0, "stable":1, "worsened":2}
with open(os.path.join(RESULTS, "sequences.jsonl"), "w", encoding="utf-8") as fh:
    for row, toks in zip(pairs_df.itertuples(index=False), seqs):
        obj = {"patient_id": str(row.patient_id),
               "index_date": iso(row.index_date),
               "split": row.split_temporal if pd.notna(row.split_temporal) else None,
               "label": row.label, "label_id": LABEL2ID[row.label],
               "baseline_hba1c": round(row.baseline_hba1c,3),
               "target_hba1c": round(row.target_hba1c,3),
               "change": round(row.change,3), "days_between": int(row.days_between),
               "tokens": toks}
        fh.write(json.dumps(obj) + "\n")

# split files in the {patient_id, index_date, split} format of the reference splits.jsonl
def write_split(path, colname, subset_notna=True):
    d = pairs_df
    if subset_notna:
        d = d[d[colname].notna()]
    with open(path, "w", encoding="utf-8") as fh:
        for row in d.itertuples(index=False):
            fh.write(json.dumps({"patient_id": str(row.patient_id),
                                 "index_date": iso(row.index_date),
                                 "split": getattr(row, colname)}) + "\n")
    return len(d)
n_temporal = write_split(os.path.join(RESULTS, "splits_temporal.jsonl"), "split_temporal")
n_random   = write_split(os.path.join(RESULTS, "splits_random_grouped.jsonl"), "split_random", subset_notna=False)
log(f"  splits_temporal.jsonl rows: {n_temporal:,}")
log(f"  splits_random_grouped.jsonl rows: {n_random:,}")

with open(os.path.join(RESULTS, "cleaning_report.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(rep))

# ---- cohort_summary.md ----
def class_line(sub):
    vc = sub['label'].value_counts()
    tot = max(len(sub),1)
    return ", ".join(f"{k} {vc.get(k,0):,} ({vc.get(k,0)/tot*100:.1f}%)" for k in ["improved","stable","worsened"])

summ = []
summ.append("# Cohort / Labelling / Split Summary (pre-tokenise, pre-train)\n")
summ.append(f"- **Total labelled pairs:** {len(pairs_df):,} from {pairs_df['patient_id'].nunique():,} unique patients")
summ.append(f"- **Label rule:** change = target_hba1c - baseline_hba1c; improved <= -0.5, worsened >= +0.5, else stable")
summ.append(f"- **Index->target window:** next HbA1c {MIN_GAP}-{MAX_GAP} days after each index draw")
summ.append(f"- **Mean sequence length:** {pairs_df['n_tokens'].mean():.1f} tokens (median {pairs_df['n_tokens'].median():.0f}, max {pairs_df['n_tokens'].max()}), MAX_VISITS={MAX_VISITS}")
summ.append(f"- **Out-of-vocab tokens:** {len(oov)} (target: 0 -- vocabulary reused from vocab.json)\n")
summ.append("## Class balance (all pairs)")
summ.append("| class | count | pct |")
summ.append("|---|---:|---:|")
for k,v in pairs_df['label'].value_counts().items():
    summ.append(f"| {k} | {v:,} | {v/len(pairs_df)*100:.1f}% |")
summ.append("")
summ.append("## PRIMARY split — temporal + patient-disjoint")
summ.append("Assign each patient wholly by the year of their EARLIEST qualifying index draw, then keep only")
summ.append("that patient's pairs inside the partition window (train <2022, validation =2022, test =2023).")
summ.append("Pairs outside the assigned window are dropped so no patient can appear in two partitions.")
summ.append("")
summ.append("| partition | pairs | patients | index-date range | class balance |")
summ.append("|---|---:|---:|---|---|")
for p in ["train","validation","test"]:
    sub = temporal[temporal.split_temporal==p]
    summ.append(f"| {p} | {len(sub):,} | {sub['patient_id'].nunique():,} | {sub['index_date'].min().date()} .. {sub['index_date'].max().date()} | {class_line(sub)} |")
summ.append(f"\n- Patient overlap across partitions: train/val {len(overlap_tv)}, train/test {len(overlap_tt)}, val/test {len(overlap_vt)} (all must be 0)")
summ.append(f"- Pairs dropped (out-of-window, or earliest index in 2024/2025): {n_dropped:,}\n")
summ.append("## SECONDARY split — patient-grouped random 70/15/15, stratified by class (sensitivity analysis)")
summ.append("| partition | pairs | patients | class balance |")
summ.append("|---|---:|---:|---|")
for p in ["train","validation","test"]:
    sub = pairs_df[pairs_df.split_random==p]
    summ.append(f"| {p} | {len(sub):,} | {sub['patient_id'].nunique():,} | {class_line(sub)} |")
summ.append("")
summ.append("## Cleaning summary (rows removed)")
summ.append("| analyte | raw | dropped missing val/ts | out-of-range | exact dupes | kept |")
summ.append("|---|---:|---:|---:|---:|---:|")
for r in clean_rows:
    oor = [v for k,v in r.items() if k.startswith("out_of_range")][0]
    summ.append(f"| {r['analyte']} | {r['raw_rows']:,} | {r['missing_value']+r['missing_ts']:,} | {oor:,} | {r['exact_dupes']:,} | {r['kept']:,} |")
summ.append("")
summ.append("## Files written to Create_results/")
for fn in ["cleaned_hba1c.parquet","cohort_pairs.csv","sequences.jsonl","splits_temporal.jsonl",
           "splits_random_grouped.jsonl","cleaning_report.txt","cleaning_counts_by_analyte.csv","cohort_summary.md"]:
    summ.append(f"- `{fn}`")
open(os.path.join(RESULTS, "cohort_summary.md"), "w", encoding="utf-8").write("\n".join(summ))

log("\nDONE.  Review Create_results/cohort_summary.md")

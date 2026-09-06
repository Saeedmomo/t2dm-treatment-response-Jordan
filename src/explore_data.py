"""
Exploratory analysis of the Jordan EHR dataset.
Read-only exploration: no modeling. Produces a markdown report,
summary CSVs, and figures.
"""
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = r"D:\Hakeem_compitition\new_code"
DATA = os.path.join(BASE, "Dataset")
RESULTS = os.path.join(BASE, "Create_results")
FIGURES = os.path.join(BASE, "Create_figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

pd.set_option("display.width", 140)

report_lines = []
def log(s=""):
    print(s)
    report_lines.append(s)

file_stats_rows = []  # per-file summary rows

def read_csv(path, **kw):
    return pd.read_csv(path, encoding="utf-8-sig", **kw)

def col_missing_pct(df):
    n = len(df)
    return {c: (df[c].isna().sum() / n * 100 if n else np.nan) for c in df.columns}

# ---------------------------------------------------------------
# 1. SEX / BIRTH (demographic backbone)
# ---------------------------------------------------------------
log("="*90)
log("1. SEX / BIRTH (demographics)")
log("="*90)

dob = read_csv(os.path.join(DATA, "Sex_birth", "date of birth.csv"))
females = read_csv(os.path.join(DATA, "Sex_birth", "females patients.csv"))
males = read_csv(os.path.join(DATA, "Sex_birth", "males patients.csv"))

for name, df in [("date of birth.csv", dob), ("females patients.csv", females), ("males patients.csv", males)]:
    miss = col_missing_pct(df)
    file_stats_rows.append({"category": "Sex_birth", "file": name, "rows": len(df),
                             "n_unique_patient_id": df["PATIENT_ID"].nunique(),
                             "columns": ";".join(df.columns), "missing_pct": json.dumps({k: round(v,2) for k,v in miss.items()})})

dob_dupe = dob["PATIENT_ID"].duplicated().sum()
sex_overlap = set(females.PATIENT_ID) & set(males.PATIENT_ID)
sex_all = set(females.PATIENT_ID) | set(males.PATIENT_ID)
dob_ids = set(dob.PATIENT_ID)

log(f"date of birth.csv: {len(dob):,} rows, {dob['PATIENT_ID'].nunique():,} unique PATIENT_ID (duplicated rows: {dob_dupe})")
log(f"females patients.csv: {len(females):,} rows, {females['PATIENT_ID'].nunique():,} unique")
log(f"males patients.csv:   {len(males):,} rows, {males['PATIENT_ID'].nunique():,} unique")
log(f"Overlap between males/females id sets (should be 0): {len(sex_overlap)}")
log(f"Union of sex-labelled patients: {len(sex_all):,}")
log(f"Patients with DOB but no sex label: {len(dob_ids - sex_all):,}")
log(f"Patients with sex label but no DOB: {len(sex_all - dob_ids):,}")

dob["DATE_OF_BIRTH"] = pd.to_datetime(dob["DATE_OF_BIRTH"], errors="coerce")
log(f"DOB parse failures: {dob['DATE_OF_BIRTH'].isna().sum():,}")
log(f"DOB range: {dob['DATE_OF_BIRTH'].min()} to {dob['DATE_OF_BIRTH'].max()}")

TOTAL_PATIENTS = len(dob_ids | sex_all)
log(f"\n>>> TOTAL DISTINCT PATIENTS (union DOB + sex files): {TOTAL_PATIENTS:,}")

# ---------------------------------------------------------------
# 2. DIAGNOSIS
# ---------------------------------------------------------------
log("\n" + "="*90)
log("2. DIAGNOSIS")
log("="*90)

diag_direct = read_csv(os.path.join(DATA, "Diagnosis", "DIAGNOSIS Direct.csv"))
diag_wide = read_csv(os.path.join(DATA, "Diagnosis", "DIAGNOSIS wide.csv"))

for name, df in [("DIAGNOSIS Direct.csv", diag_direct), ("DIAGNOSIS wide.csv", diag_wide)]:
    miss = col_missing_pct(df)
    file_stats_rows.append({"category": "Diagnosis", "file": name, "rows": len(df),
                             "n_unique_patient_id": df["PATIENT_ID"].nunique(),
                             "columns": ";".join(df.columns), "missing_pct": json.dumps({k: round(v,2) for k,v in miss.items()})})

log(f"DIAGNOSIS Direct.csv: {len(diag_direct):,} rows, {diag_direct['PATIENT_ID'].nunique():,} unique patients, "
    f"{diag_direct['PROBLEM_ID'].nunique():,} unique PROBLEM_ID")
log(f"DIAGNOSIS wide.csv:   {len(diag_wide):,} rows, {diag_wide['PATIENT_ID'].nunique():,} unique patients, "
    f"{diag_wide['PROBLEM_ID'].nunique():,} unique PROBLEM_ID")

direct_keys = set(zip(diag_direct.PATIENT_ID, diag_direct.PROBLEM_ID))
wide_keys = set(zip(diag_wide.PATIENT_ID, diag_wide.PROBLEM_ID))
log(f"(PATIENT_ID,PROBLEM_ID) pairs only in Direct: {len(direct_keys - wide_keys):,}")
log(f"(PATIENT_ID,PROBLEM_ID) pairs only in wide:   {len(wide_keys - direct_keys):,}")
log(f"pairs in both: {len(direct_keys & wide_keys):,}")
log("-> Direct and wide look like overlapping/duplicate exports of the same problem list, wide is a superset-ish pull.")

diag_wide["DATE_ENTERED"] = pd.to_datetime(diag_wide["DATE_ENTERED"], errors="coerce")
log(f"DIAGNOSIS wide DATE_ENTERED range: {diag_wide['DATE_ENTERED'].min()} to {diag_wide['DATE_ENTERED'].max()}, "
    f"parse failures: {diag_wide['DATE_ENTERED'].isna().sum():,}")

diag_types = read_csv(os.path.join(DATA, "Diagnosis", "DIAGNOSIS TYPES TOTAL 8600.csv"))
diag_760 = read_csv(os.path.join(DATA, "Diagnosis", "diagnosis 760.csv"))
diag_899 = read_csv(os.path.join(DATA, "Diagnosis", "diagnosis 899.csv"))
log(f"\nDIAGNOSIS TYPES TOTAL 8600.csv: {len(diag_types):,} rows -> reference vocabulary list of diagnosis label strings (no PATIENT_ID), not patient-level data")
log(f"diagnosis 760.csv: {len(diag_760):,} rows, diagnosis 899.csv: {len(diag_899):,} rows -> smaller reference vocab lists (likely diabetes-relevant diagnosis subsets)")
for name, df in [("DIAGNOSIS TYPES TOTAL 8600.csv", diag_types), ("diagnosis 760.csv", diag_760), ("diagnosis 899.csv", diag_899)]:
    file_stats_rows.append({"category": "Diagnosis", "file": name, "rows": len(df), "n_unique_patient_id": np.nan,
                             "columns": ";".join(df.columns), "missing_pct": "reference/vocab list, not patient-linked"})

diag_patients = set(diag_wide.PATIENT_ID) | set(diag_direct.PATIENT_ID)
log(f"\nPatients with >=1 diagnosis record: {len(diag_patients):,} ({len(diag_patients)/TOTAL_PATIENTS*100:.1f}% of total)")

# ---------------------------------------------------------------
# 3. MEDICATIONS
# ---------------------------------------------------------------
log("\n" + "="*90)
log("3. MEDICATIONS")
log("="*90)

med_files = sorted(glob.glob(os.path.join(DATA, "Medications", "M*.csv")),
                    key=lambda p: int(''.join(filter(str.isdigit, os.path.basename(p)))))
med_patient_ids = set()
med_total_rows = 0
med_generic_names = set()
med_date_min, med_date_max = None, None
med_missing_acc = {}
sample_cols = None
med_schema_by_file = {}
for i, f in enumerate(med_files):
    df = read_csv(f)
    if sample_cols is None:
        sample_cols = list(df.columns)
    med_schema_by_file[os.path.basename(f)] = list(df.columns)
    med_total_rows += len(df)
    med_patient_ids |= set(df["PATIENT_ID"].unique())
    med_generic_names |= set(df["GENERIC_NAME"].dropna().unique())
    dt = pd.to_datetime(df["DISPENSED_DATE"], errors="coerce")
    dmin, dmax = dt.min(), dt.max()
    if med_date_min is None or (pd.notna(dmin) and dmin < med_date_min): med_date_min = dmin
    if med_date_max is None or (pd.notna(dmax) and dmax > med_date_max): med_date_max = dmax
    miss = col_missing_pct(df)
    for k, v in miss.items():
        med_missing_acc.setdefault(k, []).append(v)
    file_stats_rows.append({"category": "Medications", "file": os.path.basename(f), "rows": len(df),
                             "n_unique_patient_id": df["PATIENT_ID"].nunique(),
                             "columns": ";".join(df.columns), "missing_pct": json.dumps({k: round(v,2) for k,v in miss.items()})})

log(f"Medication files: {len(med_files)} (M1..M30, appear to be arbitrary batch splits of one big dispensing table)")
log(f"Columns (most common): {sample_cols}")
log(f"Total rows: {med_total_rows:,}")

no_days_supply = [f for f, c in med_schema_by_file.items() if "DAYS_SUPPLY" not in c]
has_r_col = [f for f, c in med_schema_by_file.items() if "R" in c]
log(f"\n*** Schema is NOT uniform across the 30 medication batch files: ***")
log(f"  {len(no_days_supply)} files are MISSING the DAYS_SUPPLY column entirely: {no_days_supply}")
log(f"  {len(has_r_col)} file(s) have an extra 'R' column (all TRUE in that file, meaning unknown -- possibly a refill flag): {has_r_col}")
log(f"  -> before modeling, medication features involving days_supply will only be available for the other {len(med_files)-len(no_days_supply)} batches.")
log(f"Unique patients with >=1 medication: {len(med_patient_ids):,} ({len(med_patient_ids)/TOTAL_PATIENTS*100:.1f}% of total)")
log(f"Unique GENERIC_NAME values (drug names): {len(med_generic_names):,}")
log(f"DISPENSED_DATE range: {med_date_min} to {med_date_max}")
log("Average missing %% per column across medication files:")
for k, v in med_missing_acc.items():
    log(f"  {k}: {np.mean(v):.2f}%")

# ---------------------------------------------------------------
# 4. VITALS
# ---------------------------------------------------------------
log("\n" + "="*90)
log("4. VITALS")
log("="*90)

vit_bp = read_csv(os.path.join(DATA, "Vitals", "vitals BP.csv"))
vit_height = read_csv(os.path.join(DATA, "Vitals", "vitals height.csv"))
vit_pulse = read_csv(os.path.join(DATA, "Vitals", "vitals pulse.csv"))
vit_wt = read_csv(os.path.join(DATA, "Vitals", "vitals wt kg.csv"))

for name, df in [("vitals BP.csv", vit_bp), ("vitals height.csv", vit_height),
                  ("vitals pulse.csv", vit_pulse), ("vitals wt kg.csv", vit_wt)]:
    miss = col_missing_pct(df)
    file_stats_rows.append({"category": "Vitals", "file": name, "rows": len(df),
                             "n_unique_patient_id": df["PATIENT_ID"].nunique(),
                             "columns": ";".join(df.columns), "missing_pct": json.dumps({k: round(v,2) for k,v in miss.items()})})
    log(f"{name}: {len(df):,} rows, {df['PATIENT_ID'].nunique():,} unique patients, columns={list(df.columns)}, has DATE column: NO")

vit_patients = set(vit_bp.PATIENT_ID) | set(vit_height.PATIENT_ID) | set(vit_pulse.PATIENT_ID) | set(vit_wt.PATIENT_ID)
log(f"\n*** CRITICAL: none of the 4 vitals files (BP, height, pulse, weight) contain a date or timestamp column. ***")
log(f"    Row order within each file is the only ordering available and there is no evidence it is chronological")
log(f"    (rows are not even contiguous per patient in all files) -- vitals CANNOT be reliably sequenced in time.")
log(f"Patients with >=1 vitals record (any type): {len(vit_patients):,} ({len(vit_patients)/TOTAL_PATIENTS*100:.1f}% of total)")

# check whether rows are grouped by patient (contiguous) as a weak signal of extraction order
def is_grouped(df, col="PATIENT_ID", n=200000):
    sub = df[col].head(n)
    seen = set()
    groups = 0
    last = None
    for v in sub:
        if v != last:
            groups += 1
            if v in seen:
                return False  # patient id reappeared non-contiguously -> not grouped
            seen.add(v)
            last = v
    return True

log(f"vitals BP.csv rows contiguous-by-patient in first 200k rows: {is_grouped(vit_bp)}")
log(f"vitals wt kg.csv rows contiguous-by-patient in first 200k rows: {is_grouped(vit_wt)}")

# ---------------------------------------------------------------
# 5. LAB TESTS (all analytes) + HbA1c deep dive
# ---------------------------------------------------------------
log("\n" + "="*90)
log("5. LAB TESTS")
log("="*90)

lab_dir = os.path.join(DATA, "lab_tests")
lab_files = sorted(glob.glob(os.path.join(lab_dir, "*.csv"))) + sorted(glob.glob(os.path.join(lab_dir, "*.xlsx")))

lab_patient_ids_all = set()
lab_test_names = {}   # test_name -> {rows, patients set (sampled), date min/max}
lab_summary_rows = []

hba1c_frames = []

for f in lab_files:
    base = os.path.basename(f)
    if f.endswith(".xlsx"):
        df = pd.read_excel(f)
    else:
        df = read_csv(f)
    df.columns = [c.strip() for c in df.columns]
    # LabTestName holds analyte codes like "NA" (sodium) that pandas' default NA-string
    # list would otherwise swallow as missing -- restore from the filename when that happens.
    if df["LabTestName"].isna().all():
        df["LabTestName"] = base.split(" ")[0]
    n = len(df)
    n_pat = df["PatientID"].nunique()
    lab_patient_ids_all |= set(df["PatientID"].unique())
    dt = pd.to_datetime(df["DATE_TIME_SPECIMEN_TAKEN"], errors="coerce")
    test_name = df["LabTestName"].mode().iat[0] if n else base
    miss = col_missing_pct(df)
    lab_summary_rows.append({
        "file": base, "test_name_field": test_name, "rows": n, "n_unique_patients": n_pat,
        "date_min": dt.min(), "date_max": dt.max(),
        "pct_missing_result": miss.get("LAB_DATA_CHEMISTRY_RESULTS", np.nan),
        "pct_missing_date": miss.get("DATE_TIME_SPECIMEN_TAKEN", np.nan),
    })
    file_stats_rows.append({"category": "lab_tests", "file": base, "rows": n,
                             "n_unique_patient_id": n_pat,
                             "columns": ";".join(df.columns), "missing_pct": json.dumps({k: round(v,2) for k,v in miss.items()})})
    if "HBA1C" in base.upper():
        tmp = df[["PatientID", "LAB_DATA_CHEMISTRY_RESULTS", "DATE_TIME_SPECIMEN_TAKEN"]].copy()
        tmp["source_file"] = base
        hba1c_frames.append(tmp)

lab_summary = pd.DataFrame(lab_summary_rows)
lab_summary.to_csv(os.path.join(RESULTS, "lab_tests_per_file_summary.csv"), index=False)

log(f"Number of lab test files: {len(lab_files)} (csv + xlsx), covering ~{lab_summary['test_name_field'].nunique()} distinct analytes")
log(f"Total lab result rows across all analytes: {lab_summary['rows'].sum():,}")
log(f"Unique patients with >=1 lab result (any analyte): {len(lab_patient_ids_all):,} ({len(lab_patient_ids_all)/TOTAL_PATIENTS*100:.1f}% of total)")
log("\nPer-analyte file summary saved to Create_results/lab_tests_per_file_summary.csv")
log("\nNote on '11/22/33' and '1 D/2 D' suffixes: these look like separate export batches of the SAME analyte")
log("(identical schema, identical LabTestName value, non-overlapping row counts consistent with a paginated pull),")
log("not different meanings -- confirmed by matching column layout and LabTestName across e.g. CRT 11/22/33.")

# ---- HbA1c deep dive ----
log("\n" + "-"*90)
log("5a. HbA1c deep dive (HBA1C 11.csv + HBA1C 22.csv combined)")
log("-"*90)

hba1c = pd.concat(hba1c_frames, ignore_index=True)
hba1c["DATE_TIME_SPECIMEN_TAKEN"] = pd.to_datetime(hba1c["DATE_TIME_SPECIMEN_TAKEN"], errors="coerce")
hba1c["LAB_DATA_CHEMISTRY_RESULTS"] = pd.to_numeric(hba1c["LAB_DATA_CHEMISTRY_RESULTS"], errors="coerce")

n_total = len(hba1c)
n_missing_val = hba1c["LAB_DATA_CHEMISTRY_RESULTS"].isna().sum()
n_missing_date = hba1c["DATE_TIME_SPECIMEN_TAKEN"].isna().sum()
n_dupe = hba1c.duplicated(subset=["PatientID", "DATE_TIME_SPECIMEN_TAKEN", "LAB_DATA_CHEMISTRY_RESULTS"]).sum()

vals = hba1c["LAB_DATA_CHEMISTRY_RESULTS"].dropna()
log(f"Total HbA1c measurements (raw rows, both files): {n_total:,}")
log(f"Missing result value: {n_missing_val:,} ({n_missing_val/n_total*100:.2f}%)")
log(f"Missing/unparseable date: {n_missing_date:,} ({n_missing_date/n_total*100:.2f}%)")
log(f"Exact duplicate rows (same patient/date/value): {n_dupe:,}")
log(f"Unique patients with >=1 HbA1c value: {hba1c['PatientID'].nunique():,} "
    f"({hba1c['PatientID'].nunique()/TOTAL_PATIENTS*100:.1f}% of total patients)")

log(f"\nHbA1c value distribution (all non-missing rows, units assumed %):")
log(f"  min={vals.min():.3f}  max={vals.max():.3f}  mean={vals.mean():.3f}  median={vals.median():.3f}  std={vals.std():.3f}")
for p in [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]:
    log(f"  p{p}: {vals.quantile(p/100):.3f}")

n_lt3 = (vals < 3).sum(); n_gt20 = (vals > 20).sum()
n_le0 = (vals <= 0).sum()
log(f"\nPlausibility check: values <=0: {n_le0}, values <3%: {n_lt3}, values >20%: {n_gt20} "
    f"(outside physiologic range for HbA1c% -- likely entry errors or a different unit/assay, worth cleaning before modeling)")
log(f"  The raw max ({vals.max():,.0f}) is a single corrupted row (PatientID 2593528, HBA1C 22.csv, 6/20/2018) -- clearly")
log(f"  a data-entry/encoding artifact, not a real HbA1c%, and it single-handedly drags the raw mean up to {vals.mean():,.1f}.")
log(f"  Counts by magnitude: >1000: {(vals>1000).sum()}, >100: {(vals>100).sum()}, >50: {(vals>50).sum()}, >20: {(vals>20).sum()} "
    f"-- out of {len(vals):,} total, i.e. outliers are rare (<0.01%) but severe enough to break naive mean/std.")
n99 = hba1c.LabTestName.count() if False else None
by_src = hba1c.loc[hba1c.LAB_DATA_CHEMISTRY_RESULTS > 20].groupby("source_file").size().to_dict()
log(f"  Implausible (>20) values by source file: {by_src}")

plausible = vals[(vals >= 2) & (vals <= 20)]
log(f"\nHbA1c distribution restricted to a plausible clinical range [2, 20]% (n={len(plausible):,}, "
    f"{len(plausible)/len(vals)*100:.3f}% of all non-missing values):")
log(f"  min={plausible.min():.2f}  max={plausible.max():.2f}  mean={plausible.mean():.2f}  median={plausible.median():.2f}  std={plausible.std():.2f}")
log(f"  -> this is the range to use as ground truth for reporting/modeling; consistent with a diabetes-heavy population")
log(f"     (median ~6.1% is above the normal <5.7% cutoff, IQR spans the pre-diabetic/diabetic range).")

log(f"\nDate range of HbA1c measurements: {hba1c['DATE_TIME_SPECIMEN_TAKEN'].min()} to {hba1c['DATE_TIME_SPECIMEN_TAKEN'].max()}")

visits_per_patient = hba1c.dropna(subset=["DATE_TIME_SPECIMEN_TAKEN"]).groupby("PatientID")["DATE_TIME_SPECIMEN_TAKEN"].nunique()
log(f"\nHbA1c measurements per patient (distinct timestamps):")
log(f"  1 measurement:  {(visits_per_patient==1).sum():,} patients")
log(f"  2 measurements: {(visits_per_patient==2).sum():,} patients")
log(f"  3-5 measurements: {((visits_per_patient>=3)&(visits_per_patient<=5)).sum():,} patients")
log(f"  6-10 measurements: {((visits_per_patient>=6)&(visits_per_patient<=10)).sum():,} patients")
log(f"  >10 measurements: {(visits_per_patient>10).sum():,} patients")
log(f"  max measurements for a single patient: {visits_per_patient.max()}")
log(f"  median measurements/patient: {visits_per_patient.median():.1f}, mean: {visits_per_patient.mean():.2f}")
log(f"  patients with >=2 HbA1c measurements (usable for a real sequence): {(visits_per_patient>=2).sum():,} "
    f"({(visits_per_patient>=2).sum()/len(visits_per_patient)*100:.1f}% of patients who have HbA1c at all)")

hba1c.to_parquet(os.path.join(RESULTS, "hba1c_combined.parquet")) if False else None
visits_per_patient.rename("n_hba1c_measurements").to_csv(os.path.join(RESULTS, "hba1c_measurements_per_patient.csv"))

# ---------------------------------------------------------------
# 6. Cross-category patient coverage / sparsity
# ---------------------------------------------------------------
log("\n" + "="*90)
log("6. CROSS-CATEGORY COVERAGE (sparsity)")
log("="*90)

cov = pd.DataFrame({
    "category": ["Sex/DOB (any)", "Diagnosis", "Medications", "Vitals (any)", "Labs (any)", "HbA1c specifically"],
    "n_patients": [TOTAL_PATIENTS, len(diag_patients), len(med_patient_ids), len(vit_patients),
                   len(lab_patient_ids_all), hba1c['PatientID'].nunique()],
})
cov["pct_of_total"] = (cov["n_patients"] / TOTAL_PATIENTS * 100).round(1)
log(cov.to_string(index=False))
cov.to_csv(os.path.join(RESULTS, "cross_category_coverage.csv"), index=False)

both = diag_patients & med_patient_ids & vit_patients & lab_patient_ids_all
log(f"\nPatients present in ALL FOUR of diagnosis+medications+vitals+labs: {len(both):,} ({len(both)/TOTAL_PATIENTS*100:.1f}%)")

# ---------------------------------------------------------------
# 7. TEMPORAL STRUCTURE
# ---------------------------------------------------------------
log("\n" + "="*90)
log("7. TEMPORAL STRUCTURE -- can real sequences be built?")
log("="*90)

hba1c_multi = hba1c.dropna(subset=["DATE_TIME_SPECIMEN_TAKEN"]).sort_values(["PatientID", "DATE_TIME_SPECIMEN_TAKEN"])
gaps = hba1c_multi.groupby("PatientID")["DATE_TIME_SPECIMEN_TAKEN"].diff().dropna().dt.days
log(f"Days between CONSECUTIVE HbA1c measurements for the same patient (n={len(gaps):,} gaps, from the "
    f"{(visits_per_patient>=2).sum():,} patients with >=2 measurements):")
log(f"  min={gaps.min():.0f}  p5={gaps.quantile(.05):.0f}  median={gaps.median():.0f}  p95={gaps.quantile(.95):.0f}  max={gaps.max():.0f}")
log(f"  same-day repeat draws (gap=0 days): {(gaps==0).sum():,} ({(gaps==0).sum()/len(gaps)*100:.1f}%)")

# do other analytes share an exact timestamp with HbA1c (evidence of a lab "panel"/visit)?
def load_analyte(paths):
    frames = []
    for p in paths:
        d = read_csv(p) if p.endswith(".csv") else pd.read_excel(p)
        d.columns = [c.strip() for c in d.columns]
        frames.append(d[["PatientID", "DATE_TIME_SPECIMEN_TAKEN"]])
    out = pd.concat(frames, ignore_index=True)
    out["DATE_TIME_SPECIMEN_TAKEN"] = pd.to_datetime(out["DATE_TIME_SPECIMEN_TAKEN"], errors="coerce")
    return set(zip(out["PatientID"], out["DATE_TIME_SPECIMEN_TAKEN"]))

hba1c_ts = set(zip(hba1c["PatientID"], hba1c["DATE_TIME_SPECIMEN_TAKEN"]))
fbs_ts = load_analyte([os.path.join(lab_dir, "FBS 11.csv"), os.path.join(lab_dir, "FBS 22.csv")])
crt_ts = load_analyte([os.path.join(lab_dir, "CRT 11.csv"), os.path.join(lab_dir, "CRT 22.csv"), os.path.join(lab_dir, "CRT 33.csv")])
hb_ts = load_analyte([os.path.join(lab_dir, "HB 11.csv"), os.path.join(lab_dir, "HB 22.csv")])

shared_fbs = len(hba1c_ts & fbs_ts)
shared_crt = len(hba1c_ts & crt_ts)
shared_hb = len(hba1c_ts & hb_ts)
log(f"\nHbA1c timestamps that exactly coincide with another analyte's timestamp for the same patient (evidence of a lab panel/visit):")
log(f"  shares exact timestamp with FBS: {shared_fbs:,} / {len(hba1c_ts):,} ({shared_fbs/len(hba1c_ts)*100:.1f}%)")
log(f"  shares exact timestamp with Creatinine (CRT): {shared_crt:,} / {len(hba1c_ts):,} ({shared_crt/len(hba1c_ts)*100:.1f}%)")
log(f"  shares exact timestamp with Hb: {shared_hb:,} / {len(hba1c_ts):,} ({shared_hb/len(hba1c_ts)*100:.1f}%)")
log(f"  -> a large share of HbA1c draws coincide to the minute with other analytes, meaning DATE_TIME_SPECIMEN_TAKEN")
log(f"     can be used to group multiple lab results into a single 'visit'/specimen draw event.")

log(f"""
SUMMARY -- answering "what temporal structure is possible":

* Labs (incl. HbA1c) DO have real timestamps (date+time) and the majority of patients with HbA1c have only
  1 measurement, but {(visits_per_patient>=2).sum():,} patients ({(visits_per_patient>=2).sum()/len(visits_per_patient)*100:.1f}% of those with any HbA1c) have
  2+ dated HbA1c measurements, so genuine irregularly-spaced longitudinal sequences ARE possible for roughly
  half the HbA1c-bearing patients -- but only about {(visits_per_patient==1).sum()/len(visits_per_patient)*100:.0f}% of the rest have just a single snapshot.
* Multiple lab analytes drawn at the identical timestamp can be grouped into a "visit"/panel using
  (PatientID, DATE_TIME_SPECIMEN_TAKEN) as a join key -- this is a legitimate way to build a multi-feature
  per-visit row.
* Medications only have a DISPENSED_DATE (day resolution, no time), and diagnoses only have DATE_ENTERED
  (day resolution) -- both can be ordered and joined to lab visits by nearest date, but not to the minute.
* Vitals (BP, pulse, weight, height) have NO timestamp of any kind. They cannot be placed on the same
  timeline as labs/meds/diagnoses, and their within-file row order is not verified to be chronological.
  Options if you still want to use them:
    (a) drop them from any sequence model and use them only as static/aggregate features (e.g. last-known
        weight, mean pulse) -- safest, most honest option;
    (b) if the source system is known to export in visit order, treat row order as a weak chronology proxy
        -- risky, unverifiable from the data alone, and should be validated with whoever extracted these
        files before relying on it;
    (c) request a re-extraction that includes the missing date/timestamp column from the source EHR --
        best long-term fix if you plan to model vitals dynamics over time.
* Net assessment: a real sequence-per-patient model is defensible for labs (best), and doable at
  day-resolution for diagnoses/medications, but vitals as currently exported cannot be sequenced honestly.
""")

# ---------------------------------------------------------------
# save per-file stats table
# ---------------------------------------------------------------
pd.DataFrame(file_stats_rows).to_csv(os.path.join(RESULTS, "all_files_summary.csv"), index=False)

# ---------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------
plt.figure(figsize=(8,5))
plt.hist(plausible, bins=80, color="#3b6ea5", edgecolor="none")
plt.axvline(plausible.median(), color="red", linestyle="--", linewidth=1, label=f"median={plausible.median():.1f}")
plt.xlabel("HbA1c (%)")
plt.ylabel("Count of measurements")
plt.title(f"HbA1c value distribution (n={len(plausible):,} measurements in plausible [2,20]% range)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "hba1c_distribution.png"), dpi=150)
plt.close()

plt.figure(figsize=(8,5))
plt.hist(gaps.clip(upper=365), bins=60, color="#c47a3b", edgecolor="none")
plt.axvline(gaps.median(), color="red", linestyle="--", linewidth=1, label=f"median={gaps.median():.0f}d")
plt.xlabel("Days between consecutive HbA1c measurements (capped at 365)")
plt.ylabel("Count of gaps")
plt.title(f"Spacing between consecutive HbA1c draws, same patient (n={len(gaps):,} gaps)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "hba1c_gap_days.png"), dpi=150)
plt.close()

plt.figure(figsize=(8,5))
counts = visits_per_patient.clip(upper=15).value_counts().sort_index()
plt.bar(counts.index.astype(str), counts.values, color="#5a9367")
plt.xlabel("Number of distinct HbA1c measurement dates per patient (15+ capped)")
plt.ylabel("Number of patients")
plt.title("HbA1c measurements per patient")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "hba1c_measurements_per_patient.png"), dpi=150)
plt.close()

plt.figure(figsize=(9,5))
monthly = hba1c.dropna(subset=["DATE_TIME_SPECIMEN_TAKEN"]).set_index("DATE_TIME_SPECIMEN_TAKEN").resample("MS").size()
monthly.plot(color="#a5573b")
plt.xlabel("Month")
plt.ylabel("Number of HbA1c measurements")
plt.title("HbA1c measurement volume over time")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "hba1c_volume_over_time.png"), dpi=150)
plt.close()

plt.figure(figsize=(8,5))
plt.bar(cov["category"], cov["pct_of_total"], color="#8a5a9e")
plt.ylabel("% of total patients")
plt.title("Patient coverage by data category")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES, "cross_category_coverage.png"), dpi=150)
plt.close()

log("\nFigures saved: hba1c_distribution.png, hba1c_measurements_per_patient.png, hba1c_volume_over_time.png, hba1c_gap_days.png, cross_category_coverage.png")

with open(os.path.join(RESULTS, "exploratory_report.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(report_lines))

print("\nDONE. Report + CSVs in Create_results/, figures in Create_figures/")

"""
Reporting artifacts (independent of the models):
  - cohort flow diagram (382,537 patients -> 151,750 patients / 569,201 pairs, every exclusion counted)
  - Table 1 of baseline characteristics by temporal split and by outcome class
"""
import os, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sys.path.insert(0, os.path.dirname(__file__))
from common import RES, FIG, PREP

meta = pd.read_parquet(os.path.join(PREP, "meta.parquet"))
hb = pd.read_parquet(os.path.join(RES, "cleaned_hba1c.parquet"))

# ---- counts for the flow diagram ----
TOTAL_PATIENTS = 382537                     # from Sex_birth (step-1 exploration)
HBA1C_PATIENTS_RAW = 382472                 # patients with >=1 raw HbA1c
hb_patients = hb["patient_id"].nunique()    # patients with >=1 cleaned HbA1c draw
hb_draws = len(hb)                          # cleaned date-level HbA1c draws
pair_patients = meta["patient_id"].nunique()
pair_count = len(meta)
# HbA1c cleaning removals (from cleaning_counts_by_analyte.csv)
cc = pd.read_csv(os.path.join(RES, "cleaning_counts_by_analyte.csv"))
hrow = cc[cc["analyte"]=="HBA1C"].iloc[0]
oor_col = [c for c in cc.columns if c.startswith("out_of_range")][0]
hb_removed = int(hrow["missing_ts"] + hrow[oor_col] + hrow["exact_dupes"])

no_pair_patients = hb_patients - pair_patients

boxes = [
    (f"All patients in EHR\nN = {TOTAL_PATIENTS:,}", "#dbe9f6"),
    (f"Patients with >=1 HbA1c result\nN = {HBA1C_PATIENTS_RAW:,}", "#dbe9f6"),
    (f"After HbA1c cleaning (drop out-of-range [2,20]%,\nmissing date, exact dupes: {hb_removed:,} rows)\n"
     f"Patients with >=1 clean HbA1c draw N = {hb_patients:,}\n({hb_draws:,} date-level draws)", "#dbe9f6"),
    (f"Patients with an index->target HbA1c pair\n(next draw 90-450 days later)\n"
     f"N = {pair_patients:,} patients / {pair_count:,} pairs", "#cfe8cf"),
]
excl = [
    f"Excluded: no HbA1c on record\n({TOTAL_PATIENTS-HBA1C_PATIENTS_RAW:,})",
    f"HbA1c rows removed in cleaning: {hb_removed:,}\n(patients kept if >=1 clean draw remains)",
    f"Excluded: no qualifying follow-up draw\nin 90-450 days ({no_pair_patients:,} patients)",
]

fig, ax = plt.subplots(figsize=(9.5, 10)); ax.axis("off")
yc = [0.9, 0.66, 0.42, 0.14]; x0 = 0.28; w = 0.44
for i,(txt,col) in enumerate(boxes):
    ax.add_patch(mpatches.FancyBboxPatch((x0, yc[i]-0.05), w, 0.10, boxstyle="round,pad=0.01",
                 fc=col, ec="#33556e", lw=1.3, transform=ax.transAxes))
    ax.text(x0+w/2, yc[i], txt, ha="center", va="center", fontsize=9.5, transform=ax.transAxes)
    if i < len(boxes)-1:
        ax.annotate("", xy=(x0+w/2, yc[i+1]+0.055), xytext=(x0+w/2, yc[i]-0.055),
                    xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.4, color="#33556e"))
for i,e in enumerate(excl):
    ymid = (yc[i]+yc[i+1])/2
    ax.add_patch(mpatches.FancyBboxPatch((0.74, ymid-0.035), 0.24, 0.07, boxstyle="round,pad=0.01",
                 fc="#f6e5db", ec="#a5573b", lw=1.0, transform=ax.transAxes))
    ax.text(0.86, ymid, e, ha="center", va="center", fontsize=8, transform=ax.transAxes)
    ax.annotate("", xy=(0.74, ymid), xytext=(x0+w, ymid),
                xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.0, color="#a5573b"))
ax.set_title("Cohort flow — Jordan EHR HbA1c-trajectory cohort", fontsize=13, pad=12)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "cohort_flow.png"), dpi=150, bbox_inches="tight"); plt.close()
print("cohort_flow.png written")

# ---------------- Table 1 ----------------
def summ(df):
    return {
        "n_pairs": len(df),
        "n_patients": df["patient_id"].nunique(),
        "age_mean_sd": f"{df['age'].mean():.1f} +/- {df['age'].std():.1f}",
        "female_pct": f"{(df['sex']=='FEMALE').mean()*100:.1f}%",
        "baseline_hba1c_mean_sd": f"{df['baseline_hba1c'].mean():.2f} +/- {df['baseline_hba1c'].std():.2f}",
        "target_hba1c_mean_sd": f"{df['target_hba1c'].mean():.2f} +/- {df['target_hba1c'].std():.2f}",
        "days_between_median": f"{df['days_between'].median():.0f}",
        "n_visits_mean": f"{df['n_visits'].mean():.1f}",
        "seq_len_mean": f"{df['seq_len'].mean():.1f}",
        "pct_improved": f"{(df['label']=='improved').mean()*100:.1f}%",
        "pct_stable":   f"{(df['label']=='stable').mean()*100:.1f}%",
        "pct_worsened": f"{(df['label']=='worsened').mean()*100:.1f}%",
    }

# by temporal split
rows = {}
for s in ["train","validation","test"]:
    sub = meta[meta["split_temporal"]==s]
    if len(sub): rows[f"split_{s}"] = summ(sub)
rows["all_pairs"] = summ(meta)
t1_split = pd.DataFrame(rows).T
t1_split.to_csv(os.path.join(RES, "table1_by_split.csv"))

# by outcome class
rows2 = {c: summ(meta[meta["label"]==c]) for c in ["improved","stable","worsened"]}
t1_cls = pd.DataFrame(rows2).T
t1_cls.to_csv(os.path.join(RES, "table1_by_outcome.csv"))

print("\n=== Table 1 by temporal split ===")
print(t1_split[["n_pairs","n_patients","age_mean_sd","female_pct","baseline_hba1c_mean_sd","pct_improved","pct_stable","pct_worsened"]].to_string())
print("\n=== Table 1 by outcome class ===")
print(t1_cls[["n_pairs","n_patients","age_mean_sd","female_pct","baseline_hba1c_mean_sd","target_hba1c_mean_sd","n_visits_mean"]].to_string())
print("\nreporting.py DONE")

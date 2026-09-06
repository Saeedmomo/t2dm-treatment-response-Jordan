# Cohort / Labelling / Split Summary (pre-tokenise, pre-train)

- **Total labelled pairs:** 569,201 from 151,750 unique patients
- **Label rule:** change = target_hba1c - baseline_hba1c; improved <= -0.5, worsened >= +0.5, else stable
- **Index->target window:** next HbA1c 90-450 days after each index draw
- **Mean sequence length:** 137.5 tokens (median 104, max 511), MAX_VISITS=20
- **Out-of-vocab tokens:** 0 (target: 0 -- vocabulary reused from vocab.json)

## Class balance (all pairs)
| class | count | pct |
|---|---:|---:|
| stable | 276,094 | 48.5% |
| improved | 150,947 | 26.5% |
| worsened | 142,160 | 25.0% |

## PRIMARY split — temporal + patient-disjoint
Assign each patient wholly by the year of their EARLIEST qualifying index draw, then keep only
that patient's pairs inside the partition window (train <2022, validation =2022, test =2023).
Pairs outside the assigned window are dropped so no patient can appear in two partitions.

| partition | pairs | patients | index-date range | class balance |
|---|---:|---:|---|---|
| train | 298,918 | 94,933 | 2011-02-02 .. 2021-12-31 | improved 84,268 (28.2%), stable 134,977 (45.2%), worsened 79,673 (26.7%) |
| validation | 28,470 | 20,995 | 2022-01-02 .. 2022-12-31 | improved 7,085 (24.9%), stable 16,232 (57.0%), worsened 5,153 (18.1%) |
| test | 30,894 | 22,908 | 2023-01-02 .. 2023-12-31 | improved 7,572 (24.5%), stable 17,307 (56.0%), worsened 6,015 (19.5%) |

- Patient overlap across partitions: train/val 0, train/test 0, val/test 0 (all must be 0)
- Pairs dropped (out-of-window, or earliest index in 2024/2025): 210,919

## SECONDARY split — patient-grouped random 70/15/15, stratified by class (sensitivity analysis)
| partition | pairs | patients | class balance |
|---|---:|---:|---|
| train | 397,387 | 106,225 | improved 105,227 (26.5%), stable 193,031 (48.6%), worsened 99,129 (24.9%) |
| validation | 86,676 | 22,762 | improved 23,183 (26.7%), stable 41,724 (48.1%), worsened 21,769 (25.1%) |
| test | 85,138 | 22,763 | improved 22,537 (26.5%), stable 41,339 (48.6%), worsened 21,262 (25.0%) |

## Cleaning summary (rows removed)
| analyte | raw | dropped missing val/ts | out-of-range | exact dupes | kept |
|---|---:|---:|---:|---:|---:|
| HBA1C | 1,139,747 | 90 | 7,618 | 584 | 1,131,455 |
| FBS | 909,870 | 2 | 5,286 | 291 | 904,291 |
| CREATININE | 1,196,292 | 3 | 10,536 | 743 | 1,185,010 |
| BUN | 175,915 | 1 | 387 | 26 | 175,501 |
| LDL | 252,440 | 2 | 9,405 | 107 | 242,926 |
| TRIGLYCERIDES | 467,632 | 2 | 3,207 | 332 | 464,091 |
| TOTAL_CHOLESTEROL | 483,905 | 2 | 1,921 | 305 | 481,677 |

## Files written to Create_results/
- `cleaned_hba1c.parquet`
- `cohort_pairs.csv`
- `sequences.jsonl`
- `splits_temporal.jsonl`
- `splits_random_grouped.jsonl`
- `cleaning_report.txt`
- `cleaning_counts_by_analyte.csv`
- `cohort_summary.md`
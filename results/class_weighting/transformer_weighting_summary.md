# Class-weighted versus unweighted transformer (V8 item 7)

Weighted transformer, macro AUROC raw 0.7264, calibrated 0.7314; worsened-class ECE after calibration 0.0087.
Unweighted transformer, macro AUROC raw 0.7312, calibrated 0.7320; worsened-class ECE after calibration 0.0109.

Unweighted minus weighted, macro AUROC: raw +0.0048 (+0.0038 to +0.0058); calibrated +0.0005 (-0.0001 to +0.0011).
LightGBM minus unweighted transformer (calibrated): +0.0316 (+0.0282 to +0.0350).
Unweighted transformer minus weighted logistic regression (calibrated): +0.0389 (+0.0352 to +0.0425).

Model ranking LightGBM > transformer > baseline retained: YES.
Manuscript update required: NO (rule: calibrated macro AUROC changes by 0.005 or more, or the ranking changes).

If NO: add one sentence to Section 4.5 and a row to Supplementary Table S21 reporting these values, and replace 'the corresponding transformer run is provided as a script' in Section 2.12.
If YES: send these three output files back for a full revision of Sections 2.12, 4.5, 4.7 and Table S21.
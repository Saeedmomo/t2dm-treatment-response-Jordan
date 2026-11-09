"""Metric helpers for 3-class HbA1c-trajectory evaluation (classes: improved=0, stable=1, worsened=2)."""
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
                             roc_curve, balanced_accuracy_score, log_loss)

CLASSES = ["improved", "stable", "worsened"]

def onehot(y, k=3):
    O = np.zeros((len(y), k)); O[np.arange(len(y)), y] = 1; return O

def auroc_all(y, P):
    out = {}
    for i, c in enumerate(CLASSES):
        out[f"auroc_{c}"] = roc_auc_score((y == i).astype(int), P[:, i])
    out["auroc_macro"] = roc_auc_score(y, P, multi_class="ovr", average="macro")
    out["auroc_micro"] = roc_auc_score(onehot(y).ravel(), P.ravel())
    return out

def auprc_all(y, P):
    return {f"auprc_{c}": average_precision_score((y == i).astype(int), P[:, i])
            for i, c in enumerate(CLASSES)}

def brier_multiclass(y, P):
    return float(np.mean(np.sum((P - onehot(y))**2, axis=1)))

def ece_ovr(y, P, bins=10):
    out = {}; eces = []
    edges = np.linspace(0, 1, bins+1)
    for i, c in enumerate(CLASSES):
        p = P[:, i]; yb = (y == i).astype(int); e = 0.0
        for b in range(bins):
            m = (p >= edges[b]) & (p < edges[b+1]) if b < bins-1 else (p >= edges[b]) & (p <= edges[b+1])
            if m.sum() > 0:
                e += m.mean() * abs(yb[m].mean() - p[m].mean())
        out[f"ece_{c}"] = e; eces.append(e)
    out["ece_mean"] = float(np.mean(eces)); return out

def calibration_slope_intercept(yb, p):
    """Logistic recalibration: fit yb ~ logit(p). slope=1, intercept=0 is perfect."""
    from sklearn.linear_model import LogisticRegression
    eps = 1e-6; logit = np.log(np.clip(p, eps, 1-eps) / np.clip(1-p, eps, 1-eps)).reshape(-1, 1)
    if len(np.unique(yb)) < 2:
        return np.nan, np.nan
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(logit, yb)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])

def operating_metrics_argmax(y, P):
    yhat = P.argmax(1)
    cm = confusion_matrix(y, yhat, labels=[0, 1, 2])
    rows = {}
    n = len(y)
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]; fn = cm[i, :].sum() - tp; fp = cm[:, i].sum() - tp; tn = n - tp - fn - fp
        sens = tp/(tp+fn) if tp+fn else np.nan
        spec = tn/(tn+fp) if tn+fp else np.nan
        ppv = tp/(tp+fp) if tp+fp else np.nan
        npv = tn/(tn+fn) if tn+fn else np.nan
        f1 = 2*ppv*sens/(ppv+sens) if (ppv and sens and not np.isnan(ppv) and not np.isnan(sens) and (ppv+sens) > 0) else np.nan
        rows[c] = dict(sensitivity=sens, specificity=spec, ppv=ppv, npv=npv, f1=f1)
    rows["_balanced_accuracy"] = balanced_accuracy_score(y, yhat)
    return cm, rows

def youden_threshold(yb, p):
    fpr, tpr, thr = roc_curve(yb, p)
    j = tpr - fpr; k = np.argmax(j)
    return float(thr[k]), float(tpr[k]), float(1-fpr[k])   # threshold, sens, spec

def binary_op_at_threshold(yb, p, t):
    yhat = (p >= t).astype(int)
    tp = int(((yhat==1)&(yb==1)).sum()); fp = int(((yhat==1)&(yb==0)).sum())
    fn = int(((yhat==0)&(yb==1)).sum()); tn = int(((yhat==0)&(yb==0)).sum())
    sens = tp/(tp+fn) if tp+fn else np.nan
    spec = tn/(tn+fp) if tn+fp else np.nan
    ppv = tp/(tp+fp) if tp+fp else np.nan
    npv = tn/(tn+fn) if tn+fn else np.nan
    f1 = 2*ppv*sens/(ppv+sens) if (ppv and sens and (ppv+sens)>0) else np.nan
    return dict(threshold=t, sensitivity=sens, specificity=spec, ppv=ppv, npv=npv, f1=f1)

def temperature_scale(P_val, y_val):
    """Fit single temperature T on validation pseudo-logits (log P). Returns T."""
    eps = 1e-8; logits = np.log(np.clip(P_val, eps, 1))
    from scipy.optimize import minimize_scalar
    def nll(T):
        z = logits / T; z = z - z.max(1, keepdims=True)
        sm = np.exp(z); sm /= sm.sum(1, keepdims=True)
        return -np.mean(np.log(sm[np.arange(len(y_val)), y_val] + eps))
    r = minimize_scalar(nll, bounds=(0.3, 5.0), method="bounded")
    return float(r.x)

def apply_temperature(P, T):
    eps = 1e-8; logits = np.log(np.clip(P, eps, 1)) / T
    logits = logits - logits.max(1, keepdims=True)
    sm = np.exp(logits); sm /= sm.sum(1, keepdims=True); return sm

def decision_curve(yb, p, pts):
    """Net benefit for a one-vs-rest positive class across threshold probabilities pts."""
    n = len(yb); prev = yb.mean(); nb_model = []; nb_all = []
    for pt in pts:
        yhat = (p >= pt).astype(int)
        tp = ((yhat==1)&(yb==1)).sum(); fp = ((yhat==1)&(yb==0)).sum()
        w = pt/(1-pt)
        nb_model.append(tp/n - fp/n*w)
        nb_all.append(prev - (1-prev)*w)
    return np.array(nb_model), np.array(nb_all)

def bootstrap_ci(metric_fn, y, P, n_boot=2000, seed=0):
    """metric_fn(y,P)->scalar. Stratified-by-nothing simple bootstrap over rows."""
    rng = np.random.default_rng(seed); n = len(y); vals = np.empty(n_boot)
    idx_all = np.arange(n)
    for b in range(n_boot):
        idx = rng.choice(idx_all, n, replace=True)
        vals[b] = metric_fn(y[idx], P[idx])
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def paired_bootstrap_diff(metric_fn, y, PA, PB, n_boot=2000, seed=0):
    """Paired bootstrap of metric(A)-metric(B); returns diff mean, CI, two-sided p."""
    rng = np.random.default_rng(seed); n = len(y); diffs = np.empty(n_boot)
    idx_all = np.arange(n)
    for b in range(n_boot):
        idx = rng.choice(idx_all, n, replace=True)
        diffs[b] = metric_fn(y[idx], PA[idx]) - metric_fn(y[idx], PB[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2*min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(diffs.mean()), float(lo), float(hi), float(min(p, 1.0))

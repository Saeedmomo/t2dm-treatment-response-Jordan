"""
Full evaluation battery on the PRIMARY temporal test set.
Consumes saved probability files in Create_results/preds/. Transformer = ensemble mean of its
3 seeds for headline metrics (+ per-seed mean/SD reported separately).

Outputs tables to Create_results/ and figures to Create_figures/.
"""
import os, sys, glob, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from common import RES, FIG, PREDS
import metrics_lib as M

os.makedirs(FIG, exist_ok=True)
CLASSES = ["improved", "stable", "worsened"]

def load_probs(model, eval_name):
    files = sorted(glob.glob(os.path.join(PREDS, f"{model}__{eval_name}__seed*.parquet")))
    if not files: return None, None, None
    dfs = [pd.read_parquet(f).sort_values(["patient_id","index_date"]).reset_index(drop=True) for f in files]
    y = dfs[0]["label_id"].to_numpy()
    Ps = [d[["p_improved","p_stable","p_worsened"]].to_numpy() for d in dfs]
    P = np.mean(Ps, axis=0)
    return y, P, Ps

MODELS = ["transformer", "lgbm", "logreg"]
NICE = {"transformer":"Transformer (BEHRT)", "lgbm":"LightGBM", "logreg":"LogReg (baseline)"}

def main():
    # load test + validation probabilities
    test = {m: load_probs(m, "temporal_test") for m in MODELS}
    val  = {m: load_probs(m, "temporal_validation") for m in MODELS}
    y = test["lgbm"][0]   # identical labels across models (same rows/order)

    # ---------- headline discrimination + calibration table (with bootstrap CIs) ----------
    rows = []
    for m in MODELS:
        yt, P, _ = test[m]
        d = {"model": NICE[m]}
        d.update(M.auroc_all(yt, P)); d.update(M.auprc_all(yt, P))
        d["brier"] = M.brier_multiclass(yt, P)
        d.update(M.ece_ovr(yt, P))
        cm, opr = M.operating_metrics_argmax(yt, P)
        d["balanced_accuracy"] = opr["_balanced_accuracy"]
        # bootstrap CIs on key metrics
        for label, fn in [("auroc_macro", lambda yy,PP: M.auroc_all(yy,PP)["auroc_macro"]),
                          ("auprc_worsened", lambda yy,PP: M.auprc_all(yy,PP)["auprc_worsened"]),
                          ("brier", M.brier_multiclass),
                          ("balanced_accuracy", lambda yy,PP: M.operating_metrics_argmax(yy,PP)[1]["_balanced_accuracy"])]:
            mean, lo, hi = M.bootstrap_ci(fn, yt, P, n_boot=2000, seed=1)
            d[f"{label}_CI"] = f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"
        rows.append(d)
        # confusion matrix
        pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES], columns=[f"pred_{c}" for c in CLASSES]
                     ).to_csv(os.path.join(RES, f"confusion_{m}.csv"))
    main_tab = pd.DataFrame(rows)
    main_tab.to_csv(os.path.join(RES, "main_results_test.csv"), index=False)
    print("=== MAIN RESULTS (temporal test) ===")
    print(main_tab[["model","auroc_macro","auroc_micro","auroc_worsened","auprc_worsened","brier","ece_mean","balanced_accuracy"]].round(4).to_string(index=False))

    # ---------- per-class operating metrics (argmax) ----------
    op_rows = []
    for m in MODELS:
        yt, P, _ = test[m]
        _, opr = M.operating_metrics_argmax(yt, P)
        for c in CLASSES:
            op_rows.append({"model": NICE[m], "class": c, **{k: round(v,4) for k,v in opr[c].items()}})
    pd.DataFrame(op_rows).to_csv(os.path.join(RES, "per_class_operating_metrics.csv"), index=False)

    # ---------- operating threshold for 'worsened' (tuned on validation, Youden J) ----------
    thr_rows = []
    for m in MODELS:
        yv, Pv, _ = val[m]; yt, Pt, _ = test[m]
        t, sens_v, spec_v = M.youden_threshold((yv==2).astype(int), Pv[:,2])
        op = M.binary_op_at_threshold((yt==2).astype(int), Pt[:,2], t)
        thr_rows.append({"model": NICE[m], "worsened_threshold(val Youden)": round(t,3),
                         **{k: round(v,4) for k,v in op.items() if k!="threshold"}})
    pd.DataFrame(thr_rows).to_csv(os.path.join(RES, "worsened_operating_point.csv"), index=False)
    print("\n=== 'worsened' operating point (threshold tuned on validation, applied to test) ===")
    print(pd.DataFrame(thr_rows).to_string(index=False))

    # ---------- calibration: slope/intercept + temperature scaling before/after ----------
    cal_rows = []
    for m in MODELS:
        yv, Pv, _ = val[m]; yt, Pt, _ = test[m]
        T = M.temperature_scale(Pv, yv)
        Pt_ts = M.apply_temperature(Pt, T)
        for tag, PP in [("raw", Pt), ("temp_scaled", Pt_ts)]:
            row = {"model": NICE[m], "variant": tag, "temperature": round(T,3),
                   "brier": round(M.brier_multiclass(yt, PP),4), **{k: round(v,4) for k,v in M.ece_ovr(yt, PP).items()}}
            for i,c in enumerate(CLASSES):
                s, ic = M.calibration_slope_intercept((yt==i).astype(int), PP[:,i])
                row[f"slope_{c}"] = round(s,3); row[f"intercept_{c}"] = round(ic,3)
            cal_rows.append(row)
    pd.DataFrame(cal_rows).to_csv(os.path.join(RES, "calibration_metrics.csv"), index=False)

    # ---------- model comparison: paired bootstrap AUROC differences ----------
    cmp_rows = []
    macro = lambda yy,PP: M.auroc_all(yy,PP)["auroc_macro"]
    wors  = lambda yy,PP: M.auroc_all(yy,PP)["auroc_worsened"]
    for a,b in [("transformer","lgbm"),("transformer","logreg"),("lgbm","logreg")]:
        for label, fn in [("AUROC_macro", macro), ("AUROC_worsened", wors)]:
            diff, lo, hi, p = M.paired_bootstrap_diff(fn, y, test[a][1], test[b][1], n_boot=2000, seed=7)
            cmp_rows.append({"metric": label, "comparison": f"{NICE[a]} - {NICE[b]}",
                             "diff": round(diff,4), "CI95": f"[{lo:.4f}, {hi:.4f}]", "p_value": round(p,4)})
    pd.DataFrame(cmp_rows).to_csv(os.path.join(RES, "model_comparison_auroc.csv"), index=False)
    print("\n=== MODEL COMPARISON (paired bootstrap AUROC diff) ===")
    print(pd.DataFrame(cmp_rows).to_string(index=False))

    # ---------- transformer seed variability ----------
    yt, P, Ps = test["transformer"]
    if Ps and len(Ps) > 1:
        seed_vals = [M.auroc_all(yt, Pi)["auroc_macro"] for Pi in Ps]
        wvals = [M.auroc_all(yt, Pi)["auroc_worsened"] for Pi in Ps]
        pd.DataFrame({"seed": range(len(Ps)), "auroc_macro": seed_vals, "auroc_worsened": wvals}).to_csv(
            os.path.join(RES, "transformer_seed_variability.csv"), index=False)
        print(f"\nTransformer AUROC macro over {len(Ps)} seeds: {np.mean(seed_vals):.4f} +/- {np.std(seed_vals):.4f}")

    # ================= FIGURES =================
    # ROC per class (ovr), all models
    from sklearn.metrics import roc_curve
    fig, axes = plt.subplots(1, 3, figsize=(15,4.5))
    for i,c in enumerate(CLASSES):
        ax = axes[i]
        for m in MODELS:
            yt, P, _ = test[m]
            fpr, tpr, _ = roc_curve((yt==i).astype(int), P[:,i])
            auc = M.auroc_all(yt,P)[f"auroc_{c}"]
            ax.plot(fpr, tpr, label=f"{NICE[m]} ({auc:.3f})")
        ax.plot([0,1],[0,1],'k--',lw=.8); ax.set_title(f"ROC one-vs-rest: {c}")
        ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"roc_ovr.png"), dpi=140); plt.close()

    # Reliability diagrams (rows=models, cols=classes) raw vs temp-scaled for transformer
    fig, axes = plt.subplots(len(MODELS), 3, figsize=(13, 4*len(MODELS)))
    for r,m in enumerate(MODELS):
        yt, P, _ = test[m]
        for i,c in enumerate(CLASSES):
            ax = axes[r,i]; p = P[:,i]; yb=(yt==i).astype(int)
            edges=np.linspace(0,1,11); mids=[]; obs=[]
            for b in range(10):
                mk=(p>=edges[b])&(p<edges[b+1] if b<9 else p<=edges[b+1])
                if mk.sum()>0: mids.append(p[mk].mean()); obs.append(yb[mk].mean())
            ax.plot([0,1],[0,1],'k--',lw=.8); ax.plot(mids,obs,'o-')
            ax.set_title(f"{NICE[m]} — {c}"); ax.set_xlabel("mean predicted"); ax.set_ylabel("observed freq")
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"reliability_diagrams.png"), dpi=140); plt.close()

    # Decision curve for 'worsened'
    pts = np.linspace(0.05, 0.6, 45)
    plt.figure(figsize=(8,5.5))
    yb = (y==2).astype(int)
    nb_all_ref=None
    for m in MODELS:
        yt, P, _ = test[m]
        nb_model, nb_all = M.decision_curve(yb, P[:,2], pts); nb_all_ref=nb_all
        plt.plot(pts, nb_model, label=NICE[m])
    plt.plot(pts, nb_all_ref, 'k--', label="treat all")
    plt.axhline(0, color="gray", lw=.8, label="treat none")
    plt.ylim(bottom=max(-0.05, min(nb_all_ref.min(), -0.05)))
    plt.xlabel("threshold probability"); plt.ylabel("net benefit")
    plt.title("Decision curve — 'worsened' (one-vs-rest)"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"decision_curve_worsened.png"), dpi=140); plt.close()

    # Temperature scaling before/after (transformer) reliability, worsened class
    yv,Pv,_ = val["transformer"]; yt,Pt,_=test["transformer"]
    T = M.temperature_scale(Pv,yv); Pts=M.apply_temperature(Pt,T)
    plt.figure(figsize=(7,5.5))
    for tag,PP,mk in [("raw",Pt,'o-'),(f"temp-scaled (T={T:.2f})",Pts,'s-')]:
        p=PP[:,2]; yb2=(yt==2).astype(int); edges=np.linspace(0,1,11); mids=[];obs=[]
        for b in range(10):
            m2=(p>=edges[b])&(p<edges[b+1] if b<9 else p<=edges[b+1])
            if m2.sum()>0: mids.append(p[m2].mean()); obs.append(yb2[m2].mean())
        plt.plot(mids,obs,mk,label=tag)
    plt.plot([0,1],[0,1],'k--',lw=.8); plt.xlabel("mean predicted"); plt.ylabel("observed")
    plt.title("Transformer calibration — worsened, before/after temperature scaling"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"calibration_temp_scaling_worsened.png"), dpi=140); plt.close()

    # Confusion matrices
    fig, axes = plt.subplots(1,3, figsize=(14,4.2))
    for ax,m in zip(axes, MODELS):
        yt,P,_=test[m]; cm,_=M.operating_metrics_argmax(yt,P)
        cmn = cm/cm.sum(1,keepdims=True)
        im=ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASSES, rotation=30); ax.set_yticklabels(CLASSES)
        for i in range(3):
            for j in range(3):
                ax.text(j,i,f"{cm[i,j]}\n{cmn[i,j]*100:.0f}%",ha="center",va="center",fontsize=8,
                        color="white" if cmn[i,j]>0.5 else "black")
        ax.set_title(NICE[m]); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"confusion_matrices.png"), dpi=140); plt.close()

    print("\nEvaluation figures + tables written.")

if __name__ == "__main__":
    main()

"""Model-comparison and evaluation runner. Run: python -m skilllens.evaluate"""
import json, joblib, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from . import SEED
from .data import generate_dataset
from .features import FEATURES, build_preprocessor

TARGET = "placed"

def split(df):
    """Fixed stratified 60/20/20 train/val/test split."""
    tr, tmp = train_test_split(df, test_size=0.4, stratify=df[TARGET], random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.5, stratify=tmp[TARGET], random_state=SEED)
    return tr, va, te

def leakage_check(df, tr, va, te):
    corr = df[FEATURES].select_dtypes("number").corrwith(df[TARGET]).abs().max()
    overlap = len(set(tr.index) & set(te.index)) + len(set(tr.index) & set(va.index)) + len(set(va.index) & set(te.index))
    assert TARGET not in FEATURES, "target in features"
    assert corr < 0.9, f"suspiciously high feature-target correlation {corr:.2f}"
    assert overlap == 0, "index overlap between splits"
    return {"max_abs_feature_target_corr": round(float(corr), 3), "split_overlap_rows": overlap,
            "target_in_features": False}

CANDIDATES = {
    "LogisticRegression (baseline)": (LogisticRegression(max_iter=2000), [{"C": c} for c in (0.1, 1, 10)]),
    "RandomForest": (RandomForestClassifier(random_state=SEED, n_jobs=-1),
                     [{"n_estimators": n, "min_samples_leaf": m} for n in (200, 400) for m in (3, 8)]),
    "HistGradientBoosting": (HistGradientBoostingClassifier(random_state=SEED),
                     [{"learning_rate": lr, "max_depth": d, "max_iter": 150} for lr in (0.03, 0.08) for d in (3, 5)]),
}

def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return dict(precision=precision_score(y, pred), recall=recall_score(y, pred),
                f1=f1_score(y, pred), roc_auc=roc_auc_score(y, p))

def failure_analysis(te, p, n=25):
    d = te.copy(); d["prob_placed"] = p.round(3); d["predicted"] = (p >= .5).astype(int)
    d["margin"] = abs(d.prob_placed - .5)
    d["wrong"] = d.predicted != d[TARGET]
    bad = d[d.wrong | (d.margin < .1)].sort_values(["wrong", "margin"], ascending=[False, True]).head(n).copy()
    def cause(r):
        c = []
        if r.margin < .1: c.append("borderline probability")
        if r[["cgpa", "aptitude_score", "branch"]].isna().any(): c.append("missing key field imputed")
        if r.cgpa >= 8 and r.internships == 0 and r.projects <= 1: c.append("strong academics but thin practical profile")
        if r.cgpa < 6.5 and r.internships >= 2: c.append("weak academics offset by experience (rare profile)")
        if r.backlogs > 0 and r.cgpa >= 7.5: c.append("backlog despite high CGPA")
        return "; ".join(c) or "inherent label noise (unobserved factors, e.g. interview outcome)"
    bad["likely_cause"] = bad.apply(cause, axis=1)
    return bad.drop(columns="margin")

def main():
    df = generate_dataset(); df.to_csv("data/students.csv", index=False)
    tr, va, te = split(df)
    leak = leakage_check(df, tr, va, te)
    rows, fitted = [], {}
    for name, (est, grid) in CANDIDATES.items():
        best = None
        for params in grid:                       # tune on VALIDATION only
            pipe = Pipeline([("pre", build_preprocessor()), ("clf", est.__class__(**{**est.get_params(), **params}))])
            pipe.fit(tr[FEATURES], tr[TARGET])
            auc = roc_auc_score(va[TARGET], pipe.predict_proba(va[FEATURES])[:, 1])
            if best is None or auc > best[0]: best = (auc, params, pipe)
        auc, params, pipe = best
        p = pipe.predict_proba(te[FEATURES])[:, 1]  # SAME test set for every model
        rows.append({"model": name, "val_auc": round(auc, 4), "params": json.dumps(params),
                     **{k: round(v, 4) for k, v in metrics(te[TARGET], p).items()}})
        fitted[name] = (pipe, p)
    res = pd.DataFrame(rows); res.to_csv("reports/model_comparison.csv", index=False)
    champ = res.sort_values("val_auc").iloc[-1].model     # chosen on validation, not test
    pipe, p = fitted[champ]
    joblib.dump(pipe, "reports/best_model.joblib")
    cm = confusion_matrix(te[TARGET], (p >= .5).astype(int))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["Not placed", "Placed"]).plot(ax=ax[0], colorbar=False)
    for n_, (_, pp) in fitted.items(): RocCurveDisplay.from_predictions(te[TARGET], pp, name=n_.split(" (")[0], ax=ax[1])
    ax[0].set_title(f"Confusion matrix - {champ}"); ax[1].set_title("ROC (same test set)")
    plt.tight_layout(); plt.savefig("reports/eval_plots.png", dpi=130)
    fa = failure_analysis(te, p); fa.to_csv("reports/failure_analysis.csv")
    json.dump({"champion": champ, "confusion_matrix": cm.tolist(), "leakage_check": leak,
               "split_sizes": [len(tr), len(va), len(te)], "n_failure_cases": len(fa)},
              open("reports/summary.json", "w"), indent=2)
    print(res.drop(columns="params").to_string(index=False)); print("champion:", champ, "| leakage:", leak)

if __name__ == "__main__":
    main()

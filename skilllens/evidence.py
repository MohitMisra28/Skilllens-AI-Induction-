"""Extra evidence generators. Run: python -m skilllens.evidence
1) robustness under missing values / noise, 2) individual explanations, 3) architecture PNG."""
import joblib, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_auc_score
from . import SEED
from .evaluate import TARGET, split
from .explain import explain_student, get_reference
from .features import FEATURES, NUM

def robustness(pipe, te, levels=(0, .1, .2, .4), seed=SEED):
    r = np.random.default_rng(seed); rows = []
    for lv in levels:
        X = te[FEATURES].copy()
        mask = r.random(X.shape) < lv                      # random missingness in any column
        Xm = X.mask(mask)
        Xn = X.copy()
        for c in NUM: Xn[c] = Xn[c] + r.normal(0, lv * te[c].std(), len(X))   # gaussian noise in std units
        for name, Z in (("missing_fraction", Xm), ("noise_std_multiple", Xn)):
            rows.append(dict(kind=name, level=lv, roc_auc=round(roc_auc_score(te[TARGET], pipe.predict_proba(Z)[:, 1]), 4)))
    unseen = te[FEATURES].copy(); unseen["branch"] = "Astrophysics"; unseen["college_tier"] = "Tier9"
    rows.append(dict(kind="unseen_categories", level=1.0, roc_auc=round(roc_auc_score(te[TARGET], pipe.predict_proba(unseen)[:, 1]), 4)))
    return pd.DataFrame(rows)

def individual_explanations(pipe, te, ref, fig_path, md_path):
    p = pipe.predict_proba(te[FEATURES])[:, 1]; d = te.assign(p=p, pred=(p >= .5).astype(int))
    picks = [("Correct: high readiness", d[(d.placed == 1) & (d.p > .85)].head(2)),
             ("Correct: low readiness", d[(d.placed == 0) & (d.p < .2)].head(2)),
             ("Wrong: predicted placed, was not", d[(d.placed == 0) & (d.p > .75)].head(1)),
             ("Wrong: predicted not placed, was placed", d[(d.placed == 1) & (d.p < .25)].head(1))]
    cases = [(t, i, r) for t, df in picks for i, r in df.iterrows()]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8)); md = ["# Individual Prediction Explanations\n",
        "Impact = change in placement probability versus an average student (positive helps, negative hurts).\n"]
    for ax, (title, idx, row) in zip(axes.ravel(), cases):
        prob, e = explain_student(pipe, {f: row[f] for f in FEATURES}, ref, top=6)
        e = e.sort_values("impact")
        ax.barh(e.feature, e.impact, color=np.where(e.impact >= 0, "#2e9e5b", "#d64545"))
        ax.set_title(f"{title}\nstudent #{idx} | P(placed)={prob:.2f} | actual={int(row.placed)}", fontsize=9)
        pos = ", ".join(f"{r.feature}={r.value:g}" if isinstance(r.value, (int, float, np.floating)) else f"{r.feature}={r.value}" for r in e[e.impact > 0].iloc[::-1].itertuples())
        neg = ", ".join(f"{r.feature}={r.value:g}" if isinstance(r.value, (int, float, np.floating)) else f"{r.feature}={r.value}" for r in e[e.impact < 0].itertuples())
        md.append(f"## {title} - student #{idx}\n- P(placed) = **{prob:.2f}**, actual = **{int(row.placed)}**\n- Helping: {pos or '-'}\n- Hurting: {neg or '-'}\n")
    plt.tight_layout(); plt.savefig(fig_path, dpi=120); plt.close()
    open(md_path, "w").write("\n".join(md) + "\n![explanations](individual_explanations.png)\n")

def architecture_png(path):
    fig, ax = plt.subplots(figsize=(14, 5)); ax.axis("off"); ax.set_xlim(0, 14); ax.set_ylim(0, 5)
    boxes = [(.2, 3.4, "data.py\n3000 students"), (2.9, 3.4, "Stratified split\n60/20/20, seed 42"), (5.6, 3.4, "Leakage checks"),
             (8.3, 3.4, "features.py\nengineer > impute\n> scale / one-hot"), (11.0, 3.4, "Train 3 models\ntune on VALIDATION"),
             (11.0, 1.7, "Score on same TEST set\nchampion = val AUC"), (8.3, 1.7, "reports/\nmodel, metrics, plots"),
             (5.6, 1.7, "explain.py\npermutation + occlusion\n+ what-if"), (2.9, 1.7, "app.py (Streamlit)\ndashboard, single, batch"), (.2, 1.7, "User\nCSV / form input")]
    for x, y, t in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 2.5, 1.1, boxstyle="round,pad=0.05", fc="#eef3fb", ec="#33507a"))
        ax.text(x + 1.25, y + .55, t, ha="center", va="center", fontsize=8.5)
    for i in range(len(boxes) - 1):
        (x1, y1, _), (x2, y2, _) = boxes[i], boxes[i + 1]
        if y1 == y2: ax.annotate("", (x2 if x2 > x1 else x2 + 2.5, y2 + .55), (x1 + 2.5 if x2 > x1 else x1, y1 + .55), arrowprops=dict(arrowstyle="->"))
        else: ax.annotate("", (x2 + 1.25, y2 + 1.1), (x1 + 1.25, y1), arrowprops=dict(arrowstyle="->"))
    ax.set_title("SkillLens architecture", fontsize=13); plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()

def main():
    df = pd.read_csv("data/students.csv"); _, _, te = split(df)
    pipe = joblib.load("reports/best_model.joblib"); ref = get_reference(df)
    rb = robustness(pipe, te); rb.to_csv("reports/robustness.csv", index=False); print(rb.to_string(index=False))
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, g in rb[rb.kind != "unseen_categories"].groupby("kind"): ax.plot(g.level, g.roc_auc, marker="o", label=k)
    ax.set_xlabel("degradation level"); ax.set_ylabel("test ROC-AUC"); ax.legend(); ax.set_title("Robustness"); plt.tight_layout()
    plt.savefig("reports/robustness.png", dpi=120); plt.close()
    individual_explanations(pipe, te, ref, "reports/individual_explanations.png", "reports/individual_explanations.md")
    architecture_png("reports/architecture.png")

if __name__ == "__main__":
    main()

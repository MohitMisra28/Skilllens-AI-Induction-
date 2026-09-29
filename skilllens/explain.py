"""Own explanation + what-if logic (model-agnostic, works on the full pipeline).
Global: permutation importance on raw columns.
Local: occlusion - replace one feature with the cohort reference value and measure
the drop/rise in placement probability."""
import numpy as np, pandas as pd
from sklearn.inspection import permutation_importance
from . import SEED
from .features import CAT, FEATURES, NUM

def get_reference(df):
    ref = {c: float(df[c].median()) for c in NUM}
    ref.update({c: df[c].mode().iloc[0] for c in CAT})
    return ref

def band(p):
    return "High" if p >= 0.7 else "Medium" if p >= 0.4 else "Low"

def global_importance(pipe, X, y, n_repeats=10):
    r = permutation_importance(pipe, X[FEATURES], y, scoring="roc_auc", n_repeats=n_repeats, random_state=SEED)
    return (pd.DataFrame({"feature": FEATURES, "importance": r.importances_mean, "std": r.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))

def _score(pipe, X): return pipe.predict_proba(X[FEATURES])[:, 1]

def occlusion_impacts(pipe, X, ref):
    """Rows x features matrix of (p_actual - p_with_feature_at_reference). >0 => feature helps."""
    X = X[FEATURES].reset_index(drop=True); base = _score(pipe, X); out = {}
    for f in FEATURES:
        Z = X.copy(); Z[f] = ref[f]
        out[f] = base - _score(pipe, Z)
    return pd.DataFrame(out), base

def explain_student(pipe, row, ref, top=None):
    imp, base = occlusion_impacts(pipe, pd.DataFrame([row]), ref)
    d = pd.DataFrame({"feature": FEATURES, "value": [row.get(f) for f in FEATURES],
                      "reference": [ref[f] for f in FEATURES], "impact": imp.iloc[0].values})
    d["direction"] = np.where(d.impact >= 0, "positive", "negative")
    d = d.reindex(d.impact.abs().sort_values(ascending=False).index).reset_index(drop=True)
    return float(base[0]), (d.head(top) if top else d)

def predict_batch(pipe, df, ref):
    """Batch scoring with band + top positive / negative factor per student."""
    imp, base = occlusion_impacts(pipe, df, ref)
    out = df.reset_index(drop=True).copy()
    out["readiness_score"] = (base * 100).round(1)
    out["readiness_band"] = [band(p) for p in base]
    out["top_positive_factor"] = imp.idxmax(axis=1).where(imp.max(axis=1) > 0, "-")
    out["top_negative_factor"] = imp.idxmin(axis=1).where(imp.min(axis=1) < 0, "-")
    return out

def what_if(pipe, row, changes, ref):
    """Apply {feature: new_value} to a student and return (before, after) probabilities."""
    new = {**row, **changes}
    both = pd.DataFrame([row, new])
    p = _score(pipe, both)
    return float(p[0]), float(p[1])

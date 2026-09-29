import joblib, pandas as pd
from skilllens.evaluate import split
from skilllens.evidence import robustness

def test_robustness_degrades_gracefully():
    _, _, te = split(pd.read_csv("data/students.csv")); pipe = joblib.load("reports/best_model.joblib")
    rb = robustness(pipe, te)
    base = rb[(rb.kind == "missing_fraction") & (rb.level == 0)].roc_auc.iloc[0]
    worst = rb[rb.kind != "unseen_categories"].roc_auc.min()
    assert base > 0.75 and worst > 0.6            # degraded but still better than chance
    assert rb[rb.kind == "unseen_categories"].roc_auc.iloc[0] > 0.6

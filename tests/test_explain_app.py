import joblib, numpy as np, pandas as pd, pytest
from streamlit.testing.v1 import AppTest
from skilllens.explain import explain_student, get_reference, global_importance, predict_batch, what_if
from skilllens.features import FEATURES

@pytest.fixture(scope="module")
def ctx():
    df = pd.read_csv("data/students.csv")
    return joblib.load("reports/best_model.joblib"), df, get_reference(df)

def test_local_explanation_has_pos_and_neg(ctx):
    pipe, df, ref = ctx
    strong = {**ref, "cgpa": 9.5, "internships": 4, "backlogs": 0}
    weak = {**ref, "cgpa": 5.0, "backlogs": 6, "internships": 0}
    ps, es = explain_student(pipe, strong, ref); pw, ew = explain_student(pipe, weak, ref)
    assert ps > pw and (es.impact > 0).any() and (ew.impact < 0).any()

def test_what_if_more_experience_helps(ctx):
    pipe, _, ref = ctx
    b, a = what_if(pipe, dict(ref), {"internships": 4, "projects": 6}, ref)
    assert a > b

def test_batch_prediction_shape_and_bands(ctx):
    pipe, df, ref = ctx
    res = predict_batch(pipe, df.drop(columns="placed").head(50), ref)
    assert len(res) == 50 and res.readiness_score.between(0, 100).all()
    assert set(res.readiness_band) <= {"High", "Medium", "Low"}

def test_batch_handles_partial_columns(ctx):
    pipe, _, ref = ctx
    res = predict_batch(pipe, pd.DataFrame({"cgpa": [8, "bad"], "internships": [1, np.nan]}).reindex(columns=FEATURES), ref)
    assert res.readiness_score.notna().all()

def test_global_importance_top_feature_is_sensible(ctx):
    pipe, df, _ = ctx
    gi = global_importance(pipe, df.sample(600, random_state=0), df.placed.loc[df.sample(600, random_state=0).index], 3)
    assert gi.feature.head(6).isin(["cgpa", "aptitude_score", "internships", "projects", "college_tier", "backlogs", "branch"]).any()

def test_app_renders_without_exception():
    at = AppTest.from_file(str(__import__("pathlib").Path(__file__).parents[1] / "app.py")).run(timeout=120)
    assert not at.exception

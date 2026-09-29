"""SkillLens dashboard. Run: streamlit run app.py"""
import json
from pathlib import Path
import joblib, pandas as pd, plotly.express as px, streamlit as st
from skilllens.data import BRANCHES, TIERS
from skilllens.explain import (band, explain_student, get_reference, global_importance,
                               predict_batch, what_if)
from skilllens.features import FEATURES

ROOT = Path(__file__).parent
st.set_page_config(page_title="SkillLens", page_icon="🎯", layout="wide")

@st.cache_resource
def load():
    df = pd.read_csv(ROOT / "data/students.csv")
    return joblib.load(ROOT / "reports/best_model.joblib"), df, get_reference(df)
pipe, data, ref = load()

@st.cache_data
def scored(): return predict_batch(pipe, data.drop(columns="placed"), ref).assign(placed=data.placed.values)
@st.cache_data
def gimp():
    te = data.sample(800, random_state=1)
    return global_importance(pipe, te, te.placed, n_repeats=5)

st.title("🎯 SkillLens")
st.caption("Predict placement readiness and explain the factors behind the result.")
t1, t2, t3, t4 = st.tabs(["📊 Dashboard", "🧑‍🎓 Single student + what-if", "📁 Batch CSV", "🔬 Model evidence"])

with t1:
    S = scored()
    f1, f2 = st.columns(2)
    br = f1.multiselect("Branch", BRANCHES, default=BRANCHES)
    ti = f2.multiselect("College tier", TIERS, default=TIERS)
    V = S[S.branch.isin(br) & S.college_tier.isin(ti)]
    c = st.columns(4)
    c[0].metric("Students", len(V))
    c[1].metric("Avg readiness", f"{V.readiness_score.mean():.1f}" if len(V) else "-")
    c[2].metric("High-readiness share", f"{(V.readiness_band == 'High').mean():.0%}" if len(V) else "-")
    c[3].metric("Actual placed rate", f"{V.placed.mean():.0%}" if len(V) else "-")
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(V, x="readiness_score", color="readiness_band", nbins=30,
        title="Readiness score distribution",
        color_discrete_map={"High": "#2e9e5b", "Medium": "#e0a800", "Low": "#d64545"}), use_container_width=True)
    g = V.groupby("branch").readiness_score.mean().reset_index()
    b.plotly_chart(px.bar(g, x="branch", y="readiness_score", title="Average readiness by branch"), use_container_width=True)
    gi = gimp()
    st.plotly_chart(px.bar(gi.sort_values("importance"), x="importance", y="feature", orientation="h",
        error_x="std", title="Global feature importance (permutation, AUC drop)"), use_container_width=True)

with t2:
    L, R = st.columns([1, 1.2])
    with L:
        st.subheader("Student profile")
        row = dict(
            branch=st.selectbox("Branch", BRANCHES), college_tier=st.selectbox("College tier", TIERS, index=1),
            cgpa=st.slider("CGPA", 4.5, 10.0, 7.3, .1), tenth_pct=st.slider("10th %", 45.0, 99.0, 80.0),
            twelfth_pct=st.slider("12th %", 45.0, 99.0, 77.0), backlogs=st.number_input("Backlogs", 0, 15, 0),
            internships=st.number_input("Internships", 0, 10, 1), projects=st.number_input("Projects", 0, 10, 2),
            coding_problems_solved=st.number_input("Coding problems solved", 0, 2000, 150),
            certifications=st.number_input("Certifications", 0, 15, 1), hackathons=st.number_input("Hackathons", 0, 10, 0),
            communication_score=st.slider("Communication (1-10)", 1.0, 10.0, 6.0, .1),
            aptitude_score=st.slider("Aptitude (0-100)", 10.0, 100.0, 60.0))
    p, exp = explain_student(pipe, row, ref)
    with R:
        st.subheader("Readiness")
        st.metric("Placement readiness score", f"{p*100:.0f} / 100", band(p))
        top = exp.head(8).sort_values("impact")
        st.plotly_chart(px.bar(top, x="impact", y="feature", orientation="h", color="direction",
            color_discrete_map={"positive": "#2e9e5b", "negative": "#d64545"},
            title="What moves this score vs. an average student (probability points)"), use_container_width=True)
        st.subheader("🔧 What-if")
        w = st.columns(2)
        ch = dict(
            internships=w[0].number_input("Internships →", 0, 10, int(row["internships"]), key="wi1"),
            projects=w[0].number_input("Projects →", 0, 10, int(row["projects"]), key="wi2"),
            certifications=w[0].number_input("Certifications →", 0, 15, int(row["certifications"]), key="wi3"),
            coding_problems_solved=w[1].number_input("Coding problems →", 0, 2000, int(row["coding_problems_solved"]), key="wi4"),
            backlogs=w[1].number_input("Backlogs →", 0, 15, int(row["backlogs"]), key="wi5"),
            cgpa=w[1].slider("CGPA →", 4.5, 10.0, float(row["cgpa"]), .1, key="wi6"))
        before, after = what_if(pipe, row, ch, ref)
        st.metric("Score after changes", f"{after*100:.0f} / 100", f"{(after-before)*100:+.1f} pts")

with t3:
    st.write("Upload a CSV with columns: " + ", ".join(FEATURES) + ". Missing columns/values are handled.")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        try:
            res = predict_batch(pipe, pd.read_csv(up), ref)
            st.dataframe(res, use_container_width=True)
            st.download_button("Download predictions", res.to_csv(index=False), "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not score file: {e}")
    st.download_button("Download sample template", data.head(5)[FEATURES].to_csv(index=False), "template.csv")

with t4:
    summ = json.load(open(ROOT / "reports/summary.json"))
    st.write(f"**Champion:** {summ['champion']} (selected on validation AUC) | Leakage check: {summ['leakage_check']}")
    st.dataframe(pd.read_csv(ROOT / "reports/model_comparison.csv"), use_container_width=True)
    st.image(str(ROOT / "reports/eval_plots.png"))
    st.subheader("Failure analysis (wrong / uncertain predictions)")
    st.dataframe(pd.read_csv(ROOT / "reports/failure_analysis.csv"), use_container_width=True)

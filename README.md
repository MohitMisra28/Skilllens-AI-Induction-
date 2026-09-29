# SkillLens - Placement Readiness Predictor

Predicts a student's placement readiness (0-100 score + High/Medium/Low band) from academics, skills, projects, coding practice and internships, and explains the factors behind each result.

## Setup and run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m skilllens.evaluate      # generates data/students.csv, trains/compares models, writes reports/
streamlit run app.py              # opens http://localhost:8501
python -m pytest -q               # 13 tests
```
Always run `python -m skilllens.evaluate` once after installing: it re-creates `reports/best_model.joblib` with *your* scikit-learn version (a model pickled on another version can fail to load).

**Cost:** everything uses free open-source Python libraries. No paid APIs, GPUs or cloud services. Optional free hosting: Streamlit Community Cloud.

Extra evidence: `python -m skilllens.evidence` (robustness, individual explanations, architecture diagram).

## Project layout
| Path | Purpose |
|---|---|
| `skilllens/data.py` | Synthetic dataset generator (13 features + `placed` target, 5% missing values) |
| `skilllens/features.py` | **Own** cleaning + feature-engineering pipeline |
| `skilllens/evaluate.py` | **Own** model-comparison runner, fixed split, leakage check, failure analysis |
| `skilllens/explain.py` | **Own** global importance, per-student explanation, what-if logic |
| `app.py` | Streamlit dashboard (dashboard, single student + what-if, batch CSV, model evidence) |
| `tests/` | 13 automated tests |
| `reports/` | Metrics, plots, model, failure CSVs |
| `ARCHITECTURE.md`, `FAILURE_LOG.md`, `AI_USAGE.md`, `REPORT.md`, `DEMO_SCRIPT.md` | Documentation |

## Using the app
- **Dashboard:** filter by branch/tier; cards, charts, global importance.
- **Single student:** enter a profile, read the score and top positive/negative factors, then change internships/projects/certs/coding/backlogs/CGPA in the *What-if* panel to see the score move.
- **Batch CSV:** upload a CSV using the columns in the downloadable template. Bad values and missing columns are handled. Download scored results.

## Results (test set, 600 students, same split for all models)
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (baseline, champion) | 0.750 | 0.827 | 0.787 | 0.831 |
| Random Forest | 0.733 | 0.827 | 0.777 | 0.815 |
| Hist. Gradient Boosting | 0.734 | 0.856 | 0.790 | 0.813 |

Data is synthetic; see `FAILURE_LOG.md` for caveats.

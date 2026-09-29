# Compliance Checklist vs. Project Brief (SkillLens)

| Brief requirement | Status | Evidence |
|---|---|---|
| Tabular dataset, clear target | Done (synthetic, `placed` 0/1) | `skilllens/data.py`, `data/students.csv` |
| Repeatable cleaning/preprocessing (missing, categorical, numeric) | Done | `skilllens/features.py` |
| Simple baseline + at least one stronger model | Done (LogReg vs RF, HistGB) | `reports/model_comparison.csv` |
| Fixed train/val/test split, no target leakage | Done | `evaluate.py` `split`, `leakage_check`, tests |
| Readiness score/category + positive/negative factors | Done | App "Single student" tab, `explain.py` |
| Single-student input + CSV batch | Done | `app.py` tabs 2 and 3 |
| Dashboard: cards, charts, filters, what-if | Done | `app.py` tabs 1 and 2 |
| Precision, recall, F1, confusion matrix, ROC-AUC | Done | `reports/model_comparison.csv`, `eval_plots.png` |
| Two+ models on exactly same test set | Done | `evaluate.py` |
| Global importance + several individual explanations | Done | app, `reports/individual_explanations.md/png` |
| Test missing values, unseen categories, noisy inputs | Done | `tests/`, `reports/robustness.csv/png` |
| Analyse >= 20 wrong/uncertain predictions with causes | Done (25 + 48 confident errors) | `FAILURE_LOG.md`, `reports/failure_analysis.csv` |
| Own work: preprocessing, comparison runner, explanation/what-if | Done, written in-house | `features.py`, `evaluate.py`, `explain.py` |
| Working repo with meaningful commit history | Partly: 4+ commits, one per milestone | `git log`. Continue committing small changes yourself |
| README with exact setup/run commands | Done | `README.md` |
| Architecture diagram + design decisions | Done | `ARCHITECTURE.md`, `reports/architecture.png` |
| Automated tests + written failure log | Done | `tests/` (13), `FAILURE_LOG.md` |
| 5-8 minute demo video | **YOU must record** | `DEMO_SCRIPT.md` |
| Concise final report | Done | `REPORT.md` |
| AI_USAGE.md | Done, **edit to match your real usage** | `AI_USAGE.md` |
| Free of cost | Yes | Only open-source libraries; no paid APIs |
| Must understand and explain everything | **YOU** | Study own-work files; see viva questions in `VIVA_PREP.md` |

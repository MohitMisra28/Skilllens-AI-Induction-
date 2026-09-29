# Demo Script (5-8 min) - record with screen capture

1. **(0:00-0:45) Problem and data.** State the goal; show `data.py` briefly; mention 13 features, synthetic, 5% missing, no leakage.
2. **(0:45-2:00) Pipeline.** Show `features.py`: engineered features, imputation, unseen-category handling. Run `python -m pytest -q` and show 12 passing.
3. **(2:00-3:30) Model comparison.** Run `python -m skilllens.evaluate`. Explain split, validation-only tuning, table, confusion matrix and ROC in `reports/eval_plots.png`. Explain why the baseline won.
4. **(3:30-6:00) App.** `streamlit run app.py`. Dashboard filters -> single student score and factor chart -> what-if (add 2 internships, watch score change) -> upload a CSV with a bad value and download results.
5. **(6:00-7:00) Failures.** Open the Model evidence tab and `FAILURE_LOG.md`: borderline vs confident errors, subgroup rates.
6. **(7:00-7:30) Wrap-up.** Limitations (synthetic data), next steps, and how AI was used (`AI_USAGE.md`).

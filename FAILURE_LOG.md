# Failure Log

Model: Logistic Regression (validation-selected champion). Test set: 600 students, **153 errors (25.5%)**:
94 false positives (predicted placed, was not) and 59 false negatives.

## 1. Where errors come from
| Finding | Evidence | Likely cause |
|---|---|---|
| Most errors are near the threshold | 56% of errors have P(placed) within 0.35-0.65 (`reports/failure_analysis.csv` holds the 25 closest) | Genuinely ambiguous profiles; label noise (interviews, luck) is not in the features |
| Confident errors exist | 48 errors with P>=0.7 or <=0.3: 35 false positives, 13 false negatives (`reports/confident_errors.csv`) | Unobserved factors: interview performance, company hiring cycle, referrals |
| Backlog paradox | 6 of the 25 closest cases: high CGPA (>=7.5) but backlogs > 0 | Model penalises backlogs linearly; real recruiters apply a hard cut-off some places and ignore it elsewhere |
| Thin practical profile | 4 cases: CGPA >= 8, no internships, <=1 project | Academics pull the score up; the model cannot see soft signals |
| Imputed key fields | 4 cases with missing CGPA/aptitude/branch | Median/"Unknown" imputation hides real information. Overall error rate is similar with/without missing key fields (26.1% vs 25.4%), so imputation is not the main driver |

## 2. Subgroup error rates (test set)
- By tier: Tier1 15.9%, Tier2 27.7%, Tier3 28.5%. Tier1 is easier because tier is a strong signal; lower tiers depend more on unmeasured factors.
- By branch: CSE 22.4% and Civil 22.4% (lowest), ECE 28.1% (highest).

## 3. Honest caveats
- The dataset is **synthetic** (`skilllens/data.py`), generated from a noisy logistic process. Absolute metrics (ROC-AUC ~0.83) describe this data, not real campuses.
- Because the generator is close to linear, the stronger models (Random Forest, Gradient Boosting) did not beat the baseline. On real data with interactions they likely would.
- The 25 saved "failure" rows are the closest-to-threshold errors; use `confident_errors.csv` for the harder cases.

## 4. Improvements to try
1. Threshold tuning (recall-weighted) instead of a fixed 0.5.
2. Add interview-round / communication-assessment features.
3. Calibrate probabilities (isotonic) before showing scores.
4. Replace with real institutional data and re-run `python -m skilllens.evaluate`.

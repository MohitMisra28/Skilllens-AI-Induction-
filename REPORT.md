# SkillLens - Final Report

**Goal.** Estimate placement readiness from tabular student data and show the factors behind each score.

**Data.** 3,000 synthetic students, 13 features (academics, backlogs, internships, projects, coding, certifications, hackathons, communication, aptitude, branch, tier), 5% injected missing values, 56.8% placed.

**Method.** Custom pipeline (feature engineering, median/"Unknown" imputation, scaling, one-hot). Fixed stratified 60/20/20 split (1800/600/600). Logistic Regression baseline vs Random Forest vs Gradient Boosting; hyper-parameters tuned on validation only.

**Results (test).** Baseline F1 0.787 / ROC-AUC 0.831; RF 0.777 / 0.815; GB 0.790 / 0.813. Confusion matrix (champion): TN 165, FP 94, FN 59, TP 282. The baseline was selected on validation AUC; the ensembles did not add value because the generating process is near-linear plus noise.

**Explainability.** Global permutation importance ranks CGPA (0.071), backlogs (0.064), internships (0.026), college tier (0.023), projects (0.019) highest. Per-student explanations use occlusion, giving signed factors; what-if re-scores edited profiles.

**Robustness.** Tests cover all-missing rows, missing columns, unseen categories, strings/negatives/extreme values, and target leakage (max feature-target correlation 0.365, zero split overlap).

**Failures.** 153/600 test errors (25.5%); 56% are borderline, 48 are confident errors driven by unobserved factors. See `FAILURE_LOG.md`.

**Limitations.** Synthetic data; performance will differ on real campuses. Predictions are advisory and should not gate opportunities for students.

**Next steps.** Real data, threshold tuning, probability calibration, interview-round features.

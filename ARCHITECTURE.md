# Architecture and Design Decisions

```mermaid
flowchart LR
  A[data.py: generate 3000 students] --> B[students.csv]
  B --> C[Split 60/20/20 stratified, seed 42]
  C --> D[Leakage checks]
  C --> E[features.py: FeatureEngineer -> impute -> scale / one-hot]
  E --> F[Train 3 models, tune on VALIDATION]
  F --> G[Score all on same TEST set]
  G --> H[Champion by validation AUC]
  H --> I[(reports/: model, metrics, plots, failure CSVs)]
  I --> J[app.py Streamlit]
  J --> K[explain.py: permutation + occlusion + what-if]
  J --> L[Batch CSV scoring]
```

## Key decisions
1. **Preprocessing is inside the sklearn Pipeline** so training and serving apply identical steps, and no test statistics leak into imputation/scaling.
2. **Robust input handling:** `FeatureEngineer` coerces strings to NaN, clips out-of-range values and adds missing columns; `OneHotEncoder(handle_unknown="ignore")` handles unseen categories. The app cannot crash on messy CSVs.
3. **Champion chosen on validation AUC, never on test**, and all models are scored on the identical test set.
4. **Model-agnostic explanations:** occlusion (swap a feature for the cohort median/mode and measure the probability change) works for any model and yields signed, human-readable factors. Permutation importance gives the global view.
5. **No leakage by construction:** no outcome-derived fields (salary, company); automated check on target-in-features, split overlap and suspiciously high correlation.
6. **Sensitive attributes excluded:** no gender or religion features, to avoid a tool that discriminates.

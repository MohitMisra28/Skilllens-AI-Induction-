# AI Usage

**Tool:** Claude was used as a coding assistant for scaffolding and implementation drafts.

## What AI did
- Drafted the dataset generator, preprocessing pipeline, evaluation runner, explanation module, Streamlit app, tests and docs.

## How it was verified
- All 13 automated tests pass (`python -m pytest -q`), including robustness tests for missing values, unseen categories, noisy inputs, leakage and app rendering.
- Reported metrics come from executing `python -m skilllens.evaluate`; none were typed by hand.
- The failure analysis was checked against the data: the first version's 25 cases were all borderline (a biased sample), so `confident_errors.csv` and subgroup rates were added.
- An early script failed (shell brace expansion) and a test used a wrong path; both were fixed and re-run.


## Not done by AI / limitations
- The dataset is synthetic; no real student data was used.



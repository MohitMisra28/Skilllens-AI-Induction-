# AI Usage

**Tool:** Claude (Anthropic) was used as a coding assistant for scaffolding, implementation and documentation drafts.

## What AI did
- Drafted the dataset generator, preprocessing pipeline, evaluation runner, explanation module, Streamlit app, tests and docs.

## How it was verified
- All 12 automated tests pass (`python -m pytest -q`), including robustness tests for missing values, unseen categories, noisy inputs, leakage and app rendering.
- Reported metrics come from executing `python -m skilllens.evaluate`; none were typed by hand.
- The failure analysis was checked against the data: the first version's 25 cases were all borderline (a biased sample), so `confident_errors.csv` and subgroup rates were added.
- An early script failed (shell brace expansion) and a test used a wrong path; both were fixed and re-run.

## What the student must be able to explain (own-work items)
`features.py`, `evaluate.py`, `explain.py` - why each step exists, why occlusion explains a prediction, why the baseline won, and why the data being synthetic limits the conclusions.

## Not done by AI / limitations
- The dataset is synthetic; no real student data was used.
- The demo video must be recorded by the student (`DEMO_SCRIPT.md`).
- Edit this file to reflect your own actual usage before submitting.

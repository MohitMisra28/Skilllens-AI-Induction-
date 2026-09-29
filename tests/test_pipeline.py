import numpy as np, pytest
from skilllens.data import generate_dataset
from skilllens.features import FEATURES, build_preprocessor
from skilllens.evaluate import split, leakage_check

@pytest.fixture(scope="module")
def df(): return generate_dataset(800)

@pytest.fixture(scope="module")
def pre(df): return build_preprocessor().fit(df[FEATURES])

def test_no_leakage(df):
    tr, va, te = split(df); leakage_check(df, tr, va, te)

def test_split_is_deterministic(df):
    assert split(df)[2].index.equals(split(df)[2].index)

def test_all_missing_row_does_not_crash(pre, df):
    X = df[FEATURES].head(3).copy(); X.iloc[0] = np.nan
    assert np.isfinite(pre.transform(X)).all()

def test_unseen_category_ok(pre, df):
    X = df[FEATURES].head(3).copy(); X["branch"] = "Astrophysics"; X["college_tier"] = "Tier9"
    assert pre.transform(X).shape[0] == 3

def test_noisy_inputs_ok(pre, df):
    X = df[FEATURES].head(4).copy().astype({"cgpa": object})
    X.loc[X.index[0], "cgpa"] = "abc"; X.loc[X.index[1], "cgpa"] = 99
    X.loc[X.index[2], "backlogs"] = -5; X.loc[X.index[3], "coding_problems_solved"] = 1e9
    assert np.isfinite(pre.transform(X)).all()

def test_missing_columns_ok(pre, df):
    assert pre.transform(df[["cgpa"]].head(2)).shape[0] == 2

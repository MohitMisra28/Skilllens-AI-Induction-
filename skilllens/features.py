"""Own preprocessing + feature-engineering pipeline."""
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM = ["cgpa", "tenth_pct", "twelfth_pct", "backlogs", "internships", "projects",
       "coding_problems_solved", "certifications", "hackathons",
       "communication_score", "aptitude_score"]
CAT = ["branch", "college_tier"]
ENGINEERED = ["academic_avg", "experience_score", "has_backlog", "coding_log"]
FEATURES = NUM + CAT

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds derived features. Coerces bad inputs (strings, negatives, out of range) to NaN
    and clips ranges so noisy input never crashes the pipeline."""
    RANGES = {"cgpa": (0, 10), "tenth_pct": (0, 100), "twelfth_pct": (0, 100),
              "communication_score": (0, 10), "aptitude_score": (0, 100)}
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in FEATURES:
            if c not in X: X[c] = np.nan
        for c in NUM:
            X[c] = pd.to_numeric(X[c], errors="coerce").clip(lower=0)
        for c, (lo, hi) in self.RANGES.items():
            X[c] = X[c].clip(lo, hi)
        X["academic_avg"] = (X.cgpa * 10 + X.tenth_pct + X.twelfth_pct) / 3
        X["experience_score"] = 2 * X.internships + X.projects + 0.5 * X.hackathons
        X["has_backlog"] = (X.backlogs > 0).astype(float).where(X.backlogs.notna())
        X["coding_log"] = np.log1p(X.coding_problems_solved)
        return X[NUM + ENGINEERED + CAT]

def build_preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    ct = ColumnTransformer([("num", num, NUM + ENGINEERED), ("cat", cat, CAT)])
    return Pipeline([("fe", FeatureEngineer()), ("ct", ct)])

def feature_names(pre):
    return list(pre.named_steps["ct"].get_feature_names_out())

"""Synthetic-but-realistic placement dataset. Placement is driven by a latent score
built from student attributes + noise. NO outcome-derived column (salary, offer company)
is included, which prevents target leakage."""
import numpy as np, pandas as pd

BRANCHES = ["CSE", "IT", "ECE", "EEE", "Mechanical", "Civil"]
TIERS = ["Tier1", "Tier2", "Tier3"]

def generate_dataset(n=3000, seed=42, missing_rate=0.05):
    r = np.random.default_rng(seed)
    branch = r.choice(BRANCHES, n, p=[.28, .17, .2, .12, .13, .10])
    tier = r.choice(TIERS, n, p=[.2, .4, .4])
    cgpa = np.clip(r.normal(7.3, 1.0, n), 4.5, 10)
    tenth = np.clip(r.normal(80, 9, n), 45, 99)
    twelfth = np.clip(r.normal(77, 10, n), 45, 99)
    backlogs = r.poisson(np.where(cgpa < 6.5, 1.5, 0.3))
    internships = r.poisson(np.where(tier == "Tier1", 1.6, 0.8))
    projects = np.clip(r.poisson(2.2, n), 0, 8)
    coding = np.clip(r.gamma(2, 60, n) * np.where(np.isin(branch, ["CSE", "IT"]), 1, 0.4), 0, 900).round()
    certs = r.poisson(1.5, n)
    hackathons = r.poisson(0.7, n)
    comm = np.clip(r.normal(6, 1.8, n), 1, 10)
    aptitude = np.clip(0.5 * cgpa * 10 + r.normal(0, 12, n) + 15, 10, 100)
    z = (0.9 * (cgpa - 7) + 0.03 * (aptitude - 60) + 0.45 * internships + 0.25 * projects
         + 0.004 * coding + 0.12 * certs + 0.2 * hackathons + 0.2 * (comm - 6)
         - 0.7 * backlogs + 0.008 * (tenth - 80)
         + np.select([tier == "Tier1", tier == "Tier2"], [0.8, 0.2], -0.5)
         + np.isin(branch, ["CSE", "IT"]) * 0.5 - np.isin(branch, ["Civil"]) * 0.4
         - 1.0 + r.normal(0, 1.1, n))            # irreducible noise
    placed = (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    df = pd.DataFrame(dict(branch=branch, college_tier=tier, cgpa=cgpa.round(2),
        tenth_pct=tenth.round(1), twelfth_pct=twelfth.round(1), backlogs=backlogs,
        internships=internships, projects=projects, coding_problems_solved=coding,
        certifications=certs, hackathons=hackathons, communication_score=comm.round(1),
        aptitude_score=aptitude.round(1), placed=placed))
    for c in ["cgpa", "tenth_pct", "twelfth_pct", "aptitude_score", "communication_score", "branch"]:
        df.loc[r.random(n) < missing_rate, c] = np.nan
    return df

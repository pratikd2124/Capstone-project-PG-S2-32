from pathlib import Path
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
import pymc as pm

ROOT = Path(__file__).parent
DATA, REF, OUT = ROOT/"DATA", ROOT/"REFERENCE", ROOT/"OUTPUTS"
QUESTION = "ry9"
MATRIX_SAMPLE_SIZE = 100   # replace with real test-retest n
MIN_EFFECT = 0.10          # replace with agreed threshold

def counts(file):
    s = pd.to_numeric(pd.read_csv(file)[QUESTION], errors="coerce")
    return np.array([(s == k).sum() for k in range(1,5)], dtype=int)

x, y = counts(DATA/"RY25_PreMay10.csv"), counts(DATA/"RY25_PostMay10.csv")
with open(REF/"unreliability-matrices.json") as f:
    P0 = np.array(json.load(f)[QUESTION], dtype=float)
B = np.loadtxt(REF/"B_matrix.txt")

with pm.Model() as model:
    P = pm.Dirichlet("P", a=P0*MATRIX_SAMPLE_SIZE + 1, shape=(4,4))
    pre = pm.Dirichlet("pre", a=np.ones(4))
    post = pm.Dirichlet("post", a=np.ones(4))
    pm.Multinomial("pre_obs", n=x.sum(), p=pm.math.dot(pre, P), observed=x)
    pm.Multinomial("post_obs", n=y.sum(), p=pm.math.dot(post, P), observed=y)
    approx = pm.fit(5000, method="advi")
    trace = approx.sample(2000)

pre_s = trace.posterior["pre"].values.reshape(-1,4)
post_s = trace.posterior["post"].values.reshape(-1,4)
effect = np.linalg.norm((pre_s-post_s) @ B.T, axis=1)
prob = np.mean(effect > MIN_EFFECT)

pd.DataFrame({"effect": effect}).to_csv(OUT/"04_bayesian_effect_samples.csv", index=False)
plt.hist(effect, bins=30); plt.axvline(MIN_EFFECT, ls="--", label="Threshold")
plt.xlabel("Projected effect"); plt.ylabel("Posterior samples")
plt.title(f"P(effect > threshold) = {prob:.3f}"); plt.legend(); plt.tight_layout()
plt.savefig(OUT/"04_bayesian_effect.png", dpi=180)
print(f"Posterior probability = {prob:.3f}")

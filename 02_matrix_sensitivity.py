from pathlib import Path
import json, numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
OUT = ROOT / "OUTPUTS"; OUT.mkdir(exist_ok=True)
QUESTION = "ry9"
N_SIMS = 300
MATRIX_SAMPLE_SIZE = 100   # replace when true test-retest n is known
rng = np.random.default_rng(42)

with open(ROOT / "REFERENCE/unreliability-matrices.json") as f:
    P0 = np.array(json.load(f)[QUESTION], dtype=float)

changes = []
for i in range(N_SIMS):
    P = np.vstack([rng.dirichlet(row*MATRIX_SAMPLE_SIZE + 1) for row in P0])
    changes.append([i, np.mean(abs(P-P0)), np.max(abs(P-P0))])

df = pd.DataFrame(changes, columns=["simulation","mean_change","max_change"])
df.to_csv(OUT / "02_matrix_sensitivity.csv", index=False)
plt.hist(df.mean_change, bins=25)
plt.xlabel("Mean absolute matrix change"); plt.ylabel("Simulations")
plt.title("Unreliability-matrix sensitivity"); plt.tight_layout()
plt.savefig(OUT / "02_matrix_uncertainty.png", dpi=180)
print(df[["mean_change","max_change"]].describe())

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA = Path(__file__).parent / "Data"
OUT = Path(__file__).parent / "OUTPUTS"
OUT.mkdir(exist_ok=True)
QUESTION = "ry9"

pre = pd.to_numeric(pd.read_csv(DATA / "RY25_PreMay10.csv")[QUESTION], errors="coerce")
post = pd.to_numeric(pd.read_csv(DATA / "RY25_PostMay10.csv")[QUESTION], errors="coerce")

pre_n = [(pre == k).sum() for k in range(1, 5)]
post_n = [(post == k).sum() for k in range(1, 5)]
result = pd.DataFrame({"response": [1,2,3,4], "pre": pre_n, "post": post_n})
result.to_csv(OUT / "01_counts.csv", index=False)

p1 = result.pre / result.pre.sum()
p2 = result.post / result.post.sum()
x = [1,2,3,4]
plt.bar([v-0.18 for v in x], p1, 0.36, label="Pre")
plt.bar([v+0.18 for v in x], p2, 0.36, label="Post")
plt.xticks(x); plt.xlabel("Response"); plt.ylabel("Proportion")
plt.title(f"Pre vs Post: {QUESTION}"); plt.legend(); plt.tight_layout()
plt.savefig(OUT / "01_response_distribution.png", dpi=180)
print(result)

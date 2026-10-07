from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
OUT = ROOT / "OUTPUTS"
MIN_EFFECT = 0.10
BAYES_CUTOFF = 0.95

bayes = pd.read_csv(OUT/"04_bayesian_effect_samples.csv")
bayes_prob = (bayes.effect > MIN_EFFECT).mean()

summary = pd.DataFrame({
    "method": ["Bayesian VI"],
    "score": [bayes_prob],
    "decision": [bayes_prob >= BAYES_CUTOFF]
})
summary.to_csv(OUT/"05_method_comparison.csv", index=False)
plt.bar(summary.method, summary.score); plt.ylim(0,1)
plt.ylabel("Probability effect exceeds threshold")
plt.title("Bayesian minimal-effect result"); plt.tight_layout()
plt.savefig(OUT/"05_method_comparison.png", dpi=180)
print(summary)
print("Add the verified existing MET decision here for final agreement comparison.")

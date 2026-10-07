from pathlib import Path
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
import onnxruntime as ort

ROOT = Path(__file__).parent
DATA, REF, OUT = ROOT/"Data", ROOT/"REFERENCE", ROOT/"OUTPUTS"
QUESTION, N_SIMS, MATRIX_SAMPLE_SIZE = "ry9", 300, 100
rng = np.random.default_rng(42)

def counts(file):
    s = pd.to_numeric(pd.read_csv(file)[QUESTION], errors="coerce")
    return np.array([(s == k).sum() for k in range(1,5)], dtype=float)

def moments(p):
    v = np.arange(1,5); mean = v @ p
    sd = np.sqrt(((v-mean)**2) @ p)
    return [mean, sd, ((v-mean)**3 @ p)/sd**3, ((v-mean)**4 @ p)/sd**4]

def asymmetry(P):
    pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    return np.mean([(P[i,j]-P[j,i])**2 for i,j in pairs])

def make_features(x, y, P):
    X, Y = x/x.sum(), y/y.sum()
    return np.array(
        list(X) + list(Y) + list(abs(X-Y)) +
        moments(X) + moments(Y) +
        [round(np.linalg.norm(X-Y),4), x.sum(), y.sum()] +
        list(P.flatten()) + [asymmetry(P)], dtype=np.float32)

x, y = counts(DATA/"RY25_PreMay10.csv"), counts(DATA/"RY25_PostMay10.csv")
with open(REF/"unreliability-matrices.json") as f:
    P0 = np.array(json.load(f)[QUESTION], dtype=float)
with open(REF/"EXP002.json") as f:
    meta = json.load(f)

mm = pd.read_json(meta["min_max_df"])
mins = dict(zip(mm.column, mm["min"])); maxs = dict(zip(mm.column, mm["max"]))
labels = meta["feature_labels"]
session = ort.InferenceSession(str(REF/"EXP002_nn_optimal_epoch.onnx"))
input_name = session.get_inputs()[0].name

def predict(P):
    f = make_features(x, y, P)
    z = np.array([(v-mins[k])/(maxs[k]-mins[k]) for v,k in zip(f, labels)], dtype=np.float32)[None,:]
    pred = np.asarray(session.run(None, {input_name:z})[0]).reshape(-1)[:2]
    shape = pred[0]*(maxs["shape"]-mins["shape"]) + mins["shape"]
    rate = pred[1]*(maxs["rate"]-mins["rate"]) + mins["rate"]
    return shape, rate

rows = []
for i in range(N_SIMS):
    P = np.vstack([rng.dirichlet(row*MATRIX_SAMPLE_SIZE + 1) for row in P0])
    shape, rate = predict(P)
    rows.append([i, shape, rate])

df = pd.DataFrame(rows, columns=["simulation","shape","rate"])
df.to_csv(OUT/"03_network_uncertainty.csv", index=False)

for col in ["shape", "rate"]:
    lo, mid, hi = df[col].quantile([.025,.5,.975])
    plt.figure(); plt.hist(df[col], bins=25)
    plt.axvline(lo, ls="--"); plt.axvline(mid); plt.axvline(hi, ls="--")
    plt.xlabel(col); plt.ylabel("Simulations")
    plt.title(f"{col}: median + 95% uncertainty interval")
    plt.tight_layout(); plt.savefig(OUT/f"03_{col}_interval.png", dpi=180); plt.close()
    print(f"{col}: median={mid:.4f}, 95% interval=({lo:.4f}, {hi:.4f})")

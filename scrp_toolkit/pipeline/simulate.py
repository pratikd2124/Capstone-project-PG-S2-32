"""RYA's d_hat statistic and its simulator, in vectorised Python.

A line-by-line port of ``d_hat_simulation_from_LL.R`` (RYA, final_code):

* a group has a fixed composition of latent types (how many students "really" belong to
  answer 1..4);
* on each trial, every student of type j answers according to row j of rho;
* d_hat is RYA's ``d_hat_calc`` with the orthogonal projection B and ``scale_true = TRUE``,
  i.e. multiplied by 0.5 * (1/n + 1/m).

The network EXP002 was trained to predict the Gamma fitted to these draws, with its rate
used as-is (Murat, 12 Oct: simulated mean / network mean = 1.003). So the draws here are on
the same scale as the network's Gamma, and either can serve as the null distribution.

This is also the "faithful latent-type simulator" Objective 2 needs: Shuyun's simulator uses
observed answers as latent types, which is exactly what the network is given as X and Y, so
conditional on the composition the two are the same model.
"""
from __future__ import annotations

import numpy as np

# RYA's B (B_matrix.txt): GramSchmidt rows of the sum-to-zero space, 3 x 4.
B = np.array([
    [-0.866025403784439, 0.288675134594813, 0.288675134594813, 0.288675134594813],
    [0.0, -0.816496580927726, 0.408248290463863, 0.408248290463863],
    [0.0, 2.35513868802566e-16, -0.707106781186547, 0.707106781186547],
])


def sigma_stack(P: np.ndarray) -> np.ndarray:
    """RYA's sigma_maker for j = 1..4: Sigma_j = diag(rho_j) - rho_j rho_j^T. Shape (4, 4, 4)."""
    P = np.asarray(P, dtype=float)
    return np.stack([np.diag(r) - np.outer(r, r) for r in P])


def _quad_form(diff: np.ndarray, sig_hat: np.ndarray) -> np.ndarray:
    """(B diff)^T (B sig B^T)^-1 (B diff), for a stack of trials."""
    v = diff @ B.T                                   # (T, 3)
    inner = np.einsum("ab,tbc,dc->tad", B, sig_hat, B)  # (T, 3, 3)
    try:
        sol = np.linalg.solve(inner, v[..., None])[..., 0]
    except np.linalg.LinAlgError:
        sol = np.einsum("tab,tb->ta", np.linalg.pinv(inner), v)
    return np.einsum("ta,ta->t", v, sol)


def d_hat_scaled(X: np.ndarray, Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    """RYA's scaled d_hat for answer counts X (T, 4) and Y (T, 4). n, m = row sums."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    n = X.sum(axis=1, keepdims=True)
    m = Y.sum(axis=1, keepdims=True)
    w = X / n**2 + Y / m**2                          # (T, 4)
    sig_hat = np.einsum("tj,jab->tab", w, sigma_stack(P))
    d = _quad_form(X / n - Y / m, sig_hat)
    return 0.5 * (1.0 / n[:, 0] + 1.0 / m[:, 0]) * d


def d_norm(p: np.ndarray, q: np.ndarray, n: float, m: float, P: np.ndarray) -> float:
    """The same statistic evaluated at proportions p, q with group sizes n, m.

    With p = X/n and q = Y/m this equals ``d_hat_scaled`` on the counts; it is used both for
    the observed statistic and to find the boundary pair.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(d_hat_scaled((p * n)[None, :], (q * m)[None, :], P)[0])


def integer_composition(props: np.ndarray, n: int) -> np.ndarray:
    """Turn proportions into whole-student type counts summing to n (largest remainder)."""
    props = np.clip(np.asarray(props, dtype=float), 0, None)
    props = props / props.sum()
    raw = props * n
    base = np.floor(raw).astype(int)
    short = int(n - base.sum())
    if short > 0:
        base[np.argsort(-(raw - base))[:short]] += 1
    return base


def simulate_d_hat(type_x: np.ndarray, type_y: np.ndarray, P: np.ndarray, trials: int = 5000,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Draws of scaled d_hat for groups with latent-type counts type_x, type_y (RYA's trial loop)."""
    rng = rng or np.random.default_rng(2026)
    P = np.asarray(P, dtype=float)
    P = P / P.sum(axis=1, keepdims=True)             # rmultinom needs rows that sum to 1
    type_x = np.asarray(type_x, dtype=int)
    type_y = np.asarray(type_y, dtype=int)
    X = np.zeros((trials, 4))
    Y = np.zeros((trials, 4))
    for j in range(4):
        if type_x[j]:
            X += rng.multinomial(int(type_x[j]), P[j], size=trials)
        if type_y[j]:
            Y += rng.multinomial(int(type_y[j]), P[j], size=trials)
    out = d_hat_scaled(X, Y, P)
    return out[np.isfinite(out)]

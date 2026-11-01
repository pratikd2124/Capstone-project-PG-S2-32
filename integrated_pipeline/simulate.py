"""RYA's distance d_hat and RYA's simulator, rewritten in Python (d_hat_simulation_from_LL.R).

How the simulator works: each group has a fixed make-up of students by their "real"
answer. On each run, every student answers through their row of rho, so some give a
different answer than their real one. d_hat is then worked out exactly as RYA's
d_hat_calc does (with the projection B, scaled by (1/n + 1/m) / 2). Repeating this
thousands of times shows how much d_hat varies by chance.

The network EXP002 was trained to copy a Gamma fitted to exactly these draws, so the two
are on the same scale (checked on the ry9 example: simulated mean 0.4756, network 0.4765).
"""
from __future__ import annotations

import numpy as np

# RYA's B (B_matrix.txt): GramSchmidt rows of the sum-to-zero space, 3 x 4.
B = np.array([
    [-0.866025403784439, 0.288675134594813, 0.288675134594813, 0.288675134594813],
    [0.0, -0.816496580927726, 0.408248290463863, 0.408248290463863],
    [0.0, 2.35513868802566e-16, -0.707106781186547, 0.707106781186547],
])


_SIGMA_CACHE: dict = {}
_SIGMA_CACHE_MAX = 4096


def sigma_stack(P: np.ndarray) -> np.ndarray:
    """RYA's sigma_maker for j = 1..4: Sigma_j = diag(rho_j) - rho_j rho_j^T. Shape (4, 4, 4).

    O4: the boundary search evaluates d_hat thousands of times for the same rho, so the stack
    is built once per rho and reused (same numbers; returned read-only so it can't be changed).
    """
    P = np.asarray(P, dtype=float)
    key = P.tobytes()
    stack = _SIGMA_CACHE.get(key)
    if stack is None:
        stack = np.stack([np.diag(r) - np.outer(r, r) for r in P])
        stack.setflags(write=False)
        if len(_SIGMA_CACHE) >= _SIGMA_CACHE_MAX:
            _SIGMA_CACHE.clear()
        _SIGMA_CACHE[key] = stack
    return stack


def _quad_form(diff: np.ndarray, sig_hat: np.ndarray) -> np.ndarray:
    """(B diff)^T (B sig B^T)^-1 (B diff), for a stack of trials."""
    v = diff @ B.T                                   # (T, 3)
    inner = np.einsum("ab,tbc,dc->tad", B, sig_hat, B)  # (T, 3, 3)
    try:
        sol = np.linalg.solve(inner, v[..., None])[..., 0]
    except np.linalg.LinAlgError:
        sol = np.einsum("tab,tb->ta", np.linalg.pinv(inner), v)
    return np.einsum("ta,ta->t", v, sol)


_CLOSED_FORM_MIN_ROWS = 64     # below this the general path is faster (the boundary search: 1 row)


def _quad_form_closed(diff: np.ndarray, w: np.ndarray, P: np.ndarray) -> np.ndarray:
    """The same quadratic form as ``_quad_form``, for many trials at once (O4, Murat).

    B sig_hat B^T = sum_j w_j (B Sigma_j B^T), so the four 3x3 blocks are built once per rho and
    each trial's 3x3 matrix is a weighted sum of them; it is then inverted in closed form
    (adjugate / determinant) instead of one LAPACK solve per trial. About 6x faster on the
    simulator's 4,000 trials. Same maths; results agree to ~1e-15 relative. Rows where the
    3x3 matrix is (nearly) singular are left NaN for the caller to redo the general way.
    """
    v = diff @ B.T                                                      # (T, 3)
    blocks = np.einsum("ab,jbc,dc->jad", B, sigma_stack(P), B).reshape(4, 9)
    a, b, c, d, e, f, g, h, i = (w @ blocks).T                          # each (T,)
    co_a, co_b, co_c = e * i - f * h, f * g - d * i, d * h - e * g      # cofactors of the first row
    det = a * co_a + b * co_b + c * co_c
    inv = np.stack([co_a, c * h - b * i, b * f - c * e,
                    co_b, a * i - c * g, c * d - a * f,
                    co_c, b * g - a * h, a * e - b * d], axis=1).reshape(-1, 3, 3)
    scale = np.abs(np.stack([a, b, c, d, e, f, g, h, i])).max(axis=0)
    ok = np.abs(det) > 1e-12 * scale ** 3
    sol = np.einsum("tab,tb->ta", inv, v) / np.where(ok, det, np.nan)[:, None]
    return np.einsum("ta,ta->t", v, sol)


def d_hat_scaled(X: np.ndarray, Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    """RYA's scaled d_hat for answer counts X (T, 4) and Y (T, 4). n, m = row sums."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    n = X.sum(axis=1, keepdims=True)
    m = Y.sum(axis=1, keepdims=True)
    w = X / n**2 + Y / m**2                          # (T, 4)
    diff = X / n - Y / m
    if len(X) >= _CLOSED_FORM_MIN_ROWS:
        d = _quad_form_closed(diff, w, P)
        redo = ~np.isfinite(d)
        if redo.any():                               # (near-)singular rows: the general way
            d[redo] = _quad_form(diff[redo], np.einsum("tj,jab->tab", w[redo], sigma_stack(P)))
    else:
        sig_hat = np.einsum("tj,jab->tab", w, sigma_stack(P))
        d = _quad_form(diff, sig_hat)
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

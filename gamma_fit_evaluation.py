
import numpy as np
from scipy.stats import gamma


def evaluate_gamma_fit(draws, seed=2026, train_fraction=0.7):

    values = np.asarray(draws, dtype=float).reshape(-1)

    if values.size < 200:
        raise ValueError("Fewer than 200 draws")

    if not np.isfinite(values).all():
        raise ValueError("Invalid simulated values")

    if (values < 0).any():
        raise ValueError("Negative simulated statistic")

    zeros = float(np.mean(values == 0))

    if zeros:
        raise ValueError(
            f"Exact-zero fraction {zeros:.4%}"
        )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(values)

    split = int(train_fraction * len(values))
    train = shuffled[:split]
    test = np.sort(shuffled[split:])

    shape, loc, scale = gamma.fit(train, floc=0)

    empirical_q95 = float(np.quantile(test, 0.95))
    gamma_q95 = float(
        gamma.ppf(0.95, shape, loc=0, scale=scale)
    )

    cdf = gamma.cdf(test, shape, loc=0, scale=scale)
    n = len(test)

    ks_stat = max(
        np.max(np.abs(cdf - np.arange(n) / n)),
        np.max(np.abs(cdf - np.arange(1, n + 1) / n))
    )

    empirical_tail = float(np.mean(test > empirical_q95))
    gamma_tail = float(
        gamma.sf(empirical_q95, shape, loc=0, scale=scale)
    )

    return {
        "shape": float(shape),
        "rate": float(1 / scale),
        "ks_stat": float(ks_stat),
        "empirical_q95": empirical_q95,
        "gamma_q95": gamma_q95,
        "q95_error": abs(gamma_q95 - empirical_q95)
                     / max(empirical_q95, 1e-12),
        "tail_error": abs(empirical_tail - gamma_tail),
        "zero_fraction": zeros,
        "n_train": len(train),
        "n_test": len(test),
    }

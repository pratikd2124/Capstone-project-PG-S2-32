"""User-adjustable settings for the GemFinder final pipeline V2."""

RANDOM_SEED = 42
POSTERIOR_DRAWS = 20_000
DIRICHLET_PRIOR_ALPHA = 0.5
MIN_GROUP_SIZE = 5

# Intentionally None until an externally justified GemFinder minimal-effect
# boundary is supplied. Never replace this with an exploratory value.
OFFICIAL_EPSILON = None

DIAGNOSTIC_QUANTILES = [0.25, 0.50, 0.75, 0.90]

# Historical comparison defaults. These are genuine grade subgroup examples
# already used successfully in the project, not whole-file Pre-vs-Post tests.
DEFAULT_DATASET = "post"
DEFAULT_GROUP_COLUMN = "grade"
DEFAULT_GROUP_PAIRS = [(4, 7), (11, 6), (12, 5)]

# Historical rho matrices are rounded to two/three decimals. Row sums such as
# 0.99 and 1.01 are accepted and normalised exactly once during preprocessing.
RHO_ROW_SUM_TOLERANCE = 0.03
RHO_PERTURBATIONS = [-0.10, -0.05, -0.01, 0.0, 0.01, 0.05, 0.10]
SAMPLE_SIZE_GRID = [20, 50, 100, 250, 500, 1000, 2500, 5000, 10_000]

# Keep sensitivity figures simple by using one representative comparison.
PREFERRED_SENSITIVITY_ITEM = "ry9"

# If genuine test-retest transition counts become available, this many rho
# posterior draws may additionally be propagated through EXP002. The full
# d_hat posterior still uses POSTERIOR_DRAWS unless overridden at runtime.
RHO_NETWORK_DRAWS = 200

# Final plot readability limits.
MAX_BAYES_FIGURE_COMPARISONS = 15
MAX_HEATMAP_COMPARISONS = 15

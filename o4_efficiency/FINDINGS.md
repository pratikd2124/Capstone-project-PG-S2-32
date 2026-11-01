# O4 findings: efficiency and cost optimisation (Murat)

What was measured, what was changed, and what was tried and dropped. Timings are from an Apple M2 laptop; other machines give other numbers, but the ratios should hold. Commands to re-run each part are given in brackets.

## Where the time went (single model)

`python -m o4_efficiency.cli profile`

- One `predict()` took about 517 µs: preparing the 40 inputs with pandas 76%, scaling 8%, the network 16%. The pandas cost comes from building and joining 16 small Series per call.
- 99.1% of EXP002's 1,261,058 parameters are in its 19 hidden 256×256 layers. The weights take 4.8 MB; each row needs about 2 KB of working memory. Memory is not a constraint.

## Batching (kept)

`EfficientPredictor.predict_batch` in `batched.py` prepares the inputs for many items at once with numpy and runs the network on up to 4,096 rows per call. RYA's model file is unchanged; only its batch dimension is opened, in memory.

| Workload | Before | Now |
|---|---|---|
| One item | 0.53 ms | 0.30 ms |
| 40-item school report | 21 ms | 0.8 ms (about 27×) |
| 200,000 predictions in one run | 105 s | 1.6 s (about 64×) |

The gain grows with the number of items per call. The old code was already fast for a single user, so batching mainly helps bulk work: sensitivity runs, benchmarks, tests.

Same answers as `predict`, NaN cases included. On the M2 it is bit-identical on 20,000 random items, on all 63,328 raw training rows turned back into counts, and on the 3 RY25 schools in both surveys. On other CPUs a few inputs can differ in the last bit (up to 9e-16, seen on GitHub's Linux runners), because BLAS adds the terms in another order; the model outputs still matched. RYA's own `np.dot` code varies across machines in the same way.

## fp16 weights (kept, opt-in)

`EfficientPredictor(paths, weights="fp16")`; `python -m o4_efficiency.cli export-model` writes the browser file.

- The weights are stored as 16-bit and turned back into 32-bit inside the model. The file halves (4.8 MB → 2.4 MB) and the maths still runs in 32-bit, so speed is unchanged. The benefit is a smaller download in the browser.
- Outputs move by about 1e-4 (median), at most about 1% on 2,000 random items.
- **Decisions through the pipeline's boundary test** (`pipeline-cost --equivalence-cases 3000`, 1 Nov): on 3,000 benchmark comparisons, fp16 changed 2 decisions (0.07%). On the network route it changed 2 of 2,302 (0.09%); the fallback route doesn't use the network, so nothing changed there. Yardstick: Pratik's rho range (±5% error mass) says 2.17% of network-route decisions could flip anyway. fp16's effect is about 1/24 of that.
- Earlier check with a simplified test (epsilon = 0, 12 Oct, 40,000 comparisons): 0.16% of decisions changed, against 0.32% from just fixing the 2-decimal rounding in the matrices.
- The default stays `"fp32"`, RYA's model exactly. Whether fp16 is used is a team decision about the equivalence tolerance.

## Tried and not used

- **int8 quantisation** (onnxruntime `quantize_dynamic`): 2.9× faster network and a 4× smaller file, but gamma_mean moved by 26% (median) and up to 7.5×; per-channel was worse (66% median). Rounding errors build up over the 20 layers.
- **Smaller networks trained to copy EXP002** (distillation): 4×128 (0.21 MB, 23× faster) changed 0.74% of decisions in the simplified test; 4×256 (0.80 MB) 0.58%; 6×256 (1.30 MB) 0.56%. Wider or longer training stopped helping at about 0.56%, above the rounding yardstick. Could be revisited if the team accepts a looser tolerance.
- **Running the network in PyTorch**: 1.7× faster but not bit-identical (~1e-6), and no help in the browser, which runs ONNX.

## The integrated pipeline

`python -m o4_efficiency.cli pipeline-cost`

Measured on 200 benchmark comparisons (1 Nov):

| Path | Per comparison | CPU per year at 200k comparisons |
|---|---|---|
| Original: straight to the network | ~0.6 ms | ~2 min |
| Pipeline, before O4's changes | ~9.1 ms | ~30 min |
| Pipeline, now (`run`) | ~6.4 ms | ~21 min |
| Pipeline, now, a school report of 40 through `run_batch` | ~5.8 ms | ~19 min |

- Network route (PASS, 77% of comparisons): 5.2 → 3.3 ms. Fallback route (FLAGGED): 21 → 17 ms.
- Four changes, none of which changes a decision, p-value, rho range, flag or reason (checked on 3,009 cases: 3,000 benchmark comparisons plus the demo scenarios):
  - The network used to be called five times per comparison: once for the comparison and once for each of the four rho variants of step 3+. Now all five go in one `predict_batch` call (`NetworkNull.many`, `Pipeline._network_nulls_many`).
  - The boundary search evaluates d_hat thousands of times for the same rho; the four covariance matrices it needs are now built once per rho and reused (`sigma_stack` cache in `simulate.py`).
  - The simulator's d_hat on its 4,000 trials builds each trial's 3x3 matrix from four per-rho blocks and inverts it in closed form instead of one LAPACK solve per trial (`_quad_form_closed`): about 6x faster for that step. It agrees with the general path to ~1e-15; the only values that moved were a few fallback q95s, by 1e-15 relative. Singular rows are redone the general way. Single-row calls (the boundary search) keep the general path, which is faster there.
  - `Pipeline.run_batch` (for example a school's 40 items) now runs the cheap checks for every request, then sends all network-route requests and their rho variants through the network in one call, sharing variants between requests on the same item and group pair. Same results as `run()`; a 40-item report takes ~232 ms instead of ~257 ms.
- Where the time goes now (stages overlap): fallback simulation (O2), rho range (O3) and screening (O1, SIN-WEI's gate, run twice per comparison with pandas) take most of it; the network is about 5%. The rest can't be cut much further without changing random draws (and so results) or teammates' code.
- At 8 million comparisons a year (200,000 responses × 40 items, the upper end), the pipeline needs about 13–14 CPU hours. How many comparisons 200,000 responses means is still to be confirmed.

## Fallback cost

RYA's simulation, written student by student as in the R code, took up to about 14 minutes per item for a class against a national-sized group (100,000 vs 30, 10,000 trials, measured in Python). Drawing each answer type's students together gives the same distribution in about 15–25 ms whatever the group size. The pipeline's `simulate_d_hat` already works this way.

## Other findings

- `score_school` reads each RY25 survey file once instead of twice (`load_survey` in `school_scoring.py`): 0.80 s → 0.42 s, same output.
- Only 3 schools appear in both RY25 surveys (358 school ids in Pre, 326 in Post, no shared access codes). Either most schools took one survey or the ids were hashed differently per file. Worth checking before before/after comparisons on real data.
- EXP002's Gamma is for d_hat *with* the ½(1/n + 1/m) factor: re-running RYA's simulation for 25 training rows gave a mean ratio of 1.003 to EXP002's Gamma mean.

## Still to do

- Run the fp16 model in a browser (onnxruntime-web) and compare with Python.
- Agree the equivalence tolerance with the team.
- Screening (O1) runs SIN-WEI's gate twice per comparison with pandas; a batched version would be hers to make.

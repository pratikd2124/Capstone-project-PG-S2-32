# Objective 1 Handoff v2 — Prototype-ready Input Screening

**Owner:** SIN-WEI SU  
**Project:** Statistical Core Refinement: Robustness and Efficiency of Minimal-Effects Testing  
**Component:** Objective 1 — Pre-Inference Input Validation and Neural Network Compatibility Screening

## Confirmed matrix inventory

- 59 matrices total
- 53 in-scope 4x4 matrices
- 6 out-of-scope examples

## Public interface

```python
result = gate.screen_case(
    observed_x=observed_x,
    observed_y=observed_y,
    nn_x=nn_x,
    nn_y=nn_y,
    P_matrix=P_matrix,
    question=question,
)
```

If a separate NN pair is not yet available, omit `nn_x` and `nn_y`; the module defaults to the observed pair.

## Routing logic

1. Validate input counts and P matrix.
2. Check the original observed pair for degeneracy.
3. Run the 54-feature compatibility classifier on the exact pair supplied to the Gamma NN.
4. Route to fallback if either the degeneracy safeguard or classifier flags the case.

Possible outputs:
- `pass_to_gamma_nn`
- `fallback_required`
- `invalid_input`

## Detector

- StandardScaler + LogisticRegression
- 54 pre-inference features
- 477 high-confidence labelled cases
  - 333 compatible
  - 144 incompatible
- threshold = 0.10
- no retraining or threshold change in O1-09

## O1-09 result

Classifier-only incompatible recall:
- s=1.00: 100.0%
- s=0.75: 93.1%
- s=0.50: 92.4%
- s=0.25: 92.4%
- s=0.10: 90.3%

Combined classifier + original-pair degeneracy safeguard:
- s=1.00: 100.0%
- s=0.75: 98.6%
- s=0.50: 98.6%
- s=0.25: 99.3%
- s=0.10: 99.3%

At s=0.10 the combined rule caught 143/144 incompatible cases.
Under tested shrink scenarios about 15–17% of compatible cases were routed to fallback.

These are provisional sensitivity results, not production error rates.

## Important design rule

`observed_x / observed_y` are used for the degeneracy safeguard.

`nn_x / nn_y` must be the exact pair supplied to the Gamma neural network and are used by the compatibility classifier.

Objective 1 does not implement the boundary transformation itself.

## Remaining RYA questions

1. Where does the production boundary transformation occur?
2. What production epsilon is used?
3. Is original observed X/Y still available before transformation?
4. Can RYA provide the current JavaScript boundary implementation?

## Files expected

Core:
- o1_screening.py
- O1_07_compatibility_detector_v1.joblib
- O1_07_compatibility_detector_v1_metadata.json
- O1_README.md
- integration_example.py
- O1_demo_cases.json
- HANDOFF_MANIFEST.json

Evidence:
- O1_classifier_inputs_unlabelled.csv
- O1_07_integration_validation_results.csv
- O1_07_detector_integration.ipynb
- O1_08_training_space_and_descriptor_alignment.ipynb
- O1_09_boundary_pair_sensitivity.ipynb
- O1_09_classifier_only_sensitivity.csv
- O1_09_combined_rule_incompatible_summary.csv
- O1_09_combined_rule_compatible_summary.csv

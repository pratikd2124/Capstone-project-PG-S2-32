# Objective 1 Handoff — Input Screening

**Owner:** SIN-WEI SU  
**Component:** Objective 1 — Pre-Inference Input Validation and Neural Network Compatibility Screening

## Purpose
This component is the gate immediately before the existing Gamma neural network.

It checks:
1. whether the unreliability matrix is mathematically valid and structurally represented in historical RYA instance space; and
2. whether the full comparison appears compatible with the Gamma approximation used by the current neural-network pathway.

## Public interface
```python
from o1_screening import O1ScreeningGate

gate = O1ScreeningGate(
    detector_model_path="O1_07_compatibility_detector_v1.joblib",
    detector_metadata_path="O1_07_compatibility_detector_v1_metadata.json",
    screening_reference_csv="O1_classifier_inputs_unlabelled.csv",  # optional
)

result = gate.screen_case(
    x=[40, 52, 7, 14],
    y=[9, 17, 54, 12],
    P_matrix=P,
    question="ry9",
)
```

Possible routes:
- `pass_to_gamma_nn`
- `fallback_required`
- `invalid_input`

## Integration contract for Shlok
```python
screening = gate.screen_case(x=x, y=y, P_matrix=P_matrix, question=question)

if screening["route"] == "pass_to_gamma_nn":
    # call existing / Objective 4 Gamma NN
    pass
elif screening["route"] == "fallback_required":
    # call Objective 2 fallback inference
    pass
else:
    # reject/report invalid input
    pass
```

## Detector summary
- Model: StandardScaler + Logistic Regression
- Inputs: 54 pre-inference features
- Training set: 477 high-confidence labelled cases
  - 333 compatible
  - 144 incompatible
- Routing threshold: 0.10
- Threshold status: provisional high-recall operating threshold

## Validation behaviour
The module checks 4-category scope, 4x4 P shape, non-negative counts, non-negative P entries, finite values, non-empty groups and P row-sum validity.

Small row-sum deviations up to 0.02 are currently accepted with a warning and are not renormalised. Confirm this tolerance with RYA before production use.

## Matrix descriptors
For RYA/paper traceability the module reports:
- `certainty_P`: sum of diagonal probabilities
- `rya_asymmetry_P`: mean squared difference across the six unique off-diagonal symmetric pairs

The trained detector keeps its historical asymmetry definition internally. Do not silently alter the trained model just to rename this feature.

## Training-space diagnostic
When `O1_classifier_inputs_unlabelled.csv` is supplied, the module also reports nearest-neighbour distance in the standardised 16-dimensional P-matrix space and an empirical percentile.

This is diagnostic only. No 95th/99th percentile hard fallback rule is currently validated.

Observed examples:

| Case | Distance | Empirical percentile | Interpretation |
|---|---:|---:|---|
| ry9 live matrix | 0.1612 | 1.46% | Strongly represented |
| incompatible ry24 instance | 0.2467 | 47.68% | Ordinary matrix; failure driven elsewhere |
| synthetic OOD matrix | 21.1176 | 100.00% | Clearly outside represented matrix space |

## Acceptance tests
### Safe case
- question: `ry9`
- x: `[40, 52, 7, 14]`
- y: `[9, 17, 54, 12]`
- expected route: `pass_to_gamma_nn`
- expected incompatibility probability: approximately `0.0000448`
- previously verified downstream Gamma output: shape `9.9705`, rate `20.9264`

### Fallback case
- exp_code: `orthog_B_3_large_classes_2166`
- question: `ry24`
- n_X = 6
- n_Y = 5215
- expected route: `fallback_required`
- expected incompatibility probability: approximately `0.999999897`
- use the exact instance-specific P matrix from the historical screening dataset

## Files Shlok should receive
Required:
- `o1_screening.py`
- `O1_07_compatibility_detector_v1.joblib`
- `O1_07_compatibility_detector_v1_metadata.json`

Recommended:
- `O1_classifier_inputs_unlabelled.csv`
- `O1_07_integration_validation_results.csv`
- `O1_07_detector_integration.ipynb`
- `O1_08_training_space_and_descriptor_alignment.ipynb`
- `O1_demo_cases.json`

## Open confirmations before RYA presentation
1. Confirm the x/y pair entering O1 is the exact same boundary-instance pair passed to the Gamma NN.
2. Confirm the authoritative matrix/item count. Current analysis observed 59 total matrices, with 53 4x4 matrices in O1 scope; the architecture briefing mentions 58.
3. Confirm whether row-sum deviations up to 0.02 should remain warnings.
4. Confirm whether live inference uses an instance-specific P matrix or selects P from the question identifier.

## What O1 does not own
- fallback statistical method (Objective 2)
- uncertainty/Bayesian work (Objective 3)
- model compression (Objective 4)
- full-pipeline integration and benchmarking (team lead)

O1's output is a routing decision plus supporting diagnostics.

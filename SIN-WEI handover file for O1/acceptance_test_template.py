from o1_screening import O1ScreeningGate

gate = O1ScreeningGate(
    detector_model_path="O1_07_compatibility_detector_v1.joblib",
    detector_metadata_path="O1_07_compatibility_detector_v1_metadata.json",
)

P = [
    [0.70, 0.10, 0.10, 0.10],
    [0.10, 0.70, 0.10, 0.10],
    [0.10, 0.10, 0.70, 0.10],
    [0.10, 0.10, 0.10, 0.70],
]

r = gate.screen_case(
    observed_x=[10, 0, 0, 0],
    observed_y=[3, 2, 4, 1],
    P_matrix=P,
)
assert r["route"] == "fallback_required"
assert r["reason"] == "original_pair_degenerate"
print("PASS: degenerate safeguard")

r = gate.screen_case(
    observed_x=[1, -1, 3, 2],
    observed_y=[1, 2, 3, 4],
    P_matrix=P,
)
assert r["route"] == "invalid_input"
print("PASS: invalid input")

print("Interface-level acceptance tests completed.")

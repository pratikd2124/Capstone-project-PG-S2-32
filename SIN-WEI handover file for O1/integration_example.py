import numpy as np
from o1_screening import O1ScreeningGate

gate = O1ScreeningGate(
    detector_model_path="O1_07_compatibility_detector_v1.joblib",
    detector_metadata_path="O1_07_compatibility_detector_v1_metadata.json",
    screening_reference_csv="O1_classifier_inputs_unlabelled.csv",
)

# Replace with the exact P matrix used by the team prototype.
P = np.eye(4)

screening = gate.screen_case(
    x=[40, 52, 7, 14],
    y=[9, 17, 54, 12],
    P_matrix=P,
    question="ry9",
)

print(screening)

if screening["route"] == "pass_to_gamma_nn":
    print("Call the Gamma-NN stage.")
elif screening["route"] == "fallback_required":
    print("Call the Objective 2 fallback stage.")
else:
    print("Reject/report invalid input.")

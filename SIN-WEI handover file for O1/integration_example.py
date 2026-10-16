from o1_screening import O1ScreeningGate

gate = O1ScreeningGate(
    detector_model_path="O1_07_compatibility_detector_v1.joblib",
    detector_metadata_path="O1_07_compatibility_detector_v1_metadata.json",
    screening_reference_csv="O1_classifier_inputs_unlabelled.csv",
)

observed_x = [40, 52, 7, 14]
observed_y = [9, 17, 54, 12]

# Until RYA confirms the boundary step, these may be the same as observed_x/y.
nn_x = observed_x
nn_y = observed_y

P_matrix = [
    [0.70, 0.10, 0.10, 0.10],
    [0.10, 0.70, 0.10, 0.10],
    [0.10, 0.10, 0.70, 0.10],
    [0.10, 0.10, 0.10, 0.70],
]

screening = gate.screen_case(
    observed_x=observed_x,
    observed_y=observed_y,
    nn_x=nn_x,
    nn_y=nn_y,
    P_matrix=P_matrix,
    question="example",
)

print(screening)

if screening["route"] == "fallback_required":
    print("Route to Objective 2 fallback:", screening["reason"])
elif screening["route"] == "pass_to_gamma_nn":
    print("Route to Gamma NN using the same nn_x/nn_y")
else:
    print("Invalid input:", screening.get("reason"))

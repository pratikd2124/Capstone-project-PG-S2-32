"""Statistical Core Refinement (SCRP) toolkit.

A cleaned-up, script-based version of the scattered R/Python files in the
"Statistical Core Refinement Project" folder (feature_calculation_portal_script.py,
net_dataset_init.py, train_test_loop.py, perform_exp.py, misc_functions.py) plus
the working Colab notebook that pulled them together.

Public sub-modules:
    config       - locates the project's data files (model, P-matrices, CSVs)
    reliability  - P-matrix asymmetry scoring / plotting
    features     - feature engineering shared by inference and training
    inference    - loads the trained ONNX model and scores response counts
    dataset      - builds the training dataset from instance_data/*.csv
    model        - the Net architecture and the Gamma-KL loss
    train        - training loop for a from-scratch Net
    compare      - compares a freshly trained model against the shipped ONNX one
"""

__version__ = "0.1.0"

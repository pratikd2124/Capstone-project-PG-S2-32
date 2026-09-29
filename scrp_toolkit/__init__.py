"""SCRP toolkit: RYA's Gemfinder statistical core, as one tested Python package.

RYA's original code (in the customer's Box folder) is a set of R and Python scripts that only
ran on their own machines. This package does the same job, checked line by line against their
code, so the whole team can run it and measure every experiment against the same baseline.

What's in here, roughly in the order the data flows:

    settings     every path and choice in one list (override yours in local_settings.py)
    config       finds the customer's files on disk
    reliability  the unreliability matrices and their asymmetry score
    features     turns answer counts into the model's 40 inputs
    inference    runs RYA's trained model (EXP002) and returns shape / rate
    scoring      scores every item for a real school in the RY25 survey
    dataset      rebuilds the training data exactly as RYA prepared it
    model        the network design and its training loss
    train        trains a fresh network (for experiments only)
    compare      compares a fresh network with RYA's trained one
    validate     quick check of the model against a few known answers
    baseline     the full 359-check comparison against RYA
    edge_cases   unusual inputs, for the input-safety detector
    rds          reads RYA's R data file (P.RData) without extra packages
    cli          the command-line tool: python -m scrp_toolkit.cli <command>
"""

__version__ = "0.2.0"

#!/usr/bin/env bash
# SCRP toolkit: one-command setup for Mac / Linux.
#   bash setup_env.sh        (run in the repo folder, the one with README.md)
# Creates or updates the "scrp" conda environment, registers the notebook kernel,
# then checks everything works.
set -e

command -v conda >/dev/null || { echo "conda not found: install Anaconda or Miniconda first."; exit 1; }
eval "$(conda shell.bash hook)"

echo "[1/4] Creating the 'scrp' environment..."
if conda env list | grep -q '^scrp '; then
    conda env update -f environment.yml --prune
else
    conda env create -f environment.yml
fi

echo "[2/4] Activating it..."
conda activate scrp

echo "[3/4] Registering the notebook kernel 'Python (scrp)'..."
python -m ipykernel install --user --name scrp --display-name "Python (scrp)"

echo "[4/4] Checking everything works..."
python -c "import sys, numpy, pandas, torch, onnxruntime; print('   Python', sys.version.split()[0], '| numpy', numpy.__version__, '| pandas', pandas.__version__, '| torch', torch.__version__, '| onnxruntime', onnxruntime.__version__)"
python -m pytest -q -p no:cacheprovider tests
python -m scrp_toolkit.cli baseline --skip-dataset

echo
echo "All set. From now on: conda activate scrp"
echo "Full check (about 2 min): python -m scrp_toolkit.cli baseline"

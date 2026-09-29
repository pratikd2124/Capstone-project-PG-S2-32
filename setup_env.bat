@echo off
rem ------------------------------------------------------------------------------
rem  SCRP toolkit: one-click setup for Windows.
rem
rem  Run this from the Anaconda Prompt, in the repo folder (the one with README.md):
rem      setup_env.bat
rem
rem  It creates (or updates) the "scrp" conda environment, registers it for Jupyter /
rem  VS Code as "Python (scrp)", then checks everything works.
rem ------------------------------------------------------------------------------

where conda >nul 2>nul
if errorlevel 1 (
    echo conda was not found. Open the "Anaconda Prompt" from the Start menu and run this again.
    exit /b 1
)

echo.
echo [1/4] Creating the "scrp" environment (first time takes several minutes)...
call conda env list | findstr /b /c:"scrp " >nul
if errorlevel 1 (
    call conda env create -f environment.yml || goto :failed
) else (
    echo      "scrp" already exists - updating it instead.
    call conda env update -f environment.yml --prune || goto :failed
)

echo.
echo [2/4] Activating it...
call conda activate scrp || goto :failed

echo.
echo [3/4] Registering the notebook kernel "Python (scrp)"...
python -m ipykernel install --user --name scrp --display-name "Python (scrp)" || goto :failed

echo.
echo [4/4] Checking everything works...
python -c "import sys, numpy, pandas, torch, onnxruntime; print('   Python', sys.version.split()[0], '| numpy', numpy.__version__, '| pandas', pandas.__version__, '| torch', torch.__version__, '| onnxruntime', onnxruntime.__version__)" || goto :failed
python -m pytest -q -p no:cacheprovider tests || goto :failed
python -m scrp_toolkit.cli baseline --skip-dataset || goto :failed

echo.
echo ==============================================================================
echo  All set. From now on:  conda activate scrp
echo  Full check (about 2 min):  python -m scrp_toolkit.cli baseline
echo  Walkthrough: open walkthrough.ipynb in VS Code, kernel "Python (scrp)"
echo ==============================================================================
exit /b 0

:failed
echo.
echo Setup stopped at the step above. Copy the error message and send it to Shlok.
exit /b 1

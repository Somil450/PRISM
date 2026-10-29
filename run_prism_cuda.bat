@echo off
REM ============================================================================
REM PRISM - Provenance & Reasoning Intelligence System for Multimedia
REM High-Performance Forensic Dashboard Launcher with NVIDIA CUDA Acceleration
REM ============================================================================

title PRISM Forensic Intelligence System - CUDA Launcher
color 0B

echo.
echo ============================================================================
echo   ____  ____  ___ ____  __  __
echo  ^|  _ \^|  _ \^|_ _/ ___^|^|  \/  ^|   PROVENANCE ^& REASONING INTELLIGENCE
echo  ^| ^|_) ^| ^|_) ^|^| ^|\___ \^| ^|\/^| ^|   SYSTEM FOR MULTIMEDIA
echo  ^|  __/^|  _ ^< ^| ^| ___) ^| ^|  ^| ^|   NIST / SWGDE Forensic Integrity
echo  ^|_^|   ^|_^| \_\___^|____/^|_^|  ^|_^|   Dual-Stream ViT + Multi-Agent D-S Fusion
echo ============================================================================
echo.

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Find working Python executable with PyTorch & Streamlit
set "PYTHON_EXE="

REM 1. Check py -3.13
py -3.13 -c "import torch, streamlit" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=py -3.13"
    set "STREAMLIT_EXE=py -3.13 -m streamlit"
    echo [OK] Python 3.13 with PyTorch CUDA detected.
    goto :PYTHON_FOUND
)

REM 2. Check standard python command
python -c "import torch, streamlit" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=python"
    set "STREAMLIT_EXE=python -m streamlit"
    echo [OK] System Python with PyTorch CUDA detected.
    goto :PYTHON_FOUND
)

REM 3. Check virtualenv if valid
if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    "%SCRIPT_DIR%.venv\Scripts\python.exe" -c "import torch, streamlit" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"
        set "STREAMLIT_EXE=%SCRIPT_DIR%.venv\Scripts\streamlit.exe"
        echo [OK] Virtual environment .venv detected.
        goto :PYTHON_FOUND
    )
)

REM Fallback default
set "PYTHON_EXE=py -3.13"
set "STREAMLIT_EXE=py -3.13 -m streamlit"
echo [!] Defaulting to py -3.13 ...

:PYTHON_FOUND
echo.
echo [INFO] Auditing PyTorch Neural Hardware and CUDA acceleration...
%PYTHON_EXE% -c "import torch; d = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'; print(f'[CUDA STATUS] PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | Device: {d}')"
echo.

echo ============================================================================
echo Select PRISM Operational Interface:
echo   [1] PRISM Forensic Web Dashboard (Streamlit GUI - Recommended)
echo   [2] Terminal Interactive Forensic Console
echo   [3] Automated Benchmark Evaluation (100+ Perturbed Samples)
echo   [4] Verify CUDA Diagnostics and Agent Telemetry
echo   [0] Exit
echo ============================================================================
echo.

set /p choice="Select option [Default: 1]: "
if "%choice%"=="" set choice=1

if "%choice%"=="1" (
    echo.
    echo [LAUNCH] Starting PRISM Forensic Intelligence Dashboard on http://localhost:8501 ...
    echo [TIP] Keep this terminal window open while using the web interface.
    echo.
    %STREAMLIT_EXE% run app.py --server.port 8501
) else if "%choice%"=="2" (
    echo.
    echo [LAUNCH] Starting Terminal Interactive Console...
    %PYTHON_EXE% da2\interactive_demo.py
) else if "%choice%"=="3" (
    echo.
    echo [LAUNCH] Running Automated Benchmark Suite...
    %PYTHON_EXE% da2\evaluate_metrics.py
) else if "%choice%"=="4" (
    echo.
    echo [DIAGNOSTICS] Testing Multi-Agent Forensic Pipeline...
    %PYTHON_EXE% -m pytest tests/test_all_agents.py tests/test_pipeline_diagnostics.py
) else (
    echo Exiting PRISM Launcher.
)

echo.
pause

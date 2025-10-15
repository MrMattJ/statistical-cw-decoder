@echo off
echo Starting Confidence-Based Statistical CW Decoder...
echo.
echo This version uses 4 progressive processors (1s/5s/10s/20s) for maximum accuracy.
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

REM Install dependencies if needed
echo Checking dependencies...
pip show numpy >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install numpy scipy scikit-learn sounddevice
)

REM Launch the decoder
pythonw statistical_decoder_confidence.py

if errorlevel 1 (
    echo.
    echo ERROR: Failed to start decoder
    echo Make sure all dependencies are installed: numpy scipy scikit-learn sounddevice
    pause
)

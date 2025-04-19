@echo off
setlocal

:: Run Constituent Analysis
:: This script runs the matrix-based constituent anomaly detection analysis

:: Change to the directory where this script is located
cd /d "%~dp0"

:: Ensure the virtual environment is activated if available
if exist "..\venv\Scripts\activate.bat" (
    call ..\venv\Scripts\activate.bat
)

:: Run the complete constituent analysis
echo Starting constituent analysis...
python run_complete_constituent_analysis.py

if %ERRORLEVEL% neq 0 (
    echo Error occurred during constituent analysis
    pause
    exit /b %ERRORLEVEL%
)

echo Constituent analysis completed!
pause
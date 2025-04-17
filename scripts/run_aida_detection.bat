@echo off
setlocal enabledelayedexpansion

:: AIDA Anomaly Detection Workflow Script for Windows

:: Change to project root directory
cd /d "%~dp0\.."

:: Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

:: Paths
set PROCESSED_DATA_DIR=data\processed
set AIDA_CPP_DIR=AIDA\C++
set INPUT_FILE=%PROCESSED_DATA_DIR%\sp500_index_processed.csv
set AIDA_EXECUTABLE=%AIDA_CPP_DIR%\build\aida_sp500_anomaly_detection.exe

:: Create build directory
mkdir "%AIDA_CPP_DIR%\build" 2>nul

:: Compile AIDA executable (adjust compilation command as needed)
echo Compiling AIDA executable...
cd "%AIDA_CPP_DIR%"
g++ -std=c++11 -O3 -fopenmp ^
    -I./include ^
    src\aida_sp500_anomaly_detection.cpp ^
    src\aida_class.cpp ^
    src\distance_metrics.cpp ^
    src\isolation_formulas.cpp ^
    src\aggregation_functions.cpp ^
    src\rng_class.cpp ^
    -o build\aida_sp500_anomaly_detection.exe

:: Return to project root
cd /d "%~dp0\.."

:: Run AIDA anomaly detection
echo Running AIDA anomaly detection...
"%AIDA_EXECUTABLE%" "%INPUT_FILE%"

:: Process results with Python
echo Processing AIDA results...
python -m scripts.process_aida_results

echo AIDA anomaly detection completed.
pause
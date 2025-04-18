@echo off
setlocal enabledelayedexpansion

:: Run Subsequence Analysis
:: This script runs the subsequence anomaly detection analysis with multiple configurations

:: Change to the directory where this script is located
cd /d "%~dp0"

:: Default values
set SUBSEQUENCE_DIR=..\data\processed\subsequences
set OUTPUT_DIR=..\data\subsequence_results
set ALGORITHMS=all
set WINDOW_SIZES=3 5
set OVERLAP_MODE=all
set RUN_COMPARISON=true

:: Parse command line arguments
:parse_args
if "%~1"=="" goto :end_parse_args
if "%~1"=="--subsequence-dir" (
    set SUBSEQUENCE_DIR=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--output-dir" (
    set OUTPUT_DIR=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--algorithms" (
    set ALGORITHMS=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--window-sizes" (
    set WINDOW_SIZES=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--overlap-mode" (
    set OVERLAP_MODE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--no-comparison" (
    set RUN_COMPARISON=false
    shift
    goto :parse_args
)
echo Unknown option: %~1
exit /b 1

:end_parse_args

:: Ensure subdirectories exist
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

:: Build command line for python script
set CMD=python run_multiple_configs.py --subsequence-dir "%SUBSEQUENCE_DIR%" --output "%OUTPUT_DIR%" --window-sizes %WINDOW_SIZES% --algorithms %ALGORITHMS%

:: Add overlap mode
if "%OVERLAP_MODE%"=="all" (
    set CMD=%CMD% --all-overlaps
) else if "%OVERLAP_MODE%"=="overlap" (
    set CMD=%CMD% --only-overlap
) else if "%OVERLAP_MODE%"=="nonoverlap" (
    set CMD=%CMD% --only-non-overlap
)

:: Add comparison flag if needed
if "%RUN_COMPARISON%"=="true" (
    set CMD=%CMD% --run-comparison
)

:: Print the command
echo Running command: %CMD%

:: Execute the command
%CMD%

echo Subsequence analysis completed!
pause
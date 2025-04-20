@echo off
setlocal enabledelayedexpansion

:: Run Constituent Anomaly Analysis
:: This script runs the constituent anomaly analysis with configurable parameters

:: Change to the directory where this script is located
cd /d "%~dp0"

:: Default values
set INDEX_RESULTS=..\data\subsequence_results
set CONSTITUENT_RESULTS=..\data\constituent_results
set OUTPUT_DIR=..\data\constituent_analysis
set WINDOW_DAYS=3
set ALGORITHMS=all

:: Parse command line arguments
:parse_args
if "%~1"=="" goto :end_parse_args
if "%~1"=="--index-results" (
    set INDEX_RESULTS=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--constituent-results" (
    set CONSTITUENT_RESULTS=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--output" (
    set OUTPUT_DIR=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--window-days" (
    set WINDOW_DAYS=%~2
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
echo Unknown option: %~1
exit /b 1

:end_parse_args

:: Ensure output directory exists
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

:: Build command line for python script
set CMD=python run_constituent_analysis.py --index-results "%INDEX_RESULTS%" --constituent-results "%CONSTITUENT_RESULTS%" --output "%OUTPUT_DIR%" --window-days %WINDOW_DAYS% --algorithms %ALGORITHMS%

:: Print the command
echo Running command: %CMD%

:: Execute the command
%CMD%

if %ERRORLEVEL% NEQ 0 (
    echo Constituent anomaly analysis failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)

echo Constituent anomaly analysis completed successfully!
pause
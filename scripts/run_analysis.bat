@echo off
setlocal enabledelayedexpansion

:: Run complete anomaly detection analysis
:: This script runs the complete analysis pipeline with various configurations

:: Change to the directory where this script is located
cd /d "%~dp0"

:: Default values
set WINDOW_SIZES=3 5
set ALGORITHMS=all
set OVERLAP_MODES=both
set OUTPUT_DIR=..\data\comparative_analysis
set SKIP_SUBSEQUENCE=false
set SKIP_MATRIX=false
set SKIP_CONSTITUENT=false

:: Parse command line arguments
:parse_args
if "%~1"=="" goto :end_parse_args
if "%~1"=="--window-sizes" (
    set WINDOW_SIZES=%~2
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
if "%~1"=="--overlap-modes" (
    set OVERLAP_MODES=%~2
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
if "%~1"=="--skip-subsequence" (
    set SKIP_SUBSEQUENCE=true
    shift
    goto :parse_args
)
if "%~1"=="--skip-matrix" (
    set SKIP_MATRIX=true
    shift
    goto :parse_args
)
if "%~1"=="--skip-constituent" (
    set SKIP_CONSTITUENT=true
    shift
    goto :parse_args
)
echo Unknown option: %~1
exit /b 1

:end_parse_args

:: Ensure output directory exists
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

:: Print configuration
echo Running anomaly detection analysis with the following configuration:
echo Window sizes: %WINDOW_SIZES%
echo Algorithms: %ALGORITHMS%
echo Overlap modes: %OVERLAP_MODES%
echo Output directory: %OUTPUT_DIR%
echo Skip subsequence analysis: %SKIP_SUBSEQUENCE%
echo Skip matrix analysis: %SKIP_MATRIX%
echo Skip constituent analysis: %SKIP_CONSTITUENT%
echo.

:: Determine overlap settings
set OVERLAP_ARGS=
if "%OVERLAP_MODES%"=="overlap" (
    set OVERLAP_ARGS=--overlap
) else if "%OVERLAP_MODES%"=="nonoverlap" (
    set OVERLAP_ARGS=--no-overlap
) else if "%OVERLAP_MODES%"=="both" (
    set OVERLAP_ARGS=overlap nonoverlap
) else (
    echo Invalid overlap mode: %OVERLAP_MODES%
    exit /b 1
)

:: Build skip arguments
set SKIP_ARGS=
if "%SKIP_SUBSEQUENCE%"=="true" (
    set SKIP_ARGS=!SKIP_ARGS! --skip-subsequence
)
if "%SKIP_MATRIX%"=="true" (
    set SKIP_ARGS=!SKIP_ARGS! --skip-matrix
)
if "%SKIP_CONSTITUENT%"=="true" (
    set SKIP_ARGS=!SKIP_ARGS! --skip-constituent
)

:: Run analysis for each configuration
for %%s in (%WINDOW_SIZES%) do (
    for %%o in (%OVERLAP_ARGS%) do (
        set OVERLAP_FLAG=
        if "%%o"=="overlap" set OVERLAP_FLAG=--overlap
        if "%%o"=="nonoverlap" set OVERLAP_FLAG=--no-overlap
        
        :: Build command
        set CMD=python run_complete_analysis.py --window-size %%s !OVERLAP_FLAG! --algorithms %ALGORITHMS% --output "%OUTPUT_DIR%" %SKIP_ARGS%
        
        echo Running command: !CMD!
        call !CMD!
        
        if !ERRORLEVEL! EQU 0 (
            echo Analysis completed successfully for window size %%s with %%o
        ) else (
            echo Analysis failed for window size %%s with %%o
        )
        
        echo.
    )
)

echo All analyses completed!
pause
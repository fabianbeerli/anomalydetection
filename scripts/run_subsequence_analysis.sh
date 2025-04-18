#!/bin/bash

# Run Subsequence Analysis
# This script runs the subsequence anomaly detection analysis with multiple configurations

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Default values
SUBSEQUENCE_DIR="../data/processed/subsequences"
OUTPUT_DIR="../data/subsequence_results"
ALGORITHMS="all"
WINDOW_SIZES="3 5"
OVERLAP_MODE="all"  # Options: "all", "overlap", "nonoverlap"
RUN_COMPARISON=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --subsequence-dir)
            SUBSEQUENCE_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --algorithms)
            ALGORITHMS="$2"
            shift 2
            ;;
        --window-sizes)
            WINDOW_SIZES="$2"
            shift 2
            ;;
        --overlap-mode)
            OVERLAP_MODE="$2"
            shift 2
            ;;
        --no-comparison)
            RUN_COMPARISON=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Ensure subdirectories exist
mkdir -p "$OUTPUT_DIR"

# Build command line for python script
CMD="python run_multiple_configs.py --subsequence-dir $SUBSEQUENCE_DIR --output $OUTPUT_DIR --window-sizes $WINDOW_SIZES --algorithms $ALGORITHMS"

# Add overlap mode
if [ "$OVERLAP_MODE" == "all" ]; then
    CMD="$CMD --all-overlaps"
elif [ "$OVERLAP_MODE" == "overlap" ]; then
    CMD="$CMD --only-overlap"
elif [ "$OVERLAP_MODE" == "nonoverlap" ]; then
    CMD="$CMD --only-non-overlap"
fi

# Add comparison flag if needed
if [ "$RUN_COMPARISON" = true ]; then
    CMD="$CMD --run-comparison"
fi

# Print the command
echo "Running command: $CMD"

# Execute the command
eval $CMD

echo "Subsequence analysis completed!"
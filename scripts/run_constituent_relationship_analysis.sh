#!/bin/bash

# Run Constituent Anomaly Analysis
# This script runs the constituent anomaly analysis with configurable parameters

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Default values
INDEX_RESULTS="../data/subsequence_results"
CONSTITUENT_RESULTS="../data/constituent_results"
OUTPUT_DIR="../data/constituent_analysis"
WINDOW_DAYS=3
ALGORITHMS="all"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --index-results)
            INDEX_RESULTS="$2"
            shift 2
            ;;
        --constituent-results)
            CONSTITUENT_RESULTS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --window-days)
            WINDOW_DAYS="$2"
            shift 2
            ;;
        --algorithms)
            ALGORITHMS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Build command line for python script
CMD="python run_constituent_analysis.py --index-results \"$INDEX_RESULTS\" --constituent-results \"$CONSTITUENT_RESULTS\" --output \"$OUTPUT_DIR\" --window-days $WINDOW_DAYS --algorithms $ALGORITHMS"

# Print the command
echo "Running command: $CMD"

# Execute the command
eval $CMD

if [ $? -ne 0 ]; then
    echo "Constituent anomaly analysis failed with error code $?"
    exit 1
fi

echo "Constituent anomaly analysis completed successfully!"
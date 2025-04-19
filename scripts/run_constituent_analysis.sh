#!/bin/bash

# Run Constituent Analysis
# This script runs the matrix-based constituent anomaly detection analysis

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Ensure the virtual environment is activated if available
if [ -d "../venv" ]; then
    source ../venv/bin/activate
fi

# Run the complete constituent analysis
echo "Starting constituent analysis..."
python run_complete_constituent_analysis.py

echo "Constituent analysis completed!"
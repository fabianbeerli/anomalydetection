#!/bin/bash

# AIDA Anomaly Detection Workflow Script

# Ensure we're in the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Source virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Paths
PROCESSED_DATA_DIR="data/processed"
AIDA_CPP_DIR="AIDA/C++"
INPUT_FILE="$PROCESSED_DATA_DIR/sp500_index_processed.csv"
AIDA_EXECUTABLE="$AIDA_CPP_DIR/build/aida_sp500_anomaly_detection"

# Compile AIDA executable (adjust compilation command as needed)
echo "Compiling AIDA executable..."
mkdir -p "$AIDA_CPP_DIR/build"
cd "$AIDA_CPP_DIR"
g++ -std=c++11 -O3 -fopenmp \
    -I./include \
    src/aida_sp500_anomaly_detection.cpp \
    src/aida_class.cpp \
    src/distance_metrics.cpp \
    src/isolation_formulas.cpp \
    src/aggregation_functions.cpp \
    src/rng_class.cpp \
    -o build/aida_sp500_anomaly_detection

# Return to project root
cd "$PROJECT_ROOT"

# Run AIDA anomaly detection
echo "Running AIDA anomaly detection..."
"$AIDA_EXECUTABLE" "$INPUT_FILE"

# Process results with Python
echo "Processing AIDA results..."
python -m scripts.process_aida_results

echo "AIDA anomaly detection completed."
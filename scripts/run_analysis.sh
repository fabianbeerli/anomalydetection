#!/bin/bash

# Run complete anomaly detection analysis
# This script runs the complete analysis pipeline with various configurations

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Default values
WINDOW_SIZES="3 5"
ALGORITHMS="all"
OVERLAP_MODES="both"  # Options: "overlap", "nonoverlap", "both"
OUTPUT_DIR="../data/comparative_analysis"
SKIP_SUBSEQUENCE=false
SKIP_MATRIX=false
SKIP_CONSTITUENT=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --window-sizes)
            WINDOW_SIZES="$2"
            shift 2
            ;;
        --algorithms)
            ALGORITHMS="$2"
            shift 2
            ;;
        --overlap-modes)
            OVERLAP_MODES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-subsequence)
            SKIP_SUBSEQUENCE=true
            shift
            ;;
        --skip-matrix)
            SKIP_MATRIX=true
            shift
            ;;
        --skip-constituent)
            SKIP_CONSTITUENT=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "Running anomaly detection analysis with the following configuration:"
echo "Window sizes: $WINDOW_SIZES"
echo "Algorithms: $ALGORITHMS"
echo "Overlap modes: $OVERLAP_MODES"
echo "Output directory: $OUTPUT_DIR"
echo "Skip subsequence analysis: $SKIP_SUBSEQUENCE"
echo "Skip matrix analysis: $SKIP_MATRIX"
echo "Skip constituent analysis: $SKIP_CONSTITUENT"
echo ""

# Determine overlap settings
OVERLAP_ARGS=()
if [ "$OVERLAP_MODES" == "overlap" ]; then
    OVERLAP_ARGS=("--overlap")
elif [ "$OVERLAP_MODES" == "nonoverlap" ]; then
    OVERLAP_ARGS=("--no-overlap")
elif [ "$OVERLAP_MODES" == "both" ]; then
    OVERLAP_ARGS=("--overlap" "--no-overlap")
else
    echo "Invalid overlap mode: $OVERLAP_MODES"
    exit 1
fi

# Build skip arguments
SKIP_ARGS=()
if [ "$SKIP_SUBSEQUENCE" = true ]; then
    SKIP_ARGS+=("--skip-subsequence")
fi
if [ "$SKIP_MATRIX" = true ]; then
    SKIP_ARGS+=("--skip-matrix")
fi
if [ "$SKIP_CONSTITUENT" = true ]; then
    SKIP_ARGS+=("--skip-constituent")
fi

# Function to build algorithm arguments
function build_algo_args {
    local algos="$1"
    local result=""
    
    if [ "$algos" == "all" ]; then
        result="all"
    else
        result="${algos// / }"
    fi
    
    echo "$result"
}

ALGO_ARGS=$(build_algo_args "$ALGORITHMS")

# Run analysis for each configuration
for window_size in $WINDOW_SIZES; do
    for overlap_arg in "${OVERLAP_ARGS[@]}"; do
        # Build command
        cmd="python run_complete_analysis.py --window-size $window_size $overlap_arg --algorithms $ALGO_ARGS --output $OUTPUT_DIR ${SKIP_ARGS[@]}"
        
        echo "Running command: $cmd"
        eval $cmd
        
        if [ $? -eq 0 ]; then
            echo "Analysis completed successfully for window size $window_size with $overlap_arg"
        else
            echo "Analysis failed for window size $window_size with $overlap_arg"
        fi
        
        echo ""
    done
done

echo "All analyses completed!"
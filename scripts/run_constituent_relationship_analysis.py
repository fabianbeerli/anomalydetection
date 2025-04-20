#!/usr/bin/env python
"""
Script to run constituent anomaly analysis, comparing AIDA, Isolation Forest, and LOF
for detecting anomalies in S&P 500 constituents and their relationship to index anomalies.
"""
import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.utils.helpers import ensure_directory_exists

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to run constituent anomaly analysis.
    """
    parser = argparse.ArgumentParser(
        description="Run constituent anomaly analysis comparing AIDA, Isolation Forest, and LOF"
    )
    parser.add_argument(
        "--index-results", 
        type=str, 
        default=str(config.DATA_DIR / "subsequence_results"),
        help="Directory containing S&P 500 index anomaly detection results"
    )
    parser.add_argument(
        "--constituent-results", 
        type=str, 
        default=str(config.DATA_DIR / "constituent_results"),
        help="Directory containing constituent anomaly detection results"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=str(config.DATA_DIR / "constituent_analysis"),
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--window-days", 
        type=int, 
        default=5,
        help="Number of days to look around each index anomaly"
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["aida", "iforest", "lof", "all"],
        default=["all"],
        help="Algorithms to include in the analysis"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = Path(args.output)
    ensure_directory_exists(output_dir)
    
    # Call the constituent analysis script
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "analyze_constituent_anomalies.py"),
        "--index-results", args.index_results,
        "--constituent-results", args.constituent_results,
        "--output", args.output,
        "--window-days", str(args.window_days),
        "--algorithms"
    ]
    
    # Add algorithms
    if "all" in args.algorithms:
        cmd.extend(["aida", "iforest", "lof"])
    else:
        cmd.extend(args.algorithms)
    
    # Run the command
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Constituent anomaly analysis completed successfully. Results saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running constituent anomaly analysis: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
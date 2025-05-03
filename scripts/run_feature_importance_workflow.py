#!/usr/bin/env python
"""
Script to run both feature importance analysis and visualization for LOF and IForest.
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
    Main function to run feature importance analysis and visualization.
    """
    parser = argparse.ArgumentParser(description="Run feature importance analysis and visualization for LOF and IForest")
    
    # General arguments
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=str(config.DATA_DIR / "analysis_results"),
        help="Base directory for analysis results"
    )
    parser.add_argument(
        "--processed-dir", 
        type=str, 
        default=str(config.PROCESSED_DATA_DIR),
        help="Directory for processed data"
    )
    
    # Sequence analysis arguments
    parser.add_argument(
        "--ticker", 
        type=str, 
        default="sp500",
        help="Ticker to analyze"
    )
    parser.add_argument(
        "--window-sizes", 
        type=str,
        default="3",
        help="Comma-separated list of window sizes"
    )
    parser.add_argument(
        "--only-overlap", 
        action="store_true",
        help="Only analyze overlapping subsequences"
    )
    parser.add_argument(
        "--only-nonoverlap", 
        action="store_true",
        help="Only analyze non-overlapping subsequences"
    )
    
    # Debug option
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    # Create output directories
    feature_importance_results_dir = Path(args.output_dir) / "feature_importance_results"
    ensure_directory_exists(feature_importance_results_dir)
    
    feature_importance_visualization_dir = Path(args.output_dir) / "feature_importance_visualization"
    ensure_directory_exists(feature_importance_visualization_dir)
    
    # Run feature importance analysis script
    logger.info("Running feature importance analysis...")
    
    analysis_cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_feature_importance_analysis.py"),
        "--output-dir", str(feature_importance_results_dir)
    ]
    
    # Add additional arguments
    if args.ticker:
        analysis_cmd.extend(["--ticker", args.ticker])
    
    if args.window_sizes:
        analysis_cmd.extend(["--window-sizes", args.window_sizes])
    
    if args.only_overlap:
        analysis_cmd.append("--only-overlap")
    elif args.only_nonoverlap:
        analysis_cmd.append("--only-nonoverlap")
    
    if args.debug:
        analysis_cmd.append("--debug")
    
    logger.info(f"Running analysis command: {' '.join(analysis_cmd)}")
    analysis_result = subprocess.run(analysis_cmd, check=True)
    
    # Run visualization script
    logger.info("Running feature importance visualization...")
    
    visualization_cmd = [
        sys.executable,
        str(Path(__file__).parent / "visualize_feature_importance_results.py"),
        "--feature-importance-dir", str(feature_importance_results_dir),
        "--output-dir", str(feature_importance_visualization_dir),
        "--visualize-all"
    ]
    
    if args.debug:
        visualization_cmd.append("--debug")
    
    logger.info(f"Running visualization command: {' '.join(visualization_cmd)}")
    visualization_result = subprocess.run(visualization_cmd, check=True)
    
    logger.info("Feature importance analysis and visualization completed successfully")


if __name__ == "__main__":
    main()
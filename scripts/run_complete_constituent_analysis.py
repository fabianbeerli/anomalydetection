#!/usr/bin/env python
"""
Complete script to run constituent analysis and create visualizations.
This combines running the analysis and visualization in a single script.
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


def run_constituent_analysis(index_anomalies_file, output_dir):
    """
    Run constituent analysis using the run_constituent_analysis.py script.
    
    Args:
        index_anomalies_file (str): Path to the index anomalies CSV file
        output_dir (str): Directory to save results
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Running constituent analysis...")
        
        # Build command to run the constituent analysis script
        constituent_script = Path(__file__).parent / "run_constituent_analysis.py"
        
        cmd = [
            sys.executable,
            str(constituent_script),
            "--index-anomalies", index_anomalies_file,
            "--output", output_dir
        ]
        
        # Run the command
        result = subprocess.run(cmd, check=True)
        
        logger.info("Constituent analysis completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running constituent analysis: {e}")
        return False


def create_visualizations(results_file, summary_file, output_dir):
    """
    Create visualizations using the visualize_constituent_analysis.py script.
    
    Args:
        results_file (str): Path to the constituent analysis results JSON
        summary_file (str): Path to the constituent analysis summary JSON
        output_dir (str): Directory to save visualizations
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Creating visualizations...")
        
        # Build command to run the visualization script
        visualization_script = Path(__file__).parent / "visualize_constituent_analysis.py"
        
        cmd = [
            sys.executable,
            str(visualization_script),
            "--results", results_file,
            "--summary", summary_file,
            "--output", output_dir
        ]
        
        # Run the command
        result = subprocess.run(cmd, check=True)
        
        logger.info("Visualizations created successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating visualizations: {e}")
        return False


def main():
    """
    Main function to run complete constituent analysis.
    """
    parser = argparse.ArgumentParser(description="Run complete constituent analysis")
    parser.add_argument(
        "--index-anomalies", 
        type=str, 
        default=str(config.DATA_DIR / "subsequence_results/aida/w3_overlap/aida_anomalies.csv"),
        help="Path to the index anomalies CSV file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=str(config.DATA_DIR / "constituent_analysis"),
        help="Directory to save outputs"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = Path(args.output)
    ensure_directory_exists(output_dir)
    
    # Run constituent analysis
    success = run_constituent_analysis(args.index_anomalies, args.output)
    
    if success:
        # Paths to results files
        results_file = output_dir / "constituent_analysis_results.json"
        summary_file = output_dir / "constituent_analysis_summary.json"
        visualization_dir = output_dir / "visualizations"
        
        # Create visualizations if results files exist
        if results_file.exists() and summary_file.exists():
            success = create_visualizations(
                str(results_file), 
                str(summary_file), 
                str(visualization_dir)
            )
            
            if success:
                logger.info("Complete constituent analysis process finished successfully!")
            else:
                logger.warning("Constituent analysis completed, but visualization failed")
        else:
            logger.warning(f"Results files not found at {results_file} and/or {summary_file}")
    else:
        logger.error("Constituent analysis failed")


if __name__ == "__main__":
    main()
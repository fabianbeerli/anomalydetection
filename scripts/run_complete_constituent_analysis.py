#!/usr/bin/env python
"""
Run complete constituent analysis workflow:
1. Run matrix-based constituent analysis
2. Create visualizations
"""
import os
import sys
import subprocess
from pathlib import Path
import logging

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.utils.helpers import ensure_directory_exists

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the complete constituent analysis workflow."""
    # Ensure output directory exists
    constituent_dir = config.DATA_DIR / "constituent_analysis"
    ensure_directory_exists(constituent_dir)
    
    # Step 1: Run constituent analysis
    logger.info("Running constituent analysis...")
    analysis_cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_constituent_analysis.py"),
        "--index-anomalies", str(config.DATA_DIR / "subsequence_results" / "aida" / "w3_overlap" / "aida_anomalies.csv"),
        "--output", str(constituent_dir)
    ]
    
    try:
        subprocess.run(analysis_cmd, check=True)
        logger.info("Constituent analysis completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running constituent analysis: {e}")
        return
    
    # Step 2: Create visualizations
    logger.info("Creating visualizations...")
    viz_cmd = [
        sys.executable,
        str(Path(__file__).parent / "visualize_constituent_analysis.py"),
        "--results", str(constituent_dir / "constituent_analysis_results.json"),
        "--summary", str(constituent_dir / "constituent_analysis_summary.json"),
        "--output", str(constituent_dir / "visualizations")
    ]
    
    try:
        subprocess.run(viz_cmd, check=True)
        logger.info("Visualizations created successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating visualizations: {e}")
        return
    
    logger.info(f"Constituent analysis workflow completed. Results saved in {constituent_dir}")
    logger.info("Check the 'visualizations' subdirectory for visual outputs")
    logger.info("See 'detailed_summary.txt' for a comprehensive text summary")


if __name__ == "__main__":
    main()
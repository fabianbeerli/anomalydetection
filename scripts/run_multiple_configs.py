#!/usr/bin/env python
"""
Script to run anomaly detection algorithms on multiple subsequence configurations.
"""
import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path
import time

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


def run_configuration(subsequence_dir, window_size, overlap, output_dir, algorithms):
    """
    Run a single subsequence configuration.
    
    Args:
        subsequence_dir (str): Directory containing subsequence data
        window_size (int): Size of the subsequence window
        overlap (bool): Whether to use overlapping subsequences
        output_dir (str): Directory to save algorithm results
        algorithms (list): List of algorithms to run
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_subsequence_algorithms.py"),
        "--subsequence-dir", subsequence_dir,
        "--window-size", str(window_size),
        "--output", output_dir,
        "--algorithms"
    ]
    
    # Add algorithms
    cmd.extend(algorithms)
    
    # Add overlap flag
    if overlap:
        cmd.append("--overlap")
    else:
        cmd.append("--no-overlap")
    
    # Log the command
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run the command
    try:
        start_time = time.time()
        result = subprocess.run(cmd, check=True)
        end_time = time.time()
        
        logger.info(f"Configuration w{window_size}_{'overlap' if overlap else 'nonoverlap'} completed in {end_time - start_time:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running configuration: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def run_multiple_configurations(subsequence_dir, configurations, output_dir, algorithms):
    """
    Run multiple subsequence configurations.
    
    Args:
        subsequence_dir (str): Directory containing subsequence data
        configurations (list): List of configuration tuples (window_size, overlap)
        output_dir (str): Directory to save algorithm results
        algorithms (list): List of algorithms to run
        
    Returns:
        dict: Dictionary of results for each configuration
    """
    results = {}
    
    for window_size, overlap in configurations:
        # Create a meaningful configuration name
        config_name = f"w{window_size}_{'overlap' if overlap else 'nonoverlap'}"
        logger.info(f"Running configuration: {config_name}")
        
        success = run_configuration(
            subsequence_dir,
            window_size,
            overlap,
            output_dir,
            algorithms
        )
        
        results[config_name] = success
    
    return results


def main():
    """
    Main function to run multiple subsequence configurations.
    """
    parser = argparse.ArgumentParser(description="Run multiple subsequence configurations")
    parser.add_argument(
        "--subsequence-dir", 
        type=str, 
        default=str(config.PROCESSED_DATA_DIR / "subsequences"),
        help="Directory containing subsequence data"
    )
    parser.add_argument(
        "--window-sizes", 
        type=int, 
        nargs="+",
        default=[3, 5, 10],
        help="Window sizes to run (e.g., 3 5 10)"
    )
    parser.add_argument(
        "--all-overlaps", 
        action="store_true",
        help="Run both overlapping and non-overlapping versions for each window size"
    )
    parser.add_argument(
        "--only-overlap", 
        action="store_true",
        help="Only run overlapping versions"
    )
    parser.add_argument(
        "--only-non-overlap", 
        action="store_true",
        help="Only run non-overlapping versions"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=str(config.DATA_DIR / "subsequence_results"),
        help="Directory to save algorithm results"
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["aida", "iforest", "lof", "all"],
        default=["all"],
        help="Algorithms to run"
    )
    parser.add_argument(
        "--run-comparison",
        action="store_true",
        help="Run comparison script after all configurations complete"
    )
    
    args = parser.parse_args()
    
    # Determine overlap settings
    overlap_settings = []
    if args.all_overlaps:
        overlap_settings = [True, False]
    elif args.only_overlap:
        overlap_settings = [True]
    elif args.only_non_overlap:
        overlap_settings = [False]
    else:
        # Default to overlapping only if nothing specified
        overlap_settings = [True]
    
    # Create configurations
    configurations = []
    for window_size in args.window_sizes:
        for overlap in overlap_settings:
            configurations.append((window_size, overlap))
    
    # Determine which algorithms to run
    algorithms = args.algorithms
    if "all" in algorithms:
        algorithms = ["aida", "iforest", "lof"]
    
    # Run configurations
    logger.info(f"Running {len(configurations)} configurations with algorithms: {', '.join(algorithms)}")
    
    # Ensure output directory exists
    output_dir = Path(args.output)
    ensure_directory_exists(output_dir)
    
    # Run all configurations
    results = run_multiple_configurations(
        args.subsequence_dir,
        configurations,
        args.output,
        algorithms
    )
    
    # Print summary
    logger.info("\nConfiguration execution summary:")
    for config, success in results.items():
        logger.info(f"{config}: {'Success' if success else 'Failed'}")
    
    # Run comparison script if requested
    if args.run_comparison and any(results.values()):
        logger.info("Running comparison script")
        
        # Prepare configuration names for comparison
        config_names = [config for config, success in results.items() if success]
        
        # Build command
        comparison_cmd = [
            sys.executable,
            str(Path(__file__).parent / "compare_subsequence_results.py"),
            "--results-dir", args.output,
            "--configs"
        ]
        comparison_cmd.extend(config_names)
        
        # Add data path
        comparison_cmd.extend([
            "--data", str(config.PROCESSED_DATA_DIR / "index_GSPC_processed.csv"),
            "--output", str(Path(args.output).parent / "subsequence_comparison")
        ])
        
        # Run the comparison
        try:
            logger.info(f"Running command: {' '.join(comparison_cmd)}")
            subprocess.run(comparison_cmd, check=True)
            logger.info("Comparison completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running comparison: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in comparison: {e}")
    
    logger.info("All configurations completed")


if __name__ == "__main__":
    main()
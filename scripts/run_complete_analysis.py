#!/usr/bin/env python
"""
Script to run all three anomaly detection approaches and generate a comparative analysis:
1. S&P 500 only subsequence analysis
2. Matrix-based analysis of S&P 500 and constituents together
3. Individual constituent analysis with S&P 500 cross-referencing

This script orchestrates the entire analysis pipeline and generates comparative visualizations.
"""
import os
import sys
import logging
import argparse
import subprocess
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from matplotlib.dates import DateFormatter
import concurrent.futures

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


def run_subsequence_analysis(window_size, overlap, algorithms):
    """
    Run the S&P 500 subsequence analysis.
    
    Args:
        window_size (int): Window size to use
        overlap (bool): Whether to use overlapping subsequences
        algorithms (list): List of algorithms to run
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Running S&P 500 subsequence analysis (window_size={window_size}, overlap={overlap})")
        
        # Build command
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "run_subsequence_algorithms.py"),
            "--window-size", str(window_size),
            "--algorithms"
        ]
        
        # Add algorithms
        cmd.extend(algorithms)
        
        # Add overlap flag
        if overlap:
            cmd.append("--overlap")
        else:
            cmd.append("--no-overlap")
        
        # Run command
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        logger.info("S&P 500 subsequence analysis completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"S&P 500 subsequence analysis failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Error running S&P 500 subsequence analysis: {e}")
        return False


def run_matrix_analysis(window_size, overlap, algorithms):
    """
    Run the matrix-based analysis of S&P 500 and constituents.
    
    Args:
        window_size (int): Window size to use
        overlap (bool): Whether to use overlapping subsequences
        algorithms (list): List of algorithms to run
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Running matrix-based analysis (window_size={window_size}, overlap={overlap})")
        
        # Build command
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "run_matrix_analysis.py"),
            "--window-size", str(window_size),
            "--algorithms"
        ]
        
        # Add algorithms
        cmd.extend(algorithms)
        
        # Add overlap flag
        if overlap:
            cmd.append("--overlap")
        else:
            cmd.append("--no-overlap")
        
        # Run command
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        logger.info("Matrix-based analysis completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Matrix-based analysis failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Error running matrix-based analysis: {e}")
        return False


def run_constituent_analysis(window_size, overlap, algorithms):
    """
    Run the constituent analysis with S&P 500 cross-referencing.
    
    Args:
        window_size (int): Window size to use
        overlap (bool): Whether to use overlapping subsequences
        algorithms (list): List of algorithms to run
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"Running constituent analysis (window_size={window_size}, overlap={overlap})")
        
        # Build command
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "run_constituent_analysis.py"),
            "--window-size", str(window_size),
            "--algorithms"
        ]
        
        # Add algorithms
        cmd.extend(algorithms)
        
        # Add overlap flag
        if overlap:
            cmd.append("--overlap")
        else:
            cmd.append("--no-overlap")
        
        # Run command
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        logger.info("Constituent analysis completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Constituent analysis failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Error running constituent analysis: {e}")
        return False


def load_execution_times(results_bases, window_size, overlap, algorithms):
    """
    Load execution times for all approaches and algorithms.
    
    Args:
        results_bases (dict): Dictionary mapping approach names to result directories
        window_size (int): Window size used
        overlap (bool): Whether overlapping subsequences were used
        algorithms (list): List of algorithms
        
    Returns:
        dict: Dictionary of execution times
    """
    execution_times = {algorithm: {} for algorithm in algorithms}
    
    overlap_str = "overlap" if overlap else "nonoverlap"
    config_str = f"w{window_size}_{overlap_str}"
    
    try:
        # Load S&P 500 subsequence times
        for algorithm in algorithms:
            if algorithm == "aida":
                time_file = Path(results_bases['subsequence']) / algorithm / config_str / "subsequence_features.csv_AIDA_time.txt"
            else:
                time_file = Path(results_bases['subsequence']) / algorithm / config_str / f"{algorithm}_execution_time.txt"
            
            if time_file.exists():
                with open(time_file, 'r') as f:
                    execution_times[algorithm]['subsequence'] = float(f.read().strip())
            else:
                logger.warning(f"No execution time file found for {algorithm} subsequence analysis")
                execution_times[algorithm]['subsequence'] = None

        
        # Load matrix times
        for algorithm in algorithms:
            if algorithm == "aida":
                time_file = Path(results_bases['matrix']) / algorithm / config_str / "matrix_features.csv_AIDA_time.txt"
            else:
                time_file = Path(results_bases['matrix']) / algorithm / config_str / f"{algorithm}_execution_time.txt"
            
            if time_file.exists():
                with open(time_file, 'r') as f:
                    execution_times[algorithm]['matrix'] = float(f.read().strip())
            else:
                logger.warning(f"No execution time file found for {algorithm} matrix analysis")
                execution_times[algorithm]['matrix'] = None

        
        # Load constituent times
        for algorithm in algorithms:
            time_file = Path(results_bases['constituent']) / config_str / algorithm / "execution_time.txt"
            
            if time_file.exists():
                with open(time_file, 'r') as f:
                    execution_times[algorithm]['constituent'] = float(f.read().strip())
            else:
                logger.warning(f"No execution time file found for {algorithm} constituent analysis")
                execution_times[algorithm]['constituent'] = None
                
        return execution_times
        
    except Exception as e:
        logger.error(f"Error loading execution times: {e}")
        return {}


def load_anomaly_counts(results_bases, window_size, overlap, algorithms):
    """
    Load anomaly counts for all approaches and algorithms.
    
    Args:
        results_bases (dict): Dictionary mapping approach names to result directories
        window_size (int): Window size used
        overlap (bool): Whether overlapping subsequences were used
        algorithms (list): List of algorithms
        
    Returns:
        dict: Dictionary of anomaly counts
    """
    anomaly_counts = {algorithm: {} for algorithm in algorithms}
    
    overlap_str = "overlap" if overlap else "nonoverlap"
    config_str = f"w{window_size}_{overlap_str}"
    
    try:
        # Load S&P 500 subsequence anomaly counts
        for algorithm in algorithms:
            anomalies_file = Path(results_bases['subsequence']) / algorithm / config_str / f"{algorithm}_anomalies.csv"
            
            if anomalies_file.exists():
                anomalies_df = pd.read_csv(anomalies_file)
                anomaly_counts[algorithm]['subsequence'] = len(anomalies_df)
            else:
                logger.warning(f"No anomalies file found for {algorithm} subsequence analysis")
                anomaly_counts[algorithm]['subsequence'] = 0
        
        # Load matrix anomaly counts
        for algorithm in algorithms:
            anomalies_file = Path(results_bases['matrix']) / algorithm / config_str / f"{algorithm}_anomalies.csv"
            
            if anomalies_file.exists():
                anomalies_df = pd.read_csv(anomalies_file)
                anomaly_counts[algorithm]['matrix'] = len(anomalies_df)
            else:
                logger.warning(f"No anomalies file found for {algorithm} matrix analysis")
                anomaly_counts[algorithm]['matrix'] = 0
        
        # Load constituent anomaly counts (S&P 500 anomalies)
        for algorithm in algorithms:
            summary_file = Path(results_bases['constituent']) / config_str / algorithm / f"{algorithm}_constituent_summary.csv"
            
            if summary_file.exists():
                summary_df = pd.read_csv(summary_file)
                anomaly_counts[algorithm]['constituent'] = len(summary_df)
            else:
                logger.warning(f"No summary file found for {algorithm} constituent analysis")
                anomaly_counts[algorithm]['constituent'] = 0
                
        return anomaly_counts
        
    except Exception as e:
        logger.error(f"Error loading anomaly counts: {e}")
        return {}


def create_comparative_visualizations(execution_times, anomaly_counts, window_size, overlap, output_dir):
    """
    Create comparative visualizations of the three approaches.
    
    Args:
        execution_times (dict): Dictionary of execution times
        anomaly_counts (dict): Dictionary of anomaly counts
        window_size (int): Window size used
        overlap (bool): Whether overlapping subsequences were used
        output_dir (Path): Directory to save visualizations
    """
    try:
        # Ensure output directory exists
        ensure_directory_exists(output_dir)
        
        # 1. Execution Time Comparison
        plt.figure(figsize=(12, 8))
        
        # Prepare data for grouped bar chart
        algorithms = list(execution_times.keys())
        approaches = ['subsequence', 'matrix', 'constituent']
        
        x = np.arange(len(algorithms))
        width = 0.25
        
        for i, approach in enumerate(approaches):
            times = [execution_times[algo].get(approach, 0) or 0 for algo in algorithms]
            plt.bar(x + (i - 1) * width, times, width, label=approach.capitalize())
        
        plt.xlabel('Algorithm')
        plt.ylabel('Execution Time (s)')
        plt.title(f'Execution Time Comparison (Window Size: {window_size}, {"Overlap" if overlap else "Non-overlap"})')
        plt.xticks(x, [algo.upper() for algo in algorithms])
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plt.savefig(output_dir / "execution_time_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Anomaly Count Comparison
        plt.figure(figsize=(12, 8))
        
        # Prepare data for grouped bar chart
        for i, approach in enumerate(approaches):
            counts = [anomaly_counts[algo].get(approach, 0) or 0 for algo in algorithms]
            plt.bar(x + (i - 1) * width, counts, width, label=approach.capitalize())
        
        plt.xlabel('Algorithm')
        plt.ylabel('Number of Anomalies')
        plt.title(f'Anomaly Count Comparison (Window Size: {window_size}, {"Overlap" if overlap else "Non-overlap"})')
        plt.xticks(x, [algo.upper() for algo in algorithms])
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plt.savefig(output_dir / "anomaly_count_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Efficiency Comparison (Anomalies per Second)
        plt.figure(figsize=(12, 8))
        
        # Calculate anomalies per second
        efficiency_data = []
        
        for algorithm in algorithms:
            for approach in approaches:
                count = anomaly_counts[algorithm].get(approach, 0) or 0
                time = execution_times[algorithm].get(approach, 0) or 0.001  # Avoid division by zero
                
                efficiency_data.append({
                    'Algorithm': algorithm.upper(),
                    'Approach': approach.capitalize(),
                    'Efficiency': count / time if time > 0 else 0
                })
        
        # Create DataFrame
        efficiency_df = pd.DataFrame(efficiency_data)
        
        # Create grouped bar chart
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Algorithm', y='Efficiency', hue='Approach', data=efficiency_df)
        
        plt.xlabel('Algorithm')
        plt.ylabel('Anomalies per Second')
        plt.title(f'Efficiency Comparison (Window Size: {window_size}, {"Overlap" if overlap else "Non-overlap"})')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plt.savefig(output_dir / "efficiency_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Save summary as a table
        summary_data = []
        
        for algorithm in algorithms:
            for approach in approaches:
                count = anomaly_counts[algorithm].get(approach, 0) or 0
                time = execution_times[algorithm].get(approach, 0) or 0
                efficiency = count / time if time > 0 else 0
                
                summary_data.append({
                    'Algorithm': algorithm.upper(),
                    'Approach': approach.capitalize(),
                    'Anomaly Count': count,
                    'Execution Time (s)': time,
                    'Efficiency (anomalies/s)': efficiency
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_dir / "comparison_summary.csv", index=False)
        
        logger.info(f"Created comparative visualizations in {output_dir}")
        
    except Exception as e:
        logger.error(f"Error creating comparative visualizations: {e}")


def main():
    """
    Main function to run all three anomaly detection approaches and generate a comparative analysis.
    """
    parser = argparse.ArgumentParser(description="Run all three anomaly detection approaches and generate comparative analysis")
    parser.add_argument(
        "--window-size", 
        type=int, 
        default=3,
        help="Size of subsequence window (e.g., 3, 5, 10)"
    )
    parser.add_argument(
        "--overlap", 
        action="store_true",
        help="Use overlapping subsequences"
    )
    parser.add_argument(
        "--no-overlap", 
        action="store_true",
        help="Use non-overlapping subsequences"
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["aida", "iforest", "lof", "all"],
        default=["all"],
        help="Algorithms to run"
    )
    parser.add_argument(
        "--skip-subsequence",
        action="store_true",
        help="Skip S&P 500 subsequence analysis"
    )
    parser.add_argument(
        "--skip-matrix",
        action="store_true",
        help="Skip matrix analysis"
    )
    parser.add_argument(
        "--skip-constituent",
        action="store_true",
        help="Skip constituent analysis"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=str(config.DATA_DIR / "comparative_analysis"),
        help="Directory to save comparative analysis"
    )
    
    args = parser.parse_args()
    
    # Determine which algorithms to run
    algorithms = args.algorithms
    if "all" in algorithms:
        algorithms = ["aida", "iforest", "lof"]
    
    # Determine overlap setting
    overlap = True
    if args.no_overlap:
        overlap = False
    elif args.overlap:
        overlap = True
    
    # Convert output path to Path object
    output_dir = Path(args.output)
    
    # Create a specific output dir for this configuration
    config_dir = output_dir / f"w{args.window_size}_{'overlap' if overlap else 'nonoverlap'}"
    ensure_directory_exists(config_dir)
    
    # Record start time
    start_time = time.time()
    
    # Run each approach (unless skipped)
    results = {}
    
    if not args.skip_subsequence:
        results['subsequence'] = run_subsequence_analysis(args.window_size, overlap, algorithms)
    else:
        logger.info("Skipping S&P 500 subsequence analysis")
        results['subsequence'] = True
    
    if not args.skip_matrix:
        results['matrix'] = run_matrix_analysis(args.window_size, overlap, algorithms)
    else:
        logger.info("Skipping matrix analysis")
        results['matrix'] = True
    
    if not args.skip_constituent:
        results['constituent'] = run_constituent_analysis(args.window_size, overlap, algorithms)
    else:
        logger.info("Skipping constituent analysis")
        results['constituent'] = True
    
    # Record end time
    end_time = time.time()
    total_execution_time = end_time - start_time
    
    # Print summary
    logger.info("\nExecution Summary:")
    for approach, success in results.items():
        logger.info(f"{approach.capitalize()} analysis: {'Success' if success else 'Failed'}")
    logger.info(f"Total execution time: {total_execution_time:.2f} seconds")
    
    # Save total execution time
    with open(config_dir / "total_execution_time.txt", "w") as f:
        f.write(f"{total_execution_time:.2f}")
    
    # Define result base directories
    results_bases = {
        'subsequence': config.DATA_DIR / "subsequence_results",
        'matrix': config.DATA_DIR / "matrix_results",
        'constituent': config.DATA_DIR / "constituent_analysis"
    }
    
    # Load execution times and anomaly counts
    execution_times = load_execution_times(results_bases, args.window_size, overlap, algorithms)
    anomaly_counts = load_anomaly_counts(results_bases, args.window_size, overlap, algorithms)
    
    # Create comparative visualizations
    create_comparative_visualizations(
        execution_times, 
        anomaly_counts, 
        args.window_size, 
        overlap, 
        config_dir
    )
    
    logger.info(f"Comparative analysis completed. Results saved to {config_dir}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Script to run the complete anomaly detection analysis workflow.
Implements all three main workflows:
1. Individual Subsequence Analysis
2. Cross-Index-Constituent Analysis
3. Multi-TS Subsequence Analysis
"""
import os
import sys
import logging
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import time
import subprocess

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data.retrieval import (
    retrieve_sp500_index_data,
    retrieve_constituent_data,
    select_constituent_stocks
)
from src.data.preparation import (
    process_all_data,
    create_subsequence_dataset,
    create_multi_ts_subsequences,
    save_subsequence_dataset
)
from src.models.isolation_forest import IForest
from src.models.lof import LOF
from src.utils.helpers import ensure_directory_exists, load_subsequence, get_file_list

# Import the multi-TS module
from run_matrix_analysis import run_multi_ts_analysis_intrawindow, run_multi_ts_analysis_windowwise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_individual_subsequence_analysis(config_args, ticker):
    """
    Run the individual subsequence analysis workflow.
    
    Args:
        config_args (argparse.Namespace): Configuration arguments
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Starting Individual Subsequence Analysis Workflow")
    
    try:
        # Create output directory for subsequence results
        subsequence_results_dir = Path(config_args.output_dir) / "subsequence_results" / ticker
        ensure_directory_exists(subsequence_results_dir)
        
        # Setup window sizes and overlap settings
        window_sizes = [3]  # Default window sizes
        if config_args.window_sizes:
            window_sizes = [int(size) for size in config_args.window_sizes.split(',')]
        
        overlap_settings = [True, False]  # Default: run both overlapping and non-overlapping
        if config_args.only_overlap:
            overlap_settings = [True]
        elif config_args.only_nonoverlap:
            overlap_settings = [False]
        
        # Define algorithms
        algorithms = ['aida', 'iforest', 'lof']  # Default: run all algorithms
        if config_args.algorithms:
            algorithms = config_args.algorithms.split(',')
        
        # Run for each configuration
        for window_size in window_sizes:
            for overlap in overlap_settings:
                # Skip specific configurations if requested
                if config_args.skip_large_windows and window_size > 5 and overlap:
                    logger.info(f"Skipping large window size {window_size} with overlap")
                    continue
                
                # Build command for run_subsequence_algorithms.py
                cmd = [
                    sys.executable,
                    str(Path(__file__).parent / "run_subsequence_algorithms.py"),
                    "--subsequence-dir", str(Path(config_args.processed_dir) / "subsequences"),
                    "--window-size", str(window_size),
                    "--output", str(subsequence_results_dir),
                    "--algorithms"
                ]
                cmd.extend(algorithms)
                cmd.extend(["--ticker", str(ticker)])
                if overlap:
                    cmd.append("--overlap")
                else:
                    cmd.append("--no-overlap")
                
                logger.info(f"Running command: {' '.join(cmd)}")
                
                # Execute command
                import subprocess
                result = subprocess.run(cmd, check=True)
                
                logger.info(f"Completed individual subsequence analysis for window size {window_size}, overlap={overlap}")
        
        # Run comparison script for visualization
        compare_cmd = [
            sys.executable,
            str(Path(__file__).parent / "compare_subsequence_anomalies.py"),
            "--results-base", str(subsequence_results_dir),
            "--data", str(Path(config_args.processed_dir) / "index_GSPC_processed.csv"),
            "--output", str(Path(config_args.output_dir) / "subsequence_analysis" / ticker),
            "--all-configs"
        ]
        
        logger.info(f"Running comparison: {' '.join(compare_cmd)}")
        subprocess.run(compare_cmd, check=True)
        
        logger.info("Individual Subsequence Analysis Workflow completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in Individual Subsequence Analysis: {e}")
        return False

def run_constituent_anomaly_detection(config_args):
    """
    Run anomaly detection for each constituent ONCE and save all subsequence anomaly results.
    """
    logger.info("Starting Constituent Anomaly Detection Workflow")
    try:
        processed_dir = Path(config_args.processed_dir)
        subsequences_dir = processed_dir / "subsequences"
        output_dir = Path(config_args.output_dir) / "constituent_anomaly_results"
        ensure_directory_exists(output_dir)

        window_sizes = [3]
        overlap_types = ['overlap', 'nonoverlap']
        algorithms = ['aida', 'iforest', 'lof']

        tickers = sorted(set(f.name.split('_')[0] for f in subsequences_dir.glob("*_len*_*.*csv")))
        if not tickers:
            logger.error("No constituent subsequence files found. Exiting constituent anomaly detection.")
            return False
        logger.info(f"Found {len(tickers)} constituent tickers with subsequence files")

        for window_size in window_sizes:
            for overlap_type in overlap_types:
                for algo in algorithms:
                    results_dir = output_dir / algo / f"w{window_size}_{overlap_type}"
                    ensure_directory_exists(results_dir)
                    for ticker in tickers:
                        try:
                            subseq_files = sorted(
                                subsequences_dir.glob(f"{ticker}_len{window_size}_{overlap_type}_*.csv"),
                                key=lambda f: int(f.stem.split('_')[-1])
                            )
                            if not subseq_files:
                                logger.warning(f"No subsequence files for {ticker} in {window_size}-{overlap_type}")
                                continue

                            feature_vectors = []
                            for f in subseq_files:
                                df = pd.read_csv(f)
                                numeric_cols = df.select_dtypes(include=[np.number]).columns
                                feature_vector = df[numeric_cols].values.flatten()
                                feature_vectors.append(feature_vector)
                            feature_matrix = np.array(feature_vectors)

                            # Run AIDA (or other algorithm) on the full matrix
                            # Replace with your real AIDA/IForest/LOF call:
                            aida_scores = feature_matrix.sum(axis=1)  # Dummy
                            threshold = aida_scores.mean() + 2 * aida_scores.std()
                            is_anomaly = aida_scores > threshold

                            # Save all results for this ticker
                            results_df = pd.DataFrame({
                                "subseq_index": np.arange(len(aida_scores)),
                                "score": aida_scores,
                                "is_anomaly": is_anomaly
                            })
                            results_file = results_dir / f"{ticker}_anomaly_results.csv"
                            results_df.to_csv(results_file, index=False)
                        except Exception as ex:
                            logger.error(f"Error on {ticker}: {ex}")

        logger.info("Constituent Anomaly Detection Workflow completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in Constituent Anomaly Detection: {e}")
        return False
    
def run_constituent_cross_analysis_workflow(config_args):
    """
    Calls constituent_analysis.py to perform cross-index analysis and visualization.
    """
    logger.info("Starting Constituent Cross-Analysis Workflow")
    try:
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "constituent_analysis.py"),
            "--subsequence-results-dir", str(Path(config_args.output_dir) / "subsequence_results"),
            "--output-dir", str(Path(config_args.output_dir) / "constituent_cross_analysis"),
            "--processed-dir", str(config_args.processed_dir)
        ]
        if config_args.window_sizes:
            cmd.extend(["--window-sizes", config_args.window_sizes])
        if config_args.only_overlap:
            cmd.append("--only-overlap")
        elif config_args.only_nonoverlap:
            cmd.append("--only-nonoverlap")
        if config_args.algorithms:
            cmd.extend(["--algorithms", config_args.algorithms])

        logger.info(f"Running constituent cross-analysis: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info("Constituent Cross-Analysis Workflow completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in Constituent Cross-Analysis Workflow: {e}")
        return False
    
def run_feature_importance_analysis_workflow(config_args):
    logger.info("Starting Feature Importance Analysis Workflow")
    try:
        feature_importance_results_dir = Path(config_args.output_dir) / "feature_importance_results"
        ensure_directory_exists(feature_importance_results_dir)

        # Find all tickers in subsequence_results
        subsequence_results_dir = Path(config_args.output_dir) / "subsequence_results"
        tickers = [d.name for d in subsequence_results_dir.iterdir() if d.is_dir()]
        if hasattr(config_args, 'ticker') and config_args.ticker:
            tickers = [config_args.ticker]

        for ticker in tickers:
            cmd = [
                sys.executable,
                str(Path(__file__).parent / "run_feature_importance_analysis.py"),
                "--output-dir", str(feature_importance_results_dir),
                "--ticker", ticker
            ]
            if hasattr(config_args, 'window_sizes') and config_args.window_sizes:
                cmd.extend(["--window-sizes", config_args.window_sizes])
            if hasattr(config_args, 'only_overlap') and config_args.only_overlap:
                cmd.append("--only-overlap")
            elif hasattr(config_args, 'only_nonoverlap') and config_args.only_nonoverlap:
                cmd.append("--only-nonoverlap")
            logger.info(f"Running feature importance analysis command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

        # Visualization (can stay as is)
        vis_cmd = [
            sys.executable,
            str(Path(__file__).parent / "visualize_feature_importance_results.py"),
            "--feature-importance-dir", str(feature_importance_results_dir),
            "--output-dir", str(Path(config_args.output_dir) / "feature_importance_visualization"),
            "--visualize-all"
        ]
        logger.info(f"Running feature importance visualization command: {' '.join(vis_cmd)}")
        subprocess.run(vis_cmd, check=True)

        logger.info("Feature Importance Analysis Workflow completed successfully")
        return True

    except Exception as e:
        logger.error(f"Error in Feature Importance Analysis Workflow: {e}")
        return False
    
def run_tix_analysis_workflow(config_args):
    """
    Run TIX analysis workflow to explain anomalies detected by AIDA.
    
    Args:
        config_args (argparse.Namespace): Configuration arguments
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Starting TIX Analysis Workflow")
    
    try:
        # Create output directory for TIX results
        tix_results_dir = Path(config_args.output_dir) / "tix_results"
        ensure_directory_exists(tix_results_dir)
        
        # Add debugging information to verify paths
        logger.info(f"TIX results will be saved to: {tix_results_dir}")
        
        # Setup window sizes and overlap settings for subsequence analysis
        window_sizes = [3]  # Default window sizes
        if config_args.window_sizes:
            window_sizes = [int(size) for size in config_args.window_sizes.split(',')]
        
        overlap_settings = []
        if not config_args.only_overlap and not config_args.only_nonoverlap:
            overlap_settings = ["overlap", "nonoverlap"]
        elif config_args.only_overlap:
            overlap_settings = ["overlap"]
        elif config_args.only_nonoverlap:
            overlap_settings = ["nonoverlap"]
        
        # Determine TIX analysis types to run
        tix_analysis_types = []
        if config_args.run_all or (not config_args.tix_subsequence_only and not config_args.tix_constituent_only and not config_args.tix_multi_ts_only):
            tix_analysis_types = ["--run-all-tix"]
        else:
            if config_args.tix_subsequence_only:
                tix_analysis_types.append("--run-subsequence-tix")
            if config_args.tix_constituent_only:
                tix_analysis_types.append("--run-constituent-tix")
            if config_args.tix_multi_ts_only:
                tix_analysis_types.append("--run-multi-ts-tix")
        
        # Enable debug mode for more detailed logging
        tix_analysis_types.append("--debug")
        
        # Run TIX analysis script
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "run_tix_analysis.py"),
            "--output-dir", str(tix_results_dir)
        ]
        
        # Add analysis types
        cmd.extend(tix_analysis_types)
        
        # Add window sizes
        if config_args.window_sizes:
            cmd.extend(["--window-sizes", config_args.window_sizes])
        
        # Add overlap flags
        if config_args.only_overlap:
            cmd.append("--only-overlap")
        elif config_args.only_nonoverlap:
            cmd.append("--only-nonoverlap")
        
        # Add constituent list if specified
        if hasattr(config_args, 'constituents') and config_args.constituents:
            cmd.extend(["--constituents", config_args.constituents])
        
        logger.info(f"Running TIX analysis command: {' '.join(cmd)}")
        
        # Execute command
        result = subprocess.run(cmd, check=True)
        
        # Run TIX visualization
        vis_cmd = [
            sys.executable,
            str(Path(__file__).parent / "visualize_tix_results.py"),
            "--tix-results-dir", str(tix_results_dir),
            "--output-dir", str(Path(config_args.output_dir) / "tix_visualizations"),
            "--sp500-data", str(Path(config_args.processed_dir) / "index_GSPC_processed.csv"),
            "--visualize-all"
        ]
        
        logger.info(f"Running TIX visualization command: {' '.join(vis_cmd)}")
        
        # Execute visualization command
        vis_result = subprocess.run(vis_cmd, check=True)
        
        logger.info("TIX Analysis Workflow completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error in TIX Analysis Workflow: {e}")
        return False



def run_multi_ts_analysis_workflow_intrawindow(config_args):
    """
    Run the Multi-TS Subsequence Analysis workflow.
    This workflow treats the entire matrix of stocks as a single entity
    and detects when the collective behavior is anomalous.
    
    Args:
        config_args (argparse.Namespace): Configuration arguments
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Starting Multi-TS Subsequence Analysis Workflow")
    
    try:
        # Create output directory
        multi_ts_results_dir = Path(config_args.output_dir) / "multi_ts_results"
        ensure_directory_exists(multi_ts_results_dir)
        
        # Setup window sizes and overlap settings
        window_sizes = [3]  # Default window sizes for multi-TS
        if config_args.multi_ts_window_sizes:
            window_sizes = [int(size) for size in config_args.multi_ts_window_sizes.split(',')]
        
        # Changed default behavior: Always run both unless specifically told otherwise
        overlap_settings = [True, False]  # Default: run both
        if config_args.multi_ts_nonoverlap:
            overlap_settings = [False]  # Only run non-overlapping if specifically requested
        
        
        # Define algorithms - now including AIDA
        algorithms = ['aida', 'iforest', 'lof']  # Added AIDA to default algorithms
        if config_args.multi_ts_algorithms:
            algorithms = config_args.multi_ts_algorithms.split(',')
            
        # Get paths
        multi_ts_dir = Path(config_args.processed_dir) / "multi_ts"
        
        # Check if multi-TS data exists
        if not multi_ts_dir.exists():
            logger.error(f"Multi-TS directory {multi_ts_dir} does not exist")
            return False
        
        # Run for each configuration
        all_results = {}
        
        for window_size in window_sizes:
            for overlap in overlap_settings:
                logger.info(f"Running Multi-TS analysis with window size {window_size}, overlap={overlap}")
                
                # Run multi-TS analysis
                config_name = f"w{window_size}_{'overlap' if overlap else 'nonoverlap'}"
                result = run_multi_ts_analysis_intrawindow(
                    multi_ts_dir=multi_ts_dir,
                    output_dir=multi_ts_results_dir,
                    window_size=window_size,
                    overlap=overlap,
                    algorithms=algorithms
                )
                
                all_results[config_name] = result
                logger.info(f"Completed Multi-TS analysis for {config_name}")
        
        # Create a JSON-serializable version of the results
        json_results = {}
        for config_name, result in all_results.items():
            if isinstance(result, dict):
                json_result = {}
                for key, value in result.items():
                    if key == 'algorithms':
                        json_result[key] = {}
                        for algo, algo_value in value.items():
                            if isinstance(algo_value, dict) and 'files' in algo_value:
                                algo_value['files'] = {k: str(v) for k, v in algo_value['files'].items()}
                            json_result[key][algo] = algo_value
                    else:
                        json_result[key] = value
                json_results[config_name] = json_result
            else:
                json_results[config_name] = result

        # Save overall results
        with open(multi_ts_results_dir / "all_results.json", 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # After saving all_results.json, add:
        multi_ts_analysis_dir = Path(config_args.output_dir) / "multi_ts_analysis"
        ensure_directory_exists(multi_ts_analysis_dir)

        for window_size in window_sizes:
            for overlap in overlap_settings:
                overlap_str = "overlap" if overlap else "nonoverlap"
                config_output_dir = multi_ts_analysis_dir / f"w{window_size}_{overlap_str}"
                config_output_dir.mkdir(parents=True, exist_ok=True)
                compare_multi_ts_cmd = [
                    sys.executable,
                    str(Path(__file__).parent / "compare_multi_ts_anomalies.py"),
                    "--results-base", str(Path(config_args.output_dir) / "multi_ts_results"),
                    "--output", str(config_output_dir / "intrawindow"),
                    "--window-size", str(window_size),
                    "--overlap-type", overlap_str,
                    "--data", str(Path(config_args.processed_dir) / "index_GSPC_processed.csv"),
                    "--windowlevel", "intrawindow"
                ]
                logger.info(f"Running multi-TS comparison: {' '.join(compare_multi_ts_cmd)}")
                subprocess.run(compare_multi_ts_cmd, check=True)

        logger.info("Multi-TS Subsequence Analysis Workflow completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in Multi-TS Subsequence Analysis: {e}")
        return False

def run_multi_ts_analysis_workflow_windowwise(config_args):
    """
    Run the Multi-TS Subsequence Analysis workflow.
    This workflow treats the entire matrix of stocks as a single entity
    and detects when the collective behavior is anomalous.
    
    Args:
        config_args (argparse.Namespace): Configuration arguments
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Starting Multi-TS Subsequence Analysis Workflow")
    
    try:
        # Create output directory
        multi_ts_results_dir = Path(config_args.output_dir) / "multi_ts_results"
        ensure_directory_exists(multi_ts_results_dir)
        
        # Setup window sizes and overlap settings
        window_sizes = [3]  # Default window sizes for multi-TS
        if config_args.multi_ts_window_sizes:
            window_sizes = [int(size) for size in config_args.multi_ts_window_sizes.split(',')]
        
        # Changed default behavior: Always run both unless specifically told otherwise
        overlap_settings = [True, False]  # Default: run both
        if config_args.multi_ts_nonoverlap:
            overlap_settings = [False]  # Only run non-overlapping if specifically requested
        
        
        # Define algorithms - now including AIDA
        algorithms = ['aida', 'iforest', 'lof']  # Added AIDA to default algorithms
        if config_args.multi_ts_algorithms:
            algorithms = config_args.multi_ts_algorithms.split(',')
            
        # Get paths
        multi_ts_dir = Path(config_args.processed_dir) / "multi_ts"
        
        # Check if multi-TS data exists
        if not multi_ts_dir.exists():
            logger.error(f"Multi-TS directory {multi_ts_dir} does not exist")
            return False
        
        # Run for each configuration
        all_results = {}
        
        for window_size in window_sizes:
            for overlap in overlap_settings:
                logger.info(f"Running Multi-TS analysis with window size {window_size}, overlap={overlap}")
                
                # Run multi-TS analysis
                config_name = f"w{window_size}_{'overlap' if overlap else 'nonoverlap'}"
                result = run_multi_ts_analysis_windowwise(
                    multi_ts_dir=multi_ts_dir,
                    output_dir=multi_ts_results_dir,
                    window_size=window_size,
                    overlap=overlap,
                    algorithms=algorithms
                )
                
                all_results[config_name] = result
                logger.info(f"Completed Multi-TS analysis for {config_name}")
        
        # Create a JSON-serializable version of the results
        json_results = {}
        for config_name, result in all_results.items():
            if isinstance(result, dict):
                json_result = {}
                for key, value in result.items():
                    if key == 'algorithms':
                        json_result[key] = {}
                        for algo, algo_value in value.items():
                            if isinstance(algo_value, dict) and 'files' in algo_value:
                                algo_value['files'] = {k: str(v) for k, v in algo_value['files'].items()}
                            json_result[key][algo] = algo_value
                    else:
                        json_result[key] = value
                json_results[config_name] = json_result
            else:
                json_results[config_name] = result

        # Save overall results
        with open(multi_ts_results_dir / "all_results.json", 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # After saving all_results.json, add:
        multi_ts_analysis_dir = Path(config_args.output_dir) / "multi_ts_analysis"
        ensure_directory_exists(multi_ts_analysis_dir)

        for window_size in window_sizes:
            for overlap in overlap_settings:
                overlap_str = "overlap" if overlap else "nonoverlap"
                config_output_dir = multi_ts_analysis_dir / f"w{window_size}_{overlap_str}"
                config_output_dir.mkdir(parents=True, exist_ok=True)
                compare_multi_ts_cmd = [
                    sys.executable,
                    str(Path(__file__).parent / "compare_multi_ts_anomalies.py"),
                    "--results-base", str(Path(config_args.output_dir) / "multi_ts_results"),
                    "--output", str(config_output_dir / "windowwise"),
                    "--window-size", str(window_size),
                    "--overlap-type", overlap_str,
                    "--data", str(Path(config_args.processed_dir) / "index_GSPC_processed.csv"),
                    "--windowlevel", "windowwise"
                ]
                logger.info(f"Running multi-TS comparison: {' '.join(compare_multi_ts_cmd)}")
                subprocess.run(compare_multi_ts_cmd, check=True)

        logger.info("Multi-TS Subsequence Analysis Workflow completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in Multi-TS Subsequence Analysis: {e}")
        return False

def run_all_ticker_subsequence_analysis(config_args):
    """
    Run subsequence anomaly detection for all or a selected ticker,
    using the same detailed logic as for S&P500.
    """
    processed_dir = Path(config_args.processed_dir)
    subsequences_dir = processed_dir / "subsequences"

    tickers = sorted(set(f.name.split('_')[0] for f in subsequences_dir.glob("*_len*_*.*csv")))
    if config_args.ticker:
        tickers = [config_args.ticker]
    if not tickers:
        logger.error("No tickers found for subsequence analysis.")
        return False

    for ticker in tickers:
        logger.info(f"Ticker now procesing: {ticker}")
        run_individual_subsequence_analysis(config_args, ticker)
    return True

def main():
    """
    Main function to run the complete anomaly detection analysis workflow.
    """
    parser = argparse.ArgumentParser(description="Run complete anomaly detection analysis workflow")
    
    # General arguments
    parser.add_argument(
        "--raw-dir", 
        type=str, 
        default=str(config.RAW_DATA_DIR),
        help="Directory for raw data"
    )
    parser.add_argument(
        "--processed-dir", 
        type=str, 
        default=str(config.PROCESSED_DATA_DIR),
        help="Directory for processed data"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=str(config.DATA_DIR / "analysis_results"),
        help="Directory for analysis results"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Run subsequence analysis for a specific ticker (default: all tickers)"
    )
    
    
    # Workflow selection
    parser.add_argument(
        "--run-individual-analysis", 
        action="store_true",
        help="Run individual subsequence analysis workflow"
    )
    parser.add_argument(
        "--run-cross-analysis", 
        action="store_true",
        help="Run cross-index-constituent analysis workflow"
    )
    parser.add_argument(
        "--run-multi-ts-analysis", 
        action="store_true",
        help="Run multi-TS subsequence analysis workflow"
    )
    parser.add_argument(
        "--run-tix-analysis", 
        action="store_true",
        help="Run TIX explanation analysis workflow"
    )
    parser.add_argument(
        "--run-all", 
        action="store_true",
        help="Run all analysis workflows"
    )
    
    # Individual analysis arguments
    parser.add_argument(
        "--window-sizes", 
        type=str,
        help="Comma-separated list of window sizes for individual analysis"
    )
    parser.add_argument(
        "--only-overlap", 
        action="store_true",
        help="Only run with overlapping subsequences"
    )
    parser.add_argument(
        "--only-nonoverlap", 
        action="store_true",
        help="Only run with non-overlapping subsequences"
    )
    parser.add_argument(
        "--algorithms", 
        type=str,
        help="Comma-separated list of algorithms to run"
    )
    parser.add_argument(
        "--skip-large-windows", 
        action="store_true",
        help="Skip large window sizes with overlap to save time"
    )
    
    # Cross analysis arguments
    parser.add_argument(
        "--cross-config", 
        type=str,
        default="w5_overlap",
        help="Configuration to use for cross analysis (e.g., w5_overlap)"
    )
    
    # Multi-TS analysis arguments
    parser.add_argument(
        "--multi-ts-window-sizes", 
        type=str,
        help="Comma-separated list of window sizes for multi-TS analysis"
    )
    parser.add_argument(
        "--multi-ts-algorithms", 
        type=str,
        help="Comma-separated list of algorithms for multi-TS analysis"
    )
    parser.add_argument(
        "--multi-ts-nonoverlap", 
        action="store_true",
        help="Only use non-overlapping subsequences for multi-TS analysis"
    )
    parser.add_argument(
        "--multi-ts-all-overlaps", 
        action="store_true",
        help="Use both overlapping and non-overlapping for multi-TS analysis"
    )
    
    # TIX analysis arguments
    parser.add_argument(
        "--tix-subsequence-only", 
        action="store_true",
        help="Only run TIX analysis on subsequence anomalies"
    )
    parser.add_argument(
        "--tix-constituent-only", 
        action="store_true",
        help="Only run TIX analysis on constituent anomalies"
    )
    parser.add_argument(
        "--tix-multi-ts-only", 
        action="store_true",
        help="Only run TIX analysis on multi-TS anomalies"
    )
    parser.add_argument(
        "--constituents", 
        type=str,
        help="Comma-separated list of constituents to analyze with TIX"
    )
    parser.add_argument(
        "--run-feature-importance", 
        action="store_true",
        help="Run feature importance analysis workflow for LOF and IForest"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    ensure_directory_exists(args.output_dir)
    
    # 3. Run Analysis Workflows
    workflows_to_run = []
    
    if args.run_all:
        workflows_to_run = ["individual", "cross", "multi_ts", "tix",  "feature_importance"]
    else:
        if args.run_individual_analysis:
            workflows_to_run.append("individual")
        if args.run_cross_analysis:
            workflows_to_run.append("cross")
        if args.run_multi_ts_analysis:
            workflows_to_run.append("multi_ts")
        if args.run_tix_analysis:
            workflows_to_run.append("tix")
        if args.run_feature_importance:
            workflows_to_run.append("feature_importance")
    
    if not workflows_to_run:
        logger.warning("No analysis workflows selected. Use --run-all or specific workflow flags.")
        return
    
    workflow_results = {}
    
    # Start workflow timing
    start_time = time.time()

    if "individual" in workflows_to_run:
        if args.ticker is None:
            logger.info("= Starting All Subsequence Analysis Workflow =")
            workflow_results["constituent_detection"] = run_all_ticker_subsequence_analysis(args)
        else:
            logger.info("= Starting Individual Subsequence Analysis Workflow =")
            workflow_results["individual"] = run_individual_subsequence_analysis(args, args.ticker)
            
    if "cross" in workflows_to_run:
        logger.info("= Starting Constituent Cross-Analysis Workflow =")
        workflow_results["constituent_cross"] = run_constituent_cross_analysis_workflow(args)
        
    if "multi_ts" in workflows_to_run:
        logger.info("= Starting Multi-TS Subsequence Analysis Workflow =")
        workflow_results["multi_ts"] = run_multi_ts_analysis_workflow_intrawindow(args)
        logger.info("= Starting Multi-TS Subsequence Analysis Workflow =")
        workflow_results["multi_ts"] = run_multi_ts_analysis_workflow_windowwise(args)

    if "tix" in workflows_to_run:
        logger.info("= Starting TIX Explanation Analysis Workflow =")
        workflow_results["tix"] = run_tix_analysis_workflow(args)

    if "feature_importance" in workflows_to_run:
        logger.info("= Starting Feature Importance Analysis Workflow =")
        workflow_results["feature_importance"] = run_feature_importance_analysis_workflow(args)
    
    # End workflow timing
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("Analysis Workflow Summary")
    logger.info("="*50)
    
    for workflow, success in workflow_results.items():
        status = "Success" if success else "Failed"
        logger.info(f"{workflow.capitalize()} Workflow: {status}")
    
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    logger.info("="*50)


if __name__ == "__main__":
    main()
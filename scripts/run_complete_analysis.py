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
from run_matrix_analysis import run_multi_ts_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_individual_subsequence_analysis(config_args):
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
        subsequence_results_dir = Path(config_args.output_dir) / "subsequence_results"
        ensure_directory_exists(subsequence_results_dir)
        
        # Setup window sizes and overlap settings
        window_sizes = [3, 5, 10]  # Default window sizes
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
                
                # Add algorithms
                cmd.extend(algorithms)
                
                # Add overlap flag
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
            "--output", str(Path(config_args.output_dir) / "subsequence_analysis"),
            "--all-configs"
        ]
        
        logger.info(f"Running comparison: {' '.join(compare_cmd)}")
        subprocess.run(compare_cmd, check=True)
        
        logger.info("Individual Subsequence Analysis Workflow completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in Individual Subsequence Analysis: {e}")
        return False


def run_cross_index_constituent_analysis(config_args):
    """
    Run the Cross-Index-Constituent analysis workflow for all window and overlap configurations.

    Args:
        config_args (argparse.Namespace): Configuration arguments
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Starting Cross-Index-Constituent Analysis Workflow")
    try:
        # Create output directory
        cross_analysis_dir = Path(config_args.output_dir) / "cross_analysis"
        ensure_directory_exists(cross_analysis_dir)

        # Directories for input data
        subsequence_results_dir = Path(config_args.output_dir) / "subsequence_results"
        processed_dir = Path(config_args.processed_dir)

        # Parameters to sweep: two window sizes and include both overlap modes
        window_sizes = [3, 5]
        overlap_types = ['overlap', 'nonoverlap']  # match subsequence-results folder naming (e.g., w3_nonoverlap)
        algorithms = ['aida', 'iforest', 'lof']

        # Load constituent files (exclude index data)
        constituent_files = list(processed_dir.glob("*_processed.csv"))
        constituent_files = [f for f in constituent_files if "index_GSPC" not in f.name]
        if not constituent_files:
            logger.error("No constituent files found. Exiting cross analysis.")
            return False
        logger.info(f"Found {len(constituent_files)} constituent files")

        # Sweep through each configuration
        for window_size in window_sizes:
            for overlap_type in overlap_types:
                logger.info(f"Processing window={window_size}, mode={overlap_type}")

                # Load index anomalies for current config
                index_anomalies = {}
                for algo in algorithms:
                    anomalies_file = (
                        subsequence_results_dir
                        / algo
                        / f"w{window_size}_{overlap_type}"
                        / f"{algo}_anomalies.csv"
                    )
                    if anomalies_file.exists():
                        df = pd.read_csv(anomalies_file)
                        index_anomalies[algo] = df
                        logger.info(f"Loaded {len(df)} anomalies for {algo} in w{window_size}_{overlap_type}")
                    else:
                        logger.warning(f"Missing file for {algo}: {anomalies_file}")

                if not index_anomalies:
                    logger.warning(f"No index anomalies for w{window_size}_{overlap_type}, skipping")
                    continue

                # Analyze each algorithm's anomalies
                for algo, anomalies_df in index_anomalies.items():
                    if anomalies_df.empty:
                        logger.warning(f"No anomalies to process for {algo} in this config")
                        continue

                    # For non-overlap subsequences, step by window size; else step=1
                    step = window_size if overlap_type == 'nonoverlap' else 1

                    for _, anomaly in anomalies_df.iterrows():
                        if not {'start_date', 'end_date', 'index'}.issubset(anomaly.index):
                            logger.warning(f"Incomplete anomaly row for {algo}: {anomaly}")
                            continue

                        start_date = pd.to_datetime(anomaly['start_date'])
                        end_date = pd.to_datetime(anomaly['end_date'])

                        # Create anomaly directory
                        anomaly_dir = (
                            cross_analysis_dir
                            / algo
                            / f"w{window_size}_{overlap_type}"
                            / f"anomaly_{int(anomaly['index'])}"
                        )
                        ensure_directory_exists(anomaly_dir)

                        # Save anomaly metadata
                        info = {
                            'index': int(anomaly['index']),
                            'score': float(anomaly.get('score', np.nan)),
                            'start_date': str(start_date),
                            'end_date': str(end_date),
                            'algorithm': algo,
                            'window_size': window_size,
                            'mode': overlap_type,
                        }
                        with open(anomaly_dir / "anomaly_info.json", 'w') as f:
                            json.dump(info, f, indent=2)

                        constituent_anomalies = {}
                        buffer_days = 5

                        # Iterate top N constituents (or all if fewer)
                        max_const = min(30, len(constituent_files))
                        for fpath in constituent_files[:max_const]:
                            ticker = fpath.stem.replace('_processed', '')
                            try:
                                df = pd.read_csv(fpath, index_col=0, parse_dates=True)
                                period = df.loc[start_date - pd.Timedelta(days=buffer_days):
                                                 end_date + pd.Timedelta(days=buffer_days)]
                                if period.empty:
                                    continue

                                # Build subsequences
                                subs = [period.iloc[i:i+window_size]
                                        for i in range(0, len(period)-window_size+1, step)]
                                if not subs or len(subs) < 3:
                                    continue

                                # Feature extraction
                                feats = [s[['daily_return', 'volume_change', 'high_low_range']].values.flatten()
                                         for s in subs]
                                arr = np.array(feats)

                                # IForest detection
                                iforest = IForest(n_estimators=100,
                                                  max_samples=min(256, arr.shape[0]),
                                                  contamination=0.1)
                                scores, labels = iforest.fit_predict(arr)

                                # Collect overlapping anomalies
                                if np.any(labels == -1):
                                    idxs = np.where(labels == -1)[0]
                                    dates = []
                                    for i in idxs:
                                        s0, e0 = subs[i].index[0], subs[i].index[-1]
                                        if s0 <= end_date and e0 >= start_date:
                                            dates.append((s0, e0))
                                    if dates:
                                        constituent_anomalies[ticker] = {
                                            'ticker': ticker,
                                            'anomaly_count': len(dates),
                                            'max_score': float(np.max(scores[labels==-1])),
                                            'anomaly_dates': [{'start': str(s), 'end': str(e)} for s,e in dates]
                                        }
                            except Exception as ex:
                                logger.error(f"Error on {ticker}: {ex}")

                        # Write results if any constituents flagged
                        if constituent_anomalies:
                            with open(anomaly_dir / "constituent_anomalies.json", 'w') as f:
                                json.dump(constituent_anomalies, f, indent=2)
                            summary = {
                                'total_constituents_analyzed': max_const,
                                'constituents_with_anomalies': len(constituent_anomalies),
                                'anomaly_pattern': 'widespread' if len(constituent_anomalies)>15 else 'isolated',
                                'anomalous_constituents': list(constituent_anomalies.keys())
                            }
                            with open(anomaly_dir / "summary.json", 'w') as f:
                                json.dump(summary, f, indent=2)
                            logger.info(f"Cross-analysis done for {algo} anomaly {int(anomaly['index'])}"
                                        f" (w{window_size}_{overlap_type}), found {len(constituent_anomalies)} tickers")

        logger.info("Cross-Index-Constituent Analysis Workflow completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in Cross-Index-Constituent Analysis: {e}")
        return False





def run_multi_ts_analysis_workflow(config_args):
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
        window_sizes = [3, 5]  # Default window sizes for multi-TS
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
                result = run_multi_ts_analysis(
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
        
        logger.info("Multi-TS Subsequence Analysis Workflow completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in Multi-TS Subsequence Analysis: {e}")
        return False


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
    
    # Data preparation arguments
    parser.add_argument(
        "--skip-data-retrieval", 
        action="store_true",
        help="Skip data retrieval step"
    )
    parser.add_argument(
        "--skip-data-processing", 
        action="store_true",
        help="Skip data processing step"
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
    
    args = parser.parse_args()
    
    # Create output directory
    ensure_directory_exists(args.output_dir)
    
    # 1. Data Retrieval (if not skipped)
    if not args.skip_data_retrieval:
        logger.info("Starting data retrieval")
        
        # Retrieve S&P 500 index data
        sp500_file = retrieve_sp500_index_data()
        if sp500_file:
            logger.info(f"Retrieved S&P 500 index data: {sp500_file}")
        else:
            logger.error("Failed to retrieve S&P 500 index data")
            return
        
        # Retrieve constituent data
        constituents = select_constituent_stocks(top_n=10, additional_n=20)
        constituent_files = retrieve_constituent_data(constituents)
        
        logger.info(f"Retrieved data for {len(constituent_files)} constituent stocks")
    else:
        logger.info("Skipping data retrieval")
    
    # 2. Data Processing (if not skipped)
    if not args.skip_data_processing:
        logger.info("Starting data processing")
        
        # Process all data
        processed_data = process_all_data()
        
        if processed_data.get('sp500_processed'):
            logger.info(f"Processed S&P 500 index data: {processed_data['sp500_processed']}")
        else:
            logger.warning("No processed S&P 500 index data available")
        
        if processed_data.get('constituent_processed'):
            logger.info(f"Processed {len(processed_data['constituent_processed'])} constituent stock files")
        else:
            logger.warning("No processed constituent stock data available")
        
        # Create subsequence datasets
        # This is typically done in prepare_data.py, assumed to be already run
        logger.info("Data processing completed")
    else:
        logger.info("Skipping data processing")
    
    # 3. Run Analysis Workflows
    workflows_to_run = []
    
    if args.run_all:
        workflows_to_run = ["individual", "cross", "multi_ts"]
    else:
        if args.run_individual_analysis:
            workflows_to_run.append("individual")
        if args.run_cross_analysis:
            workflows_to_run.append("cross")
        if args.run_multi_ts_analysis:
            workflows_to_run.append("multi_ts")
    
    if not workflows_to_run:
        logger.warning("No analysis workflows selected. Use --run-all or specific workflow flags.")
        return
    
    workflow_results = {}
    
    # Start workflow timing
    start_time = time.time()
    
    # Run selected workflows
    if "individual" in workflows_to_run:
        logger.info("= Starting Individual Subsequence Analysis Workflow =")
        workflow_results["individual"] = run_individual_subsequence_analysis(args)
    
    if "cross" in workflows_to_run:
        logger.info("= Starting Cross-Index-Constituent Analysis Workflow =")
        workflow_results["cross"] = run_cross_index_constituent_analysis(args)
    
    if "multi_ts" in workflows_to_run:
        logger.info("= Starting Multi-TS Subsequence Analysis Workflow =")
        workflow_results["multi_ts"] = run_multi_ts_analysis_workflow(args)
    
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
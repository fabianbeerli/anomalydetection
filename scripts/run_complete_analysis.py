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
    Run the Cross-Index-Constituent analysis workflow.
    
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
        
        # Load index anomalies
        subsequence_results_dir = Path(config_args.output_dir) / "subsequence_results"
        
        # Use results from window size 5 with overlap by default
        window_size = 5
        overlap_type = "overlap"
        if config_args.cross_config:
            parts = config_args.cross_config.split('_')
            if len(parts) >= 2:
                window_size = int(parts[0][1:])  # Remove 'w' prefix
                overlap_type = parts[1]
        
        # Process each algorithm
        algorithms = ['aida', 'iforest', 'lof']
        
        index_anomalies = {}
        for algo in algorithms:
            # Path to anomalies file
            anomalies_file = subsequence_results_dir / algo / f"w{window_size}_{overlap_type}" / f"{algo}_anomalies.csv"
            
            if anomalies_file.exists():
                anomalies_df = pd.read_csv(anomalies_file)
                index_anomalies[algo] = anomalies_df
                logger.info(f"Loaded {len(anomalies_df)} index anomalies for {algo}")
            else:
                logger.warning(f"No anomalies file found for {algo} at {anomalies_file}")
        
        if not index_anomalies:
            logger.error("No index anomalies found. Exiting cross analysis.")
            return False
        
        # Load constituent data
        processed_dir = Path(config_args.processed_dir)
        constituent_files = list(processed_dir.glob("*_processed.csv"))
        constituent_files = [f for f in constituent_files if "index_GSPC" not in f.name]
        
        if not constituent_files:
            logger.error("No constituent files found. Exiting cross analysis.")
            return False
        
        logger.info(f"Found {len(constituent_files)} constituent files")
        
        # Process top anomalies for each algorithm
        for algo, anomalies_df in index_anomalies.items():
            # Select top 5 anomalies by score
            top_anomalies = anomalies_df.sort_values('score', ascending=False).head(5)
            
            if len(top_anomalies) == 0:
                logger.warning(f"No top anomalies found for {algo}")
                continue
            
            # Process each top anomaly
            for _, anomaly in top_anomalies.iterrows():
                # Extract date range for the anomaly
                if 'start_date' in anomaly and 'end_date' in anomaly:
                    start_date = pd.to_datetime(anomaly['start_date'])
                    end_date = pd.to_datetime(anomaly['end_date'])
                    
                    # Create a directory for this anomaly
                    anomaly_dir = cross_analysis_dir / algo / f"anomaly_{anomaly['index']}"
                    ensure_directory_exists(anomaly_dir)
                    
                    # Save anomaly info
                    with open(anomaly_dir / "anomaly_info.json", 'w') as f:
                        anomaly_info = {
                            'index': int(anomaly['index']),
                            'score': float(anomaly['score']),
                            'start_date': str(start_date),
                            'end_date': str(end_date),
                            'algorithm': algo
                        }
                        json.dump(anomaly_info, f, indent=2)
                    
                    # Analyze constituents for this anomaly
                    constituent_anomalies = {}
                    
                    # Analyze each constituent
                    for constituent_file in constituent_files[:min(30, len(constituent_files))]:
                        ticker = constituent_file.stem.replace('_processed', '')
                        
                        try:
                            # Load constituent data
                            constituent_df = pd.read_csv(constituent_file, index_col=0, parse_dates=True)
                            
                            # Filter to the anomaly period (plus a small buffer)
                            buffer_days = 5
                            buffer_start = start_date - pd.Timedelta(days=buffer_days)
                            buffer_end = end_date + pd.Timedelta(days=buffer_days)
                            
                            period_data = constituent_df.loc[buffer_start:buffer_end]
                            
                            if period_data.empty:
                                logger.warning(f"No data for {ticker} in anomaly period")
                                continue
                            
                            # Create subsequences for this period
                            subsequences = []
                            for i in range(len(period_data) - window_size + 1):
                                subseq = period_data.iloc[i:i+window_size]
                                subsequences.append(subseq)
                            
                            if not subsequences:
                                continue
                            
                            # Flatten subsequences for anomaly detection
                            feature_vectors = []
                            for subseq in subsequences:
                                # Use main price and volume features
                                features = subseq[['daily_return', 'volume_change', 'high_low_range']].values.flatten()
                                feature_vectors.append(features)
                            
                            # Convert to numpy array
                            feature_array = np.array(feature_vectors)
                            
                            # Run anomaly detection (use IForest for all constituents to maintain consistency)
                            iforest = IForest(
                                n_estimators=100, 
                                max_samples=min(256, feature_array.shape[0]),
                                contamination=0.1  # Higher contamination to detect more potential anomalies
                            )
                            
                            # Skip if too few samples
                            if feature_array.shape[0] < 3:
                                continue
                                
                            scores, labels = iforest.fit_predict(feature_array)
                            
                            # Check if any anomalies were detected
                            if np.any(labels == -1):
                                # Get indices of anomalous subsequences
                                anomaly_indices = np.where(labels == -1)[0]
                                
                                # Calculate overlap with index anomaly period
                                anomaly_dates = []
                                for idx in anomaly_indices:
                                    subseq_start = subsequences[idx].index[0]
                                    subseq_end = subsequences[idx].index[-1]
                                    
                                    # Check overlap with the index anomaly period
                                    if (subseq_start <= end_date and subseq_end >= start_date):
                                        anomaly_dates.append((subseq_start, subseq_end))
                                
                                if anomaly_dates:
                                    constituent_anomalies[ticker] = {
                                        'ticker': ticker,
                                        'anomaly_count': len(anomaly_dates),
                                        'max_score': float(max(scores[labels == -1])),
                                        'anomaly_dates': [
                                            {'start': str(start), 'end': str(end)} 
                                            for start, end in anomaly_dates
                                        ]
                                    }
                        
                        except Exception as e:
                            logger.error(f"Error analyzing constituent {ticker}: {e}")
                    
                    # Save constituent anomalies
                    if constituent_anomalies:
                        with open(anomaly_dir / "constituent_anomalies.json", 'w') as f:
                            json.dump(constituent_anomalies, f, indent=2)
                        
                        # Create a summary
                        summary = {
                            'total_constituents_analyzed': len(constituent_files[:min(30, len(constituent_files))]),
                            'constituents_with_anomalies': len(constituent_anomalies),
                            'anomaly_pattern': 'widespread' if len(constituent_anomalies) > 15 else 'isolated',
                            'top_anomalous_constituents': sorted(
                                constituent_anomalies.keys(), 
                                key=lambda x: constituent_anomalies[x]['max_score'],
                                reverse=True
                            )[:5]
                        }
                        
                        with open(anomaly_dir / "summary.json", 'w') as f:
                            json.dump(summary, f, indent=2)
                        
                        logger.info(f"Completed cross analysis for {algo} anomaly {anomaly['index']}")
                        logger.info(f"Found {len(constituent_anomalies)} constituent anomalies")
                        logger.info(f"Pattern: {summary['anomaly_pattern']}")
        
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
        
        overlap_settings = [True]  # Default: only overlapping for multi-TS
        if config_args.multi_ts_nonoverlap:
            overlap_settings = [False]
        elif config_args.multi_ts_all_overlaps:
            overlap_settings = [True, False]
        
        # Define algorithms
        algorithms = ['iforest', 'lof']  # Default: skip AIDA for multi-TS due to complexity
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
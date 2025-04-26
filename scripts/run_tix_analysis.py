#!/usr/bin/env python
"""
Script to run TIX (Tempered Isolation-based eXplanation) analysis on detected anomalies.
This script adds feature importance explanations to anomalies detected by AIDA.
"""
import os
import sys
import logging
import argparse
import json
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.models.tix_helper import TIXAnalyzer
from src.utils.helpers import ensure_directory_exists

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_subsequence_tix_analysis(config_args, ticker=None):
    """
    Run TIX analysis on subsequence anomalies.
    
    Args:
        config_args (argparse.Namespace): Configuration arguments
        ticker (str or None): Ticker symbol to analyze, or None for all tickers
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Running TIX analysis for subsequence anomalies")
        
        # Initialize TIX analyzer
        tix_analyzer = TIXAnalyzer(output_dir=config_args.output_dir)
        
        # Setup parameters
        algorithms = ['aida']  # Currently only supporting AIDA for TIX
        
        window_sizes = [3]  # Default window sizes
        if config_args.window_sizes:
            window_sizes = [int(size) for size in config_args.window_sizes.split(',')]
        
        overlap_types = ['overlap', 'nonoverlap']  # Default: analyze both
        if config_args.only_overlap:
            overlap_types = ['overlap']
        elif config_args.only_nonoverlap:
            overlap_types = ['nonoverlap']
        
        # Add more debugging information
        logger.info(f"Analyzing for algorithms: {algorithms}")
        logger.info(f"Window sizes: {window_sizes}")
        logger.info(f"Overlap types: {overlap_types}")
        
        # Check if subsequence results directory exists
        subsequence_results_dir = Path(config.DATA_DIR) / "analysis_results" / "subsequence_results"
        if not subsequence_results_dir.exists():
            logger.warning(f"Subsequence results directory not found: {subsequence_results_dir}")
        
        # Determine tickers to analyze
        if ticker:
            tickers = [ticker]
        else:
            tickers = [p.name for p in subsequence_results_dir.iterdir() if p.is_dir()]
        
        all_results = {}
        for ticker in tickers:
            for window_size in window_sizes:
                for overlap_type in overlap_types:
                    for algo in algorithms:
                        anomaly_dir = subsequence_results_dir / ticker / algo / f"w{window_size}_{overlap_type}"
                        anomalies_file = anomaly_dir / f"{algo}_anomalies.csv"
                        if not anomalies_file.exists():
                            logger.warning(f"Missing anomaly file: {anomalies_file}")
                            continue
                        tix_results = tix_analyzer.analyze_subsequence_anomalies_for_ticker(
                            ticker=ticker,
                            algorithm=algo,
                            window_size=window_size,
                            overlap_type=overlap_type,
                            anomalies_file=anomalies_file
                        )
                        all_results[f"{ticker}_{algo}_w{window_size}_{overlap_type}"] = tix_results

        # Save summary
        tix_summary_file = Path(config_args.output_dir) / "subsequence_tix_summary.json"
        with open(tix_summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"Subsequence TIX analysis completed. Summary saved to {tix_summary_file}")
        return True
    
    except Exception as e:
        logger.error(f"Error in subsequence TIX analysis: {e}")
        return False




def run_multi_ts_tix_analysis(config_args):
    """
    Run TIX analysis on multi-TS anomalies.
    
    Args:
        config_args (argparse.Namespace): Configuration arguments
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Running TIX analysis for multi-TS anomalies")
        
        # Initialize TIX analyzer
        tix_analyzer = TIXAnalyzer(output_dir=config_args.output_dir)
        
        # Setup parameters
        window_sizes = [3]  # Default window sizes for multi-TS
        if config_args.window_sizes:
            window_sizes = [int(size) for size in config_args.window_sizes.split(',')]
        
        overlap_types = ['overlap', 'nonoverlap']  # Default: analyze both
        if config_args.only_overlap:
            overlap_types = ['overlap']
        elif config_args.only_nonoverlap:
            overlap_types = ['nonoverlap']
        
        # Check if multi-TS directory exists
        multi_ts_dir = Path(config.PROCESSED_DATA_DIR) / "multi_ts"
        if not multi_ts_dir.exists():
            logger.warning(f"Multi-TS directory not found: {multi_ts_dir}")
        
        # Check if multi-TS results directory exists
        multi_ts_results_dir = Path(config.DATA_DIR) / "analysis_results" / "multi_ts_results"
        if not multi_ts_results_dir.exists():
            logger.warning(f"Multi-TS results directory not found: {multi_ts_results_dir}")
        
        # Run TIX analysis
        results = tix_analyzer.analyze_multi_ts_anomalies(
            window_sizes=window_sizes,
            overlap_types=overlap_types
        )
        
        # Save summary
        tix_summary_file = Path(config_args.output_dir) / "multi_ts_tix_summary.json"
        with open(tix_summary_file, 'w') as f:
            # Convert to JSON-serializable format
            json_results = {}
            for config_key, config_results in results.items():
                json_results[config_key] = {
                    'anomalies_analyzed': len(config_results),
                    'anomaly_indices': list(config_results.keys())
                }
            
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Multi-TS TIX analysis completed. Summary saved to {tix_summary_file}")
        return True
    
    except Exception as e:
        logger.error(f"Error in multi-TS TIX analysis: {e}")
        return False


def main():
    """
    Main function to run TIX analysis.
    """
    parser = argparse.ArgumentParser(description="Run TIX analysis on detected anomalies")
    
    # General arguments
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=str(config.DATA_DIR / "tix_results"),
        help="Directory for TIX analysis results"
    )
    
    # Analysis type selection
    parser.add_argument(
        "--run-subsequence-tix", 
        action="store_true",
        help="Run TIX analysis on subsequence anomalies"
    )
    parser.add_argument(
        "--run-multi-ts-tix", 
        action="store_true",
        help="Run TIX analysis on multi-TS anomalies"
    )
    parser.add_argument(
        "--run-all-tix", 
        action="store_true",
        help="Run all TIX analyses"
    )
    
    # Subsequence and multi-TS analysis arguments
    parser.add_argument(
        "--window-sizes", 
        type=str,
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
    
    # Ticker argument for subsequence TIX
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Ticker symbol for which to run subsequence TIX analysis (default: all tickers)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    # Create output directory
    ensure_directory_exists(args.output_dir)
    
    # Determine analyses to run
    analyses_to_run = []
    
    if args.run_all_tix:
        analyses_to_run = ["subsequence", "multi_ts"]
    else:
        if args.run_subsequence_tix:
            analyses_to_run.append("subsequence")
        if args.run_multi_ts_tix:
            analyses_to_run.append("multi_ts")
    
    if not analyses_to_run:
        logger.warning("No TIX analyses selected. Use --run-all-tix or specific analysis flags.")
        return
    
    # Run selected analyses
    analysis_results = {}
    
    if "subsequence" in analyses_to_run:
        logger.info("= Starting Subsequence TIX Analysis =")
        analysis_results["subsequence"] = run_subsequence_tix_analysis(args, ticker=args.ticker)
    
    
    #if "multi_ts" in analyses_to_run:
    #   logger.info("= Starting Multi-TS TIX Analysis =")
    #    analysis_results["multi_ts"] = run_multi_ts_tix_analysis(args)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("TIX Analysis Summary")
    logger.info("="*50)
    
    for analysis, success in analysis_results.items():
        status = "Success" if success else "Failed"
        logger.info(f"{analysis.capitalize()} TIX Analysis: {status}")
    
    logger.info("="*50)


if __name__ == "__main__":
    main()
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




def run_multi_ts_tix_analysis(args, features_csv, label):
    """
    Run TIX analysis for each row (sample) in the multi-TS feature matrix.
    """
    import pandas as pd
    from src.models.tix_helper import TIXAnalyzer

    df = pd.read_csv(features_csv)
    n_samples = df.shape[0]

    tix_analyzer = TIXAnalyzer(output_dir=args.output_dir)
    results = {}

    for idx, row in df.iterrows():
        sample = row.values.astype(float)
        feature_importance = tix_analyzer.analyze_single_sample(
            sample=sample,
            algorithm=getattr(args, "tix_algorithm", "aida")
        )
        results[idx] = feature_importance

    # Save results to file with label
    import json
    output_path = os.path.join(args.output_dir, f"multi_ts_tix_results_{label}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


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
    parser.add_argument(
        "--multi-ts-features-csv",
        type=str,
        required=False,
        default=str(config.DATA_DIR / "analysis_results" / "multi_ts_results" / "multi_ts_w3_overlap" / "multi_ts_features.csv"),
        help="Path to the multi_ts_features.csv file"
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
    
    # Instantiate TIXAnalyzer here
    tix_analyzer = TIXAnalyzer(output_dir=args.output_dir)

    if "subsequence" in analyses_to_run:
        logger.info("= Starting Subsequence TIX Analysis =")
        analysis_results["subsequence"] = run_subsequence_tix_analysis(args, ticker=args.ticker)
    
    
    if "multi_ts" in analyses_to_run:
        logger.info("= Starting Multi-TS TIX Analysis =")
        # Overlap
        overlap_csv = str(config.DATA_DIR / "analysis_results" / "multi_ts_results" / "multi_ts_w3_overlap" / "intrawindow" / "multi_ts_features.csv")
        overlap_anomalies_csv = str(config.DATA_DIR / "analysis_results" / "multi_ts_results" / "multi_ts_w3_overlap" / "intrawindow" / "aida" / "aida_multi_ts_anomalies.csv")
        logger.info("Running for w3_overlap")
        analysis_results["multi_ts_w3_overlap"] = tix_analyzer.analyze_multi_ts_feature_matrix(
            features_csv=overlap_csv,
            anomalies_csv=overlap_anomalies_csv,
            output_dir=os.path.join(args.output_dir, "multi_ts_w3_overlap")
        )
        # Non-overlap
        nonoverlap_csv = str(config.DATA_DIR / "analysis_results" / "multi_ts_results" / "multi_ts_w3_nonoverlap" / "intrawindow" /"multi_ts_features.csv")
        nonoverlap_anomalies_csv = str(config.DATA_DIR / "analysis_results" / "multi_ts_results" / "multi_ts_w3_nonoverlap" / "intrawindow" / "aida" / "aida_multi_ts_anomalies.csv")
        logger.info("Running for w3_nonoverlap")
        analysis_results["multi_ts_w3_nonoverlap"] = tix_analyzer.analyze_multi_ts_feature_matrix(
            features_csv=nonoverlap_csv,
            anomalies_csv=nonoverlap_anomalies_csv,
            output_dir=os.path.join(args.output_dir, "multi_ts_w3_nonoverlap")
        )
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
#!/usr/bin/env python
"""
Script to run AIDA anomaly detection on S&P 500 constituent stocks.
Integrates AIDA with the existing constituent analysis pipeline.
"""
import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import time
import glob

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data.preparation import load_ticker_data
from src.utils.helpers import ensure_directory_exists
from src.models.aida_helper import AIDAConstituentAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_processed_constituents(processed_dir):
    """
    Load processed constituent data.
    
    Args:
        processed_dir (Path): Directory containing processed constituent data
        
    Returns:
        dict: Dictionary mapping tickers to processed data
    """
    ticker_data = {}
    
    # Find all processed constituent files
    constituent_files = list(processed_dir.glob("*_processed.csv"))
    
    if not constituent_files:
        logger.error(f"No processed constituent files found in {processed_dir}")
        return ticker_data
    
    logger.info(f"Found {len(constituent_files)} processed constituent files")
    
    # Load each constituent file
    for file_path in constituent_files:
        try:
            # Extract ticker from filename (remove _processed.csv)
            ticker = file_path.stem.replace("_processed", "")
            
            # Skip index files
            if ticker.startswith("index_"):
                continue
                
            # Load data
            df = load_ticker_data(file_path)
            
            if df is not None and not df.empty:
                ticker_data[ticker] = df
                logger.debug(f"Loaded data for {ticker} with shape {df.shape}")
            else:
                logger.warning(f"Empty or invalid data for {ticker}")
                
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {e}")
    
    logger.info(f"Loaded data for {len(ticker_data)} constituents")
    return ticker_data


def main():
    """
    Main function to run AIDA analysis on constituent stocks.
    """
    parser = argparse.ArgumentParser(
        description="Run AIDA anomaly detection on S&P 500 constituent stocks"
    )
    parser.add_argument(
        "--processed-dir", 
        type=str, 
        default=str(config.PROCESSED_DATA_DIR),
        help="Directory containing processed constituent data"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=str(config.DATA_DIR / "constituent_results"),
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=0,
        help="Limit the number of constituents to analyze (0 for all)"
    )
    parser.add_argument(
        "--tickers", 
        nargs="+",
        default=[],
        help="Specific tickers to analyze (if empty, analyze all available)"
    )
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output)
    
    # Ensure output directory exists
    ensure_directory_exists(output_dir)
    
    # Load processed constituent data
    ticker_data = load_processed_constituents(processed_dir)
    
    if not ticker_data:
        logger.error("No constituent data loaded. Exiting.")
        return 1
    
    # Filter tickers if specified
    if args.tickers:
        filtered_ticker_data = {ticker: data for ticker, data in ticker_data.items() if ticker in args.tickers}
        if not filtered_ticker_data:
            logger.error(f"None of the specified tickers {args.tickers} found in data.")
            return 1
        ticker_data = filtered_ticker_data
        logger.info(f"Filtered to {len(ticker_data)} specified tickers")
    
    # Apply limit if specified
    if args.limit > 0 and args.limit < len(ticker_data):
        # Sort tickers for reproducibility
        sorted_tickers = sorted(ticker_data.keys())
        limited_tickers = sorted_tickers[:args.limit]
        ticker_data = {ticker: ticker_data[ticker] for ticker in limited_tickers}
        logger.info(f"Limited to {len(ticker_data)} tickers")
    
    # Initialize AIDA constituent analyzer
    analyzer = AIDAConstituentAnalyzer(output_dir=output_dir)
    
    # Run AIDA analysis
    logger.info(f"Running AIDA analysis on {len(ticker_data)} constituents...")
    start_time = time.time()
    results = analyzer.analyze_multiple_tickers(ticker_data)
    end_time = time.time()
    
    # Calculate statistics
    successful = sum(1 for result in results.values() if result.get('success', False))
    failed = len(results) - successful
    total_time = end_time - start_time
    
    logger.info(f"AIDA constituent analysis completed in {total_time:.2f} seconds")
    logger.info(f"Successfully analyzed: {successful}/{len(results)} constituents")
    
    if failed > 0:
        logger.warning(f"Failed to analyze {failed} constituents")
        failed_tickers = [ticker for ticker, result in results.items() if not result.get('success', False)]
        logger.warning(f"Failed tickers: {', '.join(failed_tickers[:10])}" + 
                     (f" and {len(failed_tickers) - 10} more" if len(failed_tickers) > 10 else ""))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python
"""
Script to detect anomalies in S&P 500 data using the AIDA C++ implementation.
"""
import os
import sys
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.aida_cpp_wrapper import AIDA, TemporalAIDA
from src.data.preparation import load_ticker_data
from src import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_anomalies_single_stock(stock_data_path, output_dir=None, n_subsamples=20, 
                                 alpha_min=1.0, alpha_max=1.0, metric='manhattan', 
                                 score_type='variance', plot_results=True):
    """
    Detect anomalies in a single stock using AIDA C++ implementation.
    
    Args:
        stock_data_path (str or Path): Path to the processed stock data file
        output_dir (str or Path, optional): Directory to save results
        n_subsamples (int): Number of subsamples for AIDA
        alpha_min (float): Minimum value of alpha
        alpha_max (float): Maximum value of alpha
        metric (str): Distance metric to use
        score_type (str): Score type ('expectation' or 'variance')
        plot_results (bool): Whether to plot the results
    
    Returns:
        dict: Dictionary with anomaly detection results
    """
    logger.info(f"Detecting anomalies in {stock_data_path}")
    
    # Load the data
    df = load_ticker_data(stock_data_path)
    if df is None or df.empty:
        logger.error(f"Failed to load data from {stock_data_path}")
        return None
    
    # Extract stock ticker from file path
    ticker = Path(stock_data_path).stem.replace('_processed', '')
    
    # Select features for anomaly detection
    feature_columns = [
        'daily_return', 
        'log_return',
        'volume_change', 
        'relative_volume',
        'high_low_range',
        'daily_return_zscore',
        'volume_change_zscore'
    ]
    
    # Filter for available features
    available_features = [col for col in feature_columns if col in df.columns]
    
    if not available_features:
        logger.error(f"No required features found in data for {ticker}")
        return None
    
    logger.info(f"Using features: {available_features}")
    
    # Extract the feature data
    X = df[available_features].values
    
    # Initialize and fit AIDA
    aida = AIDA(
        n_subsamples=n_subsamples,
        aggregate_type='aom',
        score_type=score_type,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        metric=metric
    )
    
    aida.fit(X)
    
    # Get anomaly scores
    anomaly_scores = aida.scores_
    
    # Determine anomalies (top 5%)
    threshold = np.percentile(anomaly_scores, 95)
    anomaly_indices = np.where(anomaly_scores > threshold)[0]
    
    # Map anomaly indices to dates
    anomaly_dates = df.index[anomaly_indices]
    
    # Prepare results
    results = {
        'ticker': ticker,
        'anomalies': []
    }
    
    # Process each anomaly
    for i, idx in enumerate(anomaly_indices):
        date = anomaly_dates[i]
        score = anomaly_scores[idx]
        
        # Get feature importances using TIX
        feature_importances = aida.explain(X, idx=idx)
        feature_explanations = {available_features[f]: float(feature_importances[f]) 
                               for f in range(len(available_features))}
        
        # Add to results
        anomaly_data = {
            'date': date.strftime('%Y-%m-%d'),
            'score': float(score),
            'index': int(idx),
            'feature_values': {col: float(df.iloc[idx][col]) for col in available_features},
            'feature_importances': feature_explanations
        }
        
        results['anomalies'].append(anomaly_data)
    
    logger.info(f"Found {len(anomaly_indices)} anomalies for {ticker}")
    
    # Save results if output directory is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save anomaly detection results
        results_path = output_dir / f"{ticker}_anomalies.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved anomaly detection results to {results_path}")
        
        # Save anomaly scores
        scores_path = output_dir / f"{ticker}_scores.csv"
        pd.DataFrame({
            'date': df.index,
            'anomaly_score': anomaly_scores
        }).to_csv(scores_path, index=False)
        
        logger.info(f"Saved anomaly scores to {scores_path}")
    
    # Plot results if requested
    if plot_results:
        # Create output directory for plots
        if output_dir:
            plot_dir = output_dir / 'plots'
            plot_dir.mkdir(exist_ok=True, parents=True)
        else:
            plot_dir = None
        
# Plot anomaly scores
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, anomaly_scores, 'b-')
        plt.scatter(anomaly_dates, anomaly_scores[anomaly_indices], c='r', s=50)
        plt.axhline(y=threshold, color='g', linestyle='--')
        plt.title(f'Anomaly Scores for {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Anomaly Score')
        plt.grid(True, alpha=0.3)
        
        if plot_dir:
            plt.savefig(plot_dir / f"{ticker}_anomaly_scores.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        # Plot price with anomalies
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Close'], 'b-')
        
        # Highlight anomalies
        for date in anomaly_dates:
            plt.axvline(x=date, color='r', alpha=0.3)
        
        plt.title(f'Close Price with Anomalies for {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        
        if plot_dir:
            plt.savefig(plot_dir / f"{ticker}_price_anomalies.png", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    return results


def detect_anomalies_temporal(stock_data_path, output_dir=None, subsequence_length=5, step=1,
                            n_subsamples=20, alpha_min=1.0, alpha_max=1.0, metric='manhattan',
                            score_type='variance', plot_results=True):
    """
    Detect anomalies in time series subsequences using Temporal AIDA.
    
    Args:
        stock_data_path (str or Path): Path to the processed stock data file
        output_dir (str or Path, optional): Directory to save results
        subsequence_length (int): Length of subsequences
        step (int): Step size between subsequences
        n_subsamples (int): Number of subsamples for AIDA
        alpha_min (float): Minimum value of alpha
        alpha_max (float): Maximum value of alpha
        metric (str): Distance metric to use
        score_type (str): Score type ('expectation' or 'variance')
        plot_results (bool): Whether to plot the results
    
    Returns:
        dict: Dictionary with anomaly detection results
    """
    logger.info(f"Detecting temporal anomalies in {stock_data_path}")
    
    # Load the data
    df = load_ticker_data(stock_data_path)
    if df is None or df.empty:
        logger.error(f"Failed to load data from {stock_data_path}")
        return None
    
    # Extract stock ticker from file path
    ticker = Path(stock_data_path).stem.replace('_processed', '')
    
    # Select features for anomaly detection
    feature_columns = [
        'daily_return', 
        'volume_change', 
        'high_low_range'
    ]
    
    # Filter for available features
    available_features = [col for col in feature_columns if col in df.columns]
    
    if not available_features:
        logger.error(f"No required features found in data for {ticker}")
        return None
    
    logger.info(f"Using features: {available_features}")
    
    # Extract the feature data
    X = df[available_features].values
    
    # Initialize and fit Temporal AIDA
    taida = TemporalAIDA(
        subsequence_length=subsequence_length,
        step=step,
        n_subsamples=n_subsamples,
        aggregate_type='aom',
        score_type=score_type,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        metric=metric
    )
    
    taida.fit(X)
    
    # Get anomaly timestamps and scores
    anomaly_timestamps, anomaly_scores = taida.get_anomaly_timestamps(threshold_percentile=95)
    
    # Map subsequence indices to dates
    anomaly_periods = []
    for start_idx, end_idx in anomaly_timestamps:
        start_date = df.index[start_idx]
        end_date = df.index[end_idx]
        anomaly_periods.append((start_date, end_date))
    
    # Prepare results
    results = {
        'ticker': ticker,
        'subsequence_length': subsequence_length,
        'step': step,
        'anomalies': []
    }
    
    # Process each anomaly
    for i, ((start_date, end_date), score) in enumerate(zip(anomaly_periods, anomaly_scores)):
        # Add to results
        anomaly_data = {
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'score': float(score),
            'subsequence_idx': i,
        }
        
        results['anomalies'].append(anomaly_data)
    
    logger.info(f"Found {len(anomaly_periods)} anomalous subsequences for {ticker}")
    
    # Save results if output directory is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save anomaly detection results
        results_path = output_dir / f"{ticker}_temporal_anomalies_L{subsequence_length}_S{step}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved temporal anomaly detection results to {results_path}")
    
    # Plot results if requested
    if plot_results:
        # Create output directory for plots
        if output_dir:
            plot_dir = output_dir / 'plots' / 'temporal'
            plot_dir.mkdir(exist_ok=True, parents=True)
        else:
            plot_dir = None
        
        # Plot price with anomalous subsequences
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['Close'], 'b-')
        
        # Highlight anomalous subsequences
        for start_date, end_date in anomaly_periods:
            plt.axvspan(start_date, end_date, color='r', alpha=0.2)
        
        plt.title(f'Close Price with Anomalous Subsequences for {ticker} (L={subsequence_length}, S={step})')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        
        if plot_dir:
            plt.savefig(plot_dir / f"{ticker}_temporal_anomalies_L{subsequence_length}_S{step}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    return results


def main():
    """
    Main function to detect anomalies using AIDA C++ implementation.
    """
    parser = argparse.ArgumentParser(description='Detect anomalies in S&P 500 data using AIDA C++ implementation')
    parser.add_argument('--mode', type=str, choices=['single', 'temporal'], default='single',
                      help='Detection mode: single (point-based) or temporal (subsequence-based)')
    parser.add_argument('--data', type=str, help='Path to data file or directory')
    parser.add_argument('--output', type=str, default='results/aida_cpp',
                      help='Directory to save results (default: results/aida_cpp)')
    parser.add_argument('--ticker', type=str, help='Ticker symbol (for single mode)')
    parser.add_argument('--sublen', type=int, default=5,
                      help='Subsequence length for temporal mode (default: 5)')
    parser.add_argument('--step', type=int, default=1,
                      help='Step size for temporal mode (default: 1)')
    parser.add_argument('--metric', type=str, default='manhattan',
                      help='Distance metric (default: manhattan)')
    parser.add_argument('--score', type=str, default='variance',
                      help='Score type (default: variance)')
    parser.add_argument('--plot', action='store_true', help='Plot results')
    
    args = parser.parse_args()
    
    # Determine paths based on arguments
    if args.data:
        data_path = Path(args.data)
    else:
        if args.mode == 'single':
            if args.ticker:
                data_path = config.PROCESSED_DATA_DIR / f"{args.ticker}_processed.csv"
            else:
                data_path = config.PROCESSED_DATA_DIR / f"{config.SP500_TICKER.replace('^', 'index_')}_processed.csv"
        else:  # temporal
            if args.ticker:
                data_path = config.PROCESSED_DATA_DIR / f"{args.ticker}_processed.csv"
            else:
                data_path = config.PROCESSED_DATA_DIR / f"{config.SP500_TICKER.replace('^', 'index_')}_processed.csv"
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Starting anomaly detection in {args.mode} mode")
    logger.info(f"Using data from {data_path}")
    logger.info(f"Results will be saved to {output_dir}")
    
    if args.mode == 'single':
        # Single stock detection
        if data_path.is_file():
            results = detect_anomalies_single_stock(
                data_path,
                output_dir=output_dir,
                metric=args.metric,
                score_type=args.score,
                plot_results=args.plot
            )
        else:
            # Process all stocks in the directory
            processed_files = list(Path(data_path).glob('*_processed.csv'))
            logger.info(f"Found {len(processed_files)} processed files")
            
            for file_path in processed_files:
                results = detect_anomalies_single_stock(
                    file_path,
                    output_dir=output_dir,
                    metric=args.metric,
                    score_type=args.score,
                    plot_results=args.plot
                )
    
    else:  # temporal
        # Temporal detection
        if data_path.is_file():
            results = detect_anomalies_temporal(
                data_path,
                output_dir=output_dir,
                subsequence_length=args.sublen,
                step=args.step,
                metric=args.metric,
                score_type=args.score,
                plot_results=args.plot
            )
        else:
            # Process all stocks in the directory
            processed_files = list(Path(data_path).glob('*_processed.csv'))
            logger.info(f"Found {len(processed_files)} processed files")
            
            for file_path in processed_files:
                results = detect_anomalies_temporal(
                    file_path,
                    output_dir=output_dir,
                    subsequence_length=args.sublen,
                    step=args.step,
                    metric=args.metric,
                    score_type=args.score,
                    plot_results=args.plot
                )
    
    logger.info("Anomaly detection completed")


if __name__ == "__main__":
    main()
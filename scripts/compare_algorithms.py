#!/usr/bin/env python
"""
Script to compare AIDA, Isolation Forest, and LOF algorithms on S&P 500 data.
"""
import os
import sys
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import time

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.aida import AIDA, TemporalAIDA, TIX
from src.models.isolation_forest import IsolationForest, TemporalIsolationForest
from src.models.lof import LOF, TemporalLOF
from src.data.preparation import load_ticker_data
from src.utils.helpers import get_file_list
from src import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compare_algorithms(stock_data_path, output_dir=None, mode='point', subsequence_length=5, step=1):
    """
    Compare AIDA, Isolation Forest, and LOF on a single stock.
    
    Args:
        stock_data_path (str or Path): Path to the processed stock data file
        output_dir (str or Path, optional): Directory to save results
        mode (str): 'point' or 'temporal'
        subsequence_length (int): Length of subsequences for temporal mode
        step (int): Step size for temporal mode
    
    Returns:
        dict: Dictionary with comparison results
    """
    logger.info(f"Comparing algorithms on {stock_data_path}")
    
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
    
    # Create output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Run comparison based on mode
    if mode == 'point':
        return compare_point_algorithms(X, df, ticker, available_features, output_dir)
    else:  # temporal
        return compare_temporal_algorithms(X, df, ticker, available_features, output_dir, subsequence_length, step)


def compare_point_algorithms(X, df, ticker, feature_names, output_dir=None):
    """
    Compare point-based anomaly detection algorithms.
    
    Args:
        X (ndarray): Feature matrix
        df (DataFrame): Original dataframe with timestamps
        ticker (str): Stock ticker
        feature_names (list): Names of features
        output_dir (Path, optional): Directory to save results
    
    Returns:
        dict: Comparison results
    """
    logger.info(f"Running point-based comparison for {ticker}")
    
    # Initialize algorithms
    algorithms = {
        'AIDA': AIDA(n_subsamples=20, subsample_size=0.1, metric='cityblock', score_type='variance', random_state=42),
        'Isolation Forest': IsolationForest(n_estimators=100, max_samples='auto', contamination=0.1, random_state=42),
        'LOF': LOF(n_neighbors=20, metric='minkowski', p=2, contamination=0.1)
    }
    
    # Results will contain execution times, anomaly indices, and other metrics
    results = {
        'ticker': ticker,
        'mode': 'point',
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'features': feature_names,
        'date_range': [df.index[0].strftime('%Y-%m-%d'), df.index[-1].strftime('%Y-%m-%d')],
        'algorithms': {}
    }
    
    # For each algorithm
    for name, algorithm in algorithms.items():
        logger.info(f"Running {name}...")
        
        # Measure execution time
        start_time = time.time()
        algorithm.fit(X)
        execution_time = time.time() - start_time
        
        # Get anomaly scores
        scores = algorithm.scores_
        
        # Get feature importances
        if hasattr(algorithm, 'feature_importances_'):
            feature_importances = algorithm.feature_importances_
        else:
            feature_importances = np.zeros(X.shape[1])
        
        # Determine anomalies (top 5%)
        threshold = np.percentile(scores, 95)
        anomaly_indices = np.where(scores > threshold)[0]
        
        # Map anomaly indices to dates
        anomaly_dates = df.index[anomaly_indices].strftime('%Y-%m-%d').tolist()
        
        # Store results
        algorithm_results = {
            'execution_time': execution_time,
            'anomaly_count': len(anomaly_indices),
            'anomaly_indices': anomaly_indices.tolist(),
            'anomaly_dates': anomaly_dates,
            'feature_importances': {feature_names[i]: float(feature_importances[i]) for i in range(len(feature_names))},
            'scores': scores.tolist()
        }
        
        results['algorithms'][name] = algorithm_results
        
        logger.info(f"{name} found {len(anomaly_indices)} anomalies in {execution_time:.2f} seconds")
    
    # Compare algorithm results
    cross_algorithm_comparison = analyze_algorithm_overlap(results)
    results['cross_algorithm_comparison'] = cross_algorithm_comparison
    
    # Save results if output directory is specified
    if output_dir:
        # Save numerical results
        results_path = output_dir / f"{ticker}_algorithm_comparison.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved comparison results to {results_path}")
        
        # Create visualization of results
        plot_dir = output_dir / 'plots'
        plot_dir.mkdir(exist_ok=True, parents=True)
        
        plot_algorithm_comparison(results, df, plot_dir)
    
    return results


def compare_temporal_algorithms(X, df, ticker, feature_names, output_dir=None, subsequence_length=5, step=1):
    """
    Compare temporal anomaly detection algorithms.
    
    Args:
        X (ndarray): Feature matrix
        df (DataFrame): Original dataframe with timestamps
        ticker (str): Stock ticker
        feature_names (list): Names of features
        output_dir (Path, optional): Directory to save results
        subsequence_length (int): Length of subsequences
        step (int): Step size between subsequences
    
    Returns:
        dict: Comparison results
    """
    logger.info(f"Running temporal comparison for {ticker} with subsequence length {subsequence_length}, step {step}")
    
    # Initialize algorithms
    algorithms = {
        'Temporal AIDA': TemporalAIDA(
            subsequence_length=subsequence_length, step=step,
            n_subsamples=20, subsample_size=0.1, metric='cityblock', score_type='variance', random_state=42
        ),
        'Temporal Isolation Forest': TemporalIsolationForest(
            subsequence_length=subsequence_length, step=step,
            n_estimators=100, max_samples='auto', contamination=0.1, random_state=42
        ),
        'Temporal LOF': TemporalLOF(
            subsequence_length=subsequence_length, step=step,
            n_neighbors=20, metric='minkowski', p=2, contamination=0.1
        )
    }
    
    # Results will contain execution times, anomaly indices, and other metrics
    results = {
        'ticker': ticker,
        'mode': 'temporal',
        'subsequence_length': subsequence_length,
        'step': step,
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'features': feature_names,
        'date_range': [df.index[0].strftime('%Y-%m-%d'), df.index[-1].strftime('%Y-%m-%d')],
        'algorithms': {}
    }
    
    # For each algorithm
    for name, algorithm in algorithms.items():
        logger.info(f"Running {name}...")
        
        # Measure execution time
        start_time = time.time()
        algorithm.fit(X)
        execution_time = time.time() - start_time
        
        # Get anomaly timestamps and scores
        anomaly_timestamps, anomaly_scores = algorithm.get_anomaly_timestamps(threshold_percentile=95)
        
        # Map subsequence indices to dates
        anomaly_periods = []
        for start_idx, end_idx in anomaly_timestamps:
            start_date = df.index[start_idx].strftime('%Y-%m-%d')
            end_date = df.index[end_idx].strftime('%Y-%m-%d')
            anomaly_periods.append([start_date, end_date])
        
        # Store results
        algorithm_results = {
            'execution_time': execution_time,
            'anomaly_count': len(anomaly_timestamps),
            'anomaly_timestamps': anomaly_timestamps.tolist(),
            'anomaly_periods': anomaly_periods,
            'scores': algorithm.scores_.tolist()
        }
        
        results['algorithms'][name] = algorithm_results
        
        logger.info(f"{name} found {len(anomaly_timestamps)} anomalous subsequences in {execution_time:.2f} seconds")
    
    # Compare algorithm results
    cross_algorithm_comparison = analyze_temporal_overlap(results)
    results['cross_algorithm_comparison'] = cross_algorithm_comparison
    
    # Save results if output directory is specified
    if output_dir:
        # Save numerical results
        results_path = output_dir / f"{ticker}_temporal_comparison_L{subsequence_length}_S{step}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved temporal comparison results to {results_path}")
        
        # Create visualization of results
        plot_dir = output_dir / 'plots' / 'temporal'
        plot_dir.mkdir(exist_ok=True, parents=True)
        
        plot_temporal_comparison(results, df, plot_dir)
    
    return results


def analyze_algorithm_overlap(results):
    """
    Analyze the overlap in anomalies detected by different algorithms.
    
    Args:
        results (dict): Results from algorithm comparison
    
    Returns:
        dict: Cross-algorithm comparison metrics
    """
    algorithms = list(results['algorithms'].keys())
    n_algorithms = len(algorithms)
    
    # Initialize comparison matrix
    overlap_matrix = np.zeros((n_algorithms, n_algorithms))
    
    # Initialize sets of anomaly indices
    anomaly_sets = {}
    for name in algorithms:
        anomaly_sets[name] = set(results['algorithms'][name]['anomaly_indices'])
    
    # Compute overlap between each pair of algorithms
    for i, algo1 in enumerate(algorithms):
        for j, algo2 in enumerate(algorithms):
            # Skip diagonal (self-comparison)
            if i == j:
                overlap_matrix[i, j] = 1.0
                continue
            
            # Compute Jaccard similarity coefficient
            intersection = len(anomaly_sets[algo1].intersection(anomaly_sets[algo2]))
            union = len(anomaly_sets[algo1].union(anomaly_sets[algo2]))
            
            if union > 0:
                overlap_matrix[i, j] = intersection / union
    
    # Compute common anomalies across all algorithms
    common_anomalies = set.intersection(*anomaly_sets.values())
    
    # Compute unique anomalies for each algorithm
    unique_anomalies = {}
    for name in algorithms:
        others = set.union(*[anomaly_sets[a] for a in algorithms if a != name])
        unique_anomalies[name] = anomaly_sets[name] - others
    
    return {
        'overlap_matrix': overlap_matrix.tolist(),
        'algorithm_names': algorithms,
        'common_anomalies': sorted(list(common_anomalies)),
        'common_anomaly_count': len(common_anomalies),
        'unique_anomalies': {name: sorted(list(indices)) for name, indices in unique_anomalies.items()},
        'unique_anomaly_counts': {name: len(indices) for name, indices in unique_anomalies.items()}
    }


def analyze_temporal_overlap(results):
    """
    Analyze the overlap in temporal anomalies detected by different algorithms.
    
    Args:
        results (dict): Results from temporal algorithm comparison
    
    Returns:
        dict: Cross-algorithm comparison metrics
    """
    algorithms = list(results['algorithms'].keys())
    n_algorithms = len(algorithms)
    
    # Initialize comparison matrix
    overlap_matrix = np.zeros((n_algorithms, n_algorithms))
    
    # Initialize sets of anomaly timestamps (as tuples for hashability)
    anomaly_sets = {}
    for name in algorithms:
        anomaly_sets[name] = set(tuple(ts) for ts in results['algorithms'][name]['anomaly_timestamps'])
    
    # Compute overlap between each pair of algorithms
    for i, algo1 in enumerate(algorithms):
        for j, algo2 in enumerate(algorithms):
            # Skip diagonal (self-comparison)
            if i == j:
                overlap_matrix[i, j] = 1.0
                continue
            
            # Compute Jaccard similarity coefficient
            intersection = len(anomaly_sets[algo1].intersection(anomaly_sets[algo2]))
            union = len(anomaly_sets[algo1].union(anomaly_sets[algo2]))
            
            if union > 0:
                overlap_matrix[i, j] = intersection / union
    
    # Compute common anomalies across all algorithms
    common_anomalies = set.intersection(*anomaly_sets.values())
    
    # Compute unique anomalies for each algorithm
    unique_anomalies = {}
    for name in algorithms:
        others = set.union(*[anomaly_sets[a] for a in algorithms if a != name])
        unique_anomalies[name] = anomaly_sets[name] - others
    
    return {
        'overlap_matrix': overlap_matrix.tolist(),
        'algorithm_names': algorithms,
        'common_anomalies': [list(ts) for ts in sorted(common_anomalies)],
        'common_anomaly_count': len(common_anomalies),
        'unique_anomalies': {name: [list(ts) for ts in sorted(indices)] for name, indices in unique_anomalies.items()},
        'unique_anomaly_counts': {name: len(indices) for name, indices in unique_anomalies.items()}
    }


def plot_algorithm_comparison(results, df, plot_dir):
    """
    Create visualizations comparing algorithm results.
    
    Args:
        results (dict): Results from algorithm comparison
        df (DataFrame): Original dataframe with timestamps
        plot_dir (Path): Directory to save plots
    """
    ticker = results['ticker']
    algorithms = list(results['algorithms'].keys())
    
    # 1. Plot stock price with anomalies from each algorithm
    plt.figure(figsize=(14, 8))
    plt.plot(df.index, df['Close'], 'k-', alpha=0.7, label='Close Price')
    
    # Use different markers for each algorithm
    markers = ['o', 's', '^']
    
    for i, algo_name in enumerate(algorithms):
        # Get anomaly dates
        anomaly_indices = results['algorithms'][algo_name]['anomaly_indices']
        
        # Plot anomalies
        plt.scatter(
            df.index[anomaly_indices],
            df['Close'].iloc[anomaly_indices],
            marker=markers[i % len(markers)],
            label=f"{algo_name} Anomalies",
            alpha=0.7,
            s=80
        )
    
    plt.title(f'Anomalies Detected in {ticker} by Different Algorithms')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_dir / f"{ticker}_algorithm_anomalies.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot anomaly scores for each algorithm
    plt.figure(figsize=(14, 10))
    
    for i, algo_name in enumerate(algorithms):
        plt.subplot(len(algorithms), 1, i+1)
        
        # Get scores
        scores = results['algorithms'][algo_name]['scores']
        
        # Plot scores
        plt.plot(df.index, scores, label=f"{algo_name} Scores")
        
        # Plot threshold line
        threshold = np.percentile(scores, 95)
        plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label='Threshold (95%)')
        
        # Plot anomalies
        anomaly_indices = results['algorithms'][algo_name]['anomaly_indices']
        plt.scatter(df.index[anomaly_indices], [scores[i] for i in anomaly_indices], 
                   color='red', s=50, alpha=0.7)
        
        plt.title(f'{algo_name} Anomaly Scores for {ticker}')
        plt.ylabel('Anomaly Score')
        if i == len(algorithms) - 1:
            plt.xlabel('Date')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir / f"{ticker}_algorithm_scores.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Plot feature importances comparison
    plt.figure(figsize=(12, 8))
    
    feature_names = results['features']
    x = np.arange(len(feature_names))
    width = 0.25
    
    # Plot bars for each algorithm
    for i, algo_name in enumerate(algorithms):
        if 'feature_importances' in results['algorithms'][algo_name]:
            importances = [results['algorithms'][algo_name]['feature_importances'][f] for f in feature_names]
            plt.bar(x + (i - 1) * width, importances, width, label=algo_name)
    
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(f'Feature Importance Comparison for {ticker}')
    plt.xticks(x, feature_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f"{ticker}_feature_importances.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Plot overlap matrix as a heatmap
    if 'cross_algorithm_comparison' in results and 'overlap_matrix' in results['cross_algorithm_comparison']:
        plt.figure(figsize=(10, 8))
        overlap_matrix = np.array(results['cross_algorithm_comparison']['overlap_matrix'])
        
        sns.heatmap(
            overlap_matrix,
            annot=True,
            fmt='.2f',
            cmap='YlGnBu',
            xticklabels=algorithms,
            yticklabels=algorithms
        )
        
        plt.title(f'Algorithm Overlap for {ticker} (Jaccard Similarity)')
        plt.tight_layout()
        plt.savefig(plot_dir / f"{ticker}_algorithm_overlap.png", dpi=300, bbox_inches='tight')
        plt.close()


def plot_temporal_comparison(results, df, plot_dir):
    """
    Create visualizations comparing temporal algorithm results.
    
    Args:
        results (dict): Results from temporal algorithm comparison
        df (DataFrame): Original dataframe with timestamps
        plot_dir (Path): Directory to save plots
    """
    ticker = results['ticker']
    algorithms = list(results['algorithms'].keys())
    subsequence_length = results['subsequence_length']
    step = results['step']
    
    # 1. Plot stock price with anomalous subsequences from each algorithm
    plt.figure(figsize=(14, 8))
    plt.plot(df.index, df['Close'], 'k-', alpha=0.7, label='Close Price')
    
    # Use different colors for each algorithm
    colors = ['r', 'g', 'b']
    
    for i, algo_name in enumerate(algorithms):
        # Get anomaly periods
        anomaly_periods = results['algorithms'][algo_name]['anomaly_periods']
        
        # Plot anomalous periods
        for j, (start_date_str, end_date_str) in enumerate(anomaly_periods):
            # Convert strings to datetime objects
            start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            
            # Add a colored span for this anomalous period
            plt.axvspan(
                start_date, end_date,
                alpha=0.2,
                color=colors[i % len(colors)],
                label=f"{algo_name} Anomaly" if j == 0 else ""
            )
    
    plt.title(f'Anomalous Subsequences Detected in {ticker} by Different Algorithms (L={subsequence_length}, S={step})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_dir / f"{ticker}_temporal_anomalies_L{subsequence_length}_S{step}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot anomaly scores for each algorithm
    # Since the subsequences don't directly map to the original time index,
    # we'll plot them against subsequence indices instead
    plt.figure(figsize=(14, 10))
    
    for i, algo_name in enumerate(algorithms):
        plt.subplot(len(algorithms), 1, i+1)
        
        # Get scores
        scores = results['algorithms'][algo_name]['scores']
        
        # Plot scores
        plt.plot(range(len(scores)), scores, label=f"{algo_name} Scores")
        
        # Plot threshold line
        threshold = np.percentile(scores, 95)
        plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label='Threshold (95%)')
        
        # Get anomaly indices from timestamps
        anomaly_indices = []
        for ts in results['algorithms'][algo_name]['anomaly_timestamps']:
            # Calculate subsequence index based on start_idx and step
            if isinstance(ts, list):
                start_idx = ts[0]
            else:
                start_idx = ts
            subseq_idx = start_idx // step if step > 0 else 0
            anomaly_indices.append(subseq_idx)
        
        # Plot anomalies
        plt.scatter(anomaly_indices, [scores[i] for i in anomaly_indices], 
                   color='red', s=50, alpha=0.7)
        
        plt.title(f'{algo_name} Anomaly Scores for {ticker} (L={subsequence_length}, S={step})')
        plt.ylabel('Anomaly Score')
        if i == len(algorithms) - 1:
            plt.xlabel('Subsequence Index')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir / f"{ticker}_temporal_scores_L{subsequence_length}_S{step}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Plot overlap matrix as a heatmap
    if 'cross_algorithm_comparison' in results and 'overlap_matrix' in results['cross_algorithm_comparison']:
        plt.figure(figsize=(10, 8))
        overlap_matrix = np.array(results['cross_algorithm_comparison']['overlap_matrix'])
        
        sns.heatmap(
            overlap_matrix,
            annot=True,
            fmt='.2f',
            cmap='YlGnBu',
            xticklabels=algorithms,
            yticklabels=algorithms
        )
        
        plt.title(f'Temporal Algorithm Overlap for {ticker} (Jaccard Similarity)')
        plt.tight_layout()
        plt.savefig(plot_dir / f"{ticker}_temporal_overlap_L{subsequence_length}_S{step}.png", dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """
    Main function to compare anomaly detection algorithms.
    """
    parser = argparse.ArgumentParser(description='Compare anomaly detection algorithms on S&P 500 data')
    parser.add_argument('--mode', type=str, choices=['point', 'temporal'], default='point',
                      help='Detection mode: point-based or temporal')
    parser.add_argument('--data', type=str, help='Path to data file or directory')
    parser.add_argument('--output', type=str, default='results/comparison',
                      help='Directory to save results (default: results/comparison)')
    parser.add_argument('--ticker', type=str, help='Ticker symbol')
    parser.add_argument('--sublen', type=int, default=5,
                      help='Subsequence length for temporal mode (default: 5)')
    parser.add_argument('--step', type=int, default=1,
                      help='Step size for temporal mode (default: 1)')
    
    args = parser.parse_args()
    
    # Determine paths based on arguments
    if args.data:
        data_path = Path(args.data)
    else:
        if args.ticker:
            data_path = config.PROCESSED_DATA_DIR / f"{args.ticker}_processed.csv"
        else:
            data_path = config.PROCESSED_DATA_DIR / f"{config.SP500_TICKER.replace('^', 'index_')}_processed.csv"
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Starting algorithm comparison in {args.mode} mode")
    logger.info(f"Using data from {data_path}")
    logger.info(f"Results will be saved to {output_dir}")
    
    # Run comparison
    if data_path.is_file():
        results = compare_algorithms(
            data_path,
            output_dir=output_dir,
            mode=args.mode,
            subsequence_length=args.sublen,
            step=args.step
        )
    else:
        # Process all stocks in the directory
        processed_files = list(Path(data_path).glob('*_processed.csv'))
        logger.info(f"Found {len(processed_files)} processed files")
        
        for file_path in processed_files:
            results = compare_algorithms(
                file_path,
                output_dir=output_dir,
                mode=args.mode,
                subsequence_length=args.sublen,
                step=args.step
            )
    
    logger.info("Algorithm comparison completed")


if __name__ == "__main__":
    main()
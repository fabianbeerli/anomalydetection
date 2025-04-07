#!/usr/bin/env python
"""
Script to detect anomalies in S&P 500 data using the AIDA algorithm.
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

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.aida import AIDA, TemporalAIDA, MultiTSAIDA, TIX
from src.data.preparation import load_ticker_data
from src.utils.helpers import get_file_list
from src import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_anomalies_single_stock(stock_data_path, output_dir=None, n_subsamples=20, subsample_size=0.1,
                                  metric='manhattan', score_type='variance', plot_results=True):
    """
    Detect anomalies in a single stock using AIDA.
    
    Args:
        stock_data_path (str or Path): Path to the processed stock data file
        output_dir (str or Path, optional): Directory to save results
        n_subsamples (int): Number of subsamples for AIDA
        subsample_size (float): Size of each subsample as a fraction of the data
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
        subsample_size=subsample_size,
        metric='cityblock',
        score_type=score_type,
        alpha_version='random',
        random_state=42
    )
    
    aida.fit(X)
    
    # Get anomaly scores
    anomaly_scores = aida.scores_
    
    # Determine anomalies (top 5%)
    threshold = np.percentile(anomaly_scores, 95)
    anomaly_indices = np.where(anomaly_scores > threshold)[0]
    
    # Map anomaly indices to dates
    anomaly_dates = df.index[anomaly_indices]
    
    # Create TIX for explanations
    tix = TIX(l_norm=1.0, temperature=1.0, n_features_to_explain=len(available_features))
    tix.fit(X, anomaly_scores)
    
    # Prepare results
    results = {
        'ticker': ticker,
        'anomalies': []
    }
    
    # Process each anomaly
    for i, idx in enumerate(anomaly_indices):
        date = anomaly_dates[i]
        score = anomaly_scores[idx]
        
        # Get feature importances
        top_features, importances = tix.explain(idx)
        feature_explanations = {available_features[f]: float(importances[j]) 
                               for j, f in enumerate(top_features)}
        
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
        
        # Plot feature explanations for the top 3 anomalies
        for i, idx in enumerate(anomaly_indices[:3]):
            date = anomaly_dates[i]
            
            fig = tix.plot_explanation(idx, available_features)
            fig.suptitle(f'Feature Importance for Anomaly on {date.strftime("%Y-%m-%d")}')
            
            if plot_dir:
                plt.savefig(plot_dir / f"{ticker}_anomaly_{i+1}_explanation.png", dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
            
            # Distance profile plot
            fig = tix.plot_distance_profile(X, idx)
            fig.suptitle(f'Distance Profile for Anomaly on {date.strftime("%Y-%m-%d")}')
            
            if plot_dir:
                plt.savefig(plot_dir / f"{ticker}_anomaly_{i+1}_distance_profile.png", dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
    
    return results


def detect_anomalies_temporal(stock_data_path, output_dir=None, subsequence_length=5, step=1,
                              n_subsamples=20, subsample_size=0.1, metric='manhattan',
                              score_type='variance', plot_results=True):
    """
    Detect anomalies in time series subsequences using Temporal AIDA.
    
    Args:
        stock_data_path (str or Path): Path to the processed stock data file
        output_dir (str or Path, optional): Directory to save results
        subsequence_length (int): Length of subsequences
        step (int): Step size between subsequences
        n_subsamples (int): Number of subsamples for AIDA
        subsample_size (float): Size of each subsample as a fraction of the data
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
        subsample_size=subsample_size,
        metric=metric,
        score_type=score_type,
        alpha_version='random',
        random_state=42
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
            mask = (df.index >= start_date) & (df.index <= end_date)
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


def detect_anomalies_multi_ts(npz_dir, output_dir=None, n_subsamples=20, subsample_size=0.1,
                             matrix_norm='frobenius', score_type='variance', plot_results=True):
    """
    Detect anomalies in multi-time series subsequence matrices using Multi-TS AIDA.
    
    Args:
        npz_dir (str or Path): Directory containing npz files with subsequence matrices
        output_dir (str or Path, optional): Directory to save results
        n_subsamples (int): Number of subsamples for AIDA
        subsample_size (float): Size of each subsample as a fraction of the data
        matrix_norm (str): Norm to use for matrix distances
        score_type (str): Score type ('expectation' or 'variance')
        plot_results (bool): Whether to plot the results
    
    Returns:
        dict: Dictionary with anomaly detection results
    """
    logger.info(f"Detecting anomalies in multi-TS matrices from {npz_dir}")
    
    # Find all npz files
    npz_dir = Path(npz_dir)
    npz_files = list(npz_dir.glob('*.npz'))
    
    if not npz_files:
        logger.error(f"No npz files found in {npz_dir}")
        return None
    
    logger.info(f"Found {len(npz_files)} npz files")
    
    # Load matrices and metadata
    matrices = []
    metadata_list = []
    
    for file_path in npz_files:
        try:
            data = np.load(file_path, allow_pickle=True)
            matrix = data['matrix']
            metadata = json.loads(data['metadata'].item())
            
            matrices.append(matrix)
            metadata_list.append(metadata)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    if not matrices:
        logger.error("Failed to load any valid matrices")
        return None
    
    # Convert to numpy array
    matrices_array = np.array(matrices)
    
    # Initialize and fit Multi-TS AIDA
    multi_aida = MultiTSAIDA(
        n_subsamples=n_subsamples,
        subsample_size=subsample_size,
        matrix_norm=matrix_norm,
        score_type=score_type,
        alpha_version='random',
        random_state=42
    )
    
    multi_aida.fit(matrices_array, metadata_list)
    
    # Get anomalies
    anomaly_indices, anomaly_scores, anomaly_metadata = multi_aida.get_anomalies(threshold_percentile=95)
    
    # Prepare results
    results = {
        'n_matrices': len(matrices),
        'matrix_shape': matrices[0].shape,
        'anomalies': []
    }
    
    # Process each anomaly
    for i, (idx, score, metadata) in enumerate(zip(anomaly_indices, anomaly_scores, anomaly_metadata)):
        # Add to results
        anomaly_data = {
            'index': int(idx),
            'score': float(score),
            'metadata': metadata
        }
        
        results['anomalies'].append(anomaly_data)
    
    logger.info(f"Found {len(anomaly_indices)} anomalous matrices")
    
    # Save results if output directory is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save anomaly detection results
        results_path = output_dir / f"multi_ts_anomalies.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved multi-TS anomaly detection results to {results_path}")
    
    # Plot results if requested
    if plot_results and output_dir:
        # Create output directory for plots
        plot_dir = output_dir / 'plots' / 'multi_ts'
        plot_dir.mkdir(exist_ok=True, parents=True)
        
        # Plot anomaly scores
        plt.figure(figsize=(12, 6))
        plt.plot(multi_aida.scores_, 'b-')
        plt.scatter(anomaly_indices, anomaly_scores, c='r', s=50)
        plt.axhline(y=np.percentile(multi_aida.scores_, 95), color='g', linestyle='--')
        plt.title('Anomaly Scores for Multi-TS Matrices')
        plt.xlabel('Matrix Index')
        plt.ylabel('Anomaly Score')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(plot_dir / "multi_ts_anomaly_scores.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot heatmaps for anomalous matrices
        for i, (idx, metadata) in enumerate(zip(anomaly_indices[:5], anomaly_metadata[:5])):
            # Get the anomalous matrix
            matrix = matrices_array[idx]
            
            # Get the feature names and tickers
            features = metadata.get('features', [f'Feature {j}' for j in range(matrix.shape[2])])
            tickers = metadata.get('tickers', [f'Stock {j}' for j in range(matrix.shape[0])])
            
            # Create a figure with subplots for each feature
            n_features = matrix.shape[2]
            fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 6), squeeze=False)
            
            for j in range(n_features):
                # Extract the matrix slice for this feature
                feature_matrix = matrix[:, :, j]
                
                # Create heatmap
                im = axes[0, j].imshow(feature_matrix, aspect='auto', cmap='viridis')
                
                # Add colorbar
                plt.colorbar(im, ax=axes[0, j])
                
                # Set labels
                axes[0, j].set_xlabel('Time Step')
                axes[0, j].set_ylabel('Stock')
                axes[0, j].set_title(f'Feature: {features[j]}')
                
                # Set ticks
                axes[0, j].set_yticks(np.arange(len(tickers)))
                axes[0, j].set_yticklabels(tickers)
            
            # Add overall title
            start_date = metadata.get('start_date', 'Unknown')
            end_date = metadata.get('end_date', 'Unknown')
            plt.suptitle(f'Anomalous Matrix {i+1} (Index: {idx})\nPeriod: {start_date} to {end_date}')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(plot_dir / f"anomaly_matrix_{i+1}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    return results


def main():
    """
    Main function to detect anomalies using AIDA.
    """
    parser = argparse.ArgumentParser(description='Detect anomalies in S&P 500 data using AIDA')
    parser.add_argument('--mode', type=str, choices=['single', 'temporal', 'multi'], default='single',
                      help='Detection mode: single (point-based), temporal (subsequence-based), or multi (multi-TS)')
    parser.add_argument('--data', type=str, help='Path to data file or directory')
    parser.add_argument('--output', type=str, default='results/aida',
                      help='Directory to save results (default: results/aida)')
    parser.add_argument('--ticker', type=str, help='Ticker symbol (for single mode)')
    parser.add_argument('--sublen', type=int, default=5,
                      help='Subsequence length for temporal mode (default: 5)')
    parser.add_argument('--step', type=int, default=1,
                      help='Step size for temporal mode (default: 1)')
    parser.add_argument('--norm', type=str, default='frobenius',
                      help='Matrix norm for multi mode (default: frobenius)')
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
        elif args.mode == 'temporal':
            if args.ticker:
                data_path = config.PROCESSED_DATA_DIR / f"{args.ticker}_processed.csv"
            else:
                data_path = config.PROCESSED_DATA_DIR / f"{config.SP500_TICKER.replace('^', 'index_')}_processed.csv"
        else:  # multi mode
            data_path = config.PROCESSED_DATA_DIR / 'multi_ts'
    
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
                    plot_results=args.plot
                )
    
    elif args.mode == 'temporal':
        # Temporal detection
        if data_path.is_file():
            results = detect_anomalies_temporal(
                data_path,
                output_dir=output_dir,
                subsequence_length=args.sublen,
                step=args.step,
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
                    plot_results=args.plot
                )
    
    else:  # multi mode
        # Multi-TS detection
        results = detect_anomalies_multi_ts(
            data_path,
            output_dir=output_dir,
            matrix_norm=args.norm,
            plot_results=args.plot
        )
    
    logger.info("Anomaly detection completed")


if __name__ == "__main__":
    main()
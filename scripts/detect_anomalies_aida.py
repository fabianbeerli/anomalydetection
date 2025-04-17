#!/usr/bin/env python
"""
Comprehensive Anomaly Detection Comparison Script

Compares multiple anomaly detection algorithms:
- AIDA (C++ implementation)
- Isolation Forest (Python sklearn)
- Local Outlier Factor (Python sklearn)
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

# Import libraries
from src.models.aida_cpp import AIDA  # C++ implementation
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from src.data.preparation import load_ticker_data
from src import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compare_anomaly_detection_methods(stock_data_path, output_dir=None):
    """
    Compare anomaly detection methods on a single stock dataset.
    
    Args:
        stock_data_path (str or Path): Path to processed stock data
        output_dir (str or Path, optional): Directory to save results
    
    Returns:
        dict: Comparison results
    """
    # Load data
    df = load_ticker_data(stock_data_path)
    ticker = Path(stock_data_path).stem.replace('_processed', '')
    
    # Select features
    feature_columns = [
        'daily_return', 
        'log_return',
        'volume_change', 
        'relative_volume',
        'high_low_range',
        'daily_return_zscore',
        'volume_change_zscore'
    ]
    
    # Validate and extract features
    available_features = [col for col in feature_columns if col in df.columns]
    
    if not available_features:
        logger.error(f"No required features found for {ticker}")
        return None
    
    # Prepare feature matrix
    X = df[available_features].values
    
    # Initialize anomaly detection methods
    methods = {
        'AIDA (C++)': AIDA(
            n_subsamples=20,
            score_type='variance',
            alpha_min=0.5,
            alpha_max=1.5
        ),
        'Isolation Forest': IsolationForest(
            contamination=0.05,
            random_state=42
        ),
        'Local Outlier Factor': LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.05
        )
    }
    
    # Results dictionary
    results = {
        'ticker': ticker,
        'methods': {}
    }
    
    # Anomaly detection
    for name, method in methods.items():
        # Detect anomalies
        if name == 'AIDA (C++)':
            method.fit(X)
            scores = method.scores_
            anomaly_threshold = np.percentile(scores, 95)
            anomalies = scores > anomaly_threshold
        elif name == 'Isolation Forest':
            scores = -method.fit_predict(X)  # Scores where lower is more anomalous
            anomalies = scores < 0
        elif name == 'Local Outlier Factor':
            method.fit(X)
            scores = -method.negative_outlier_factor_
            anomaly_threshold = np.percentile(scores, 95)
            anomalies = scores > anomaly_threshold
        
        # Collect method results
        results['methods'][name] = {
            'anomaly_indices': np.where(anomalies)[0].tolist(),
            'anomaly_dates': df.index[anomalies].strftime('%Y-%m-%d').tolist(),
            'scores': scores.tolist(),
            'anomaly_count': int(np.sum(anomalies))
        }
    
    # Compute method overlap
    all_anomalies = {name: set(indices) for name, indices in 
                     [(name, results['methods'][name]['anomaly_indices']) 
                      for name in results['methods']]}
    
    common_anomalies = set.intersection(*all_anomalies.values())
    unique_anomalies = {name: indices - set.union(*[all_anomalies[k] for k in all_anomalies if k != name]) 
                        for name, indices in all_anomalies.items()}
    
    results['overlap'] = {
        'common_anomalies': list(common_anomalies),
        'unique_anomalies': {name: list(indices) for name, indices in unique_anomalies.items()}
    }
    
    # Visualization
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        plot_dir = output_dir / 'plots'
        plot_dir.mkdir(exist_ok=True, parents=True)
        
        # Visualization: Stock Price with Detected Anomalies
        plt.figure(figsize=(15, 6))
        plt.plot(df.index, df['Close'], label='Close Price')
        
        # Plot anomalies from each method
        for name, data in results['methods'].items():
            anomaly_dates = [df.index[idx] for idx in data['anomaly_indices']]
            plt.scatter(anomaly_dates, df['Close'].loc[anomaly_dates], label=f'{name} Anomalies', alpha=0.7)
        
        plt.title(f'Anomalies Detected in {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / f'{ticker}_anomalies_comparison.png')
        plt.close()
        
        # Save results as JSON
        with open(output_dir / f'{ticker}_anomaly_comparison.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


def main():
    """
    Main function to run anomaly detection comparison.
    """
    parser = argparse.ArgumentParser(description='Compare anomaly detection methods')
    parser.add_argument('--data', type=str, help='Path to data file or directory')
    parser.add_argument('--output', type=str, default='results/anomaly_comparison',
                      help='Directory to save results')
    
    args = parser.parse_args()
    
    # Determine data path
    if args.data:
        data_path = Path(args.data)
    else:
        data_path = config.PROCESSED_DATA_DIR / f"{config.SP500_TICKER.replace('^', 'index_')}_processed.csv"
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Process files
    if data_path.is_file():
        results = compare_anomaly_detection_methods(data_path, output_dir)
    else:
        processed_files = list(data_path.glob('*_processed.csv'))
        for file_path in processed_files:
            results = compare_anomaly_detection_methods(file_path, output_dir)


if __name__ == "__main__":
    main()
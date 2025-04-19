#!/usr/bin/env python
"""
Script to run anomaly detection on constituent stocks around S&P 500 index anomalies.
Uses matrix-based approach that preserves temporal order.
"""
import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.utils.helpers import ensure_directory_exists, load_subsequence, get_file_list
from src.data.preparation import load_ticker_data

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MatrixBasedAnomalyDetector:
    """
    Anomaly detector that works with matrices to preserve temporal and cross-sectional structure.
    """
    
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.scaler = StandardScaler()
    
    def prepare_matrix_features(self, matrix_data):
        """
        Prepare features from matrix data while preserving structure.
        
        Args:
            matrix_data (np.ndarray): 3D array of shape (n_samples, n_stocks, n_features)
            
        Returns:
            np.ndarray: Feature matrix for anomaly detection
        """
        # For now, we'll use a matrix representation that captures cross-sectional and temporal patterns
        n_samples, n_stocks, n_features = matrix_data.shape
        
        # Calculate cross-sectional statistics (across stocks)
        cross_sectional_mean = np.mean(matrix_data, axis=1)  # Shape: (n_samples, n_features)
        cross_sectional_std = np.std(matrix_data, axis=1)
        cross_sectional_median = np.median(matrix_data, axis=1)
        
        # Calculate temporal statistics within each window (for each stock)
        temporal_mean = np.mean(matrix_data, axis=2)  # Shape: (n_samples, n_stocks)
        temporal_std = np.std(matrix_data, axis=2)
        
        # Calculate pairwise correlations between stocks within each sample
        correlations = np.zeros((n_samples, n_stocks * (n_stocks - 1) // 2))
        
        for i in range(n_samples):
            corr_matrix = np.corrcoef(matrix_data[i])
            # Extract upper triangle of correlation matrix (excluding diagonal)
            corr_values = corr_matrix[np.triu_indices(n_stocks, k=1)]
            correlations[i] = corr_values
        
        # Combine all features
        features = np.hstack([
            cross_sectional_mean,
            cross_sectional_std,
            cross_sectional_median,
            temporal_mean,
            temporal_std,
            correlations
        ])
        
        return features
    
    def detect_anomalies(self, matrix_data, algorithm='iforest'):
        """
        Detect anomalies in matrix data.
        
        Args:
            matrix_data (np.ndarray): 3D array of shape (n_samples, n_stocks, n_features)
            algorithm (str): 'iforest' or 'lof'
            
        Returns:
            tuple: (scores, labels)
        """
        # Prepare features from matrix data
        features = self.prepare_matrix_features(matrix_data)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Run anomaly detection
        if algorithm == 'iforest':
            detector = IsolationForest(
                n_estimators=100,
                max_samples=min(256, len(features)),
                contamination=self.contamination,
                random_state=42
            )
            detector.fit(features_scaled)
            scores = -detector.decision_function(features_scaled)
            labels = detector.predict(features_scaled)
        
        elif algorithm == 'lof':
            detector = LocalOutlierFactor(
                n_neighbors=min(20, len(features) - 1),
                contamination=self.contamination
            )
            labels = detector.fit_predict(features_scaled)
            scores = -detector.negative_outlier_factor_
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return scores, labels


def load_constituent_data(start_date, end_date, tickers):
    """
    Load constituent data for the specified date range.
    
    Args:
        start_date (datetime): Start date
        end_date (datetime): End date
        tickers (list): List of constituent tickers
        
    Returns:
        dict: Dictionary of DataFrames for each ticker
    """
    constituent_data = {}
    
    for ticker in tickers:
        try:
            # Try to load preprocessed data
            ticker_file = config.PROCESSED_DATA_DIR / f"{ticker}_processed.csv"
            
            if ticker_file.exists():
                df = load_ticker_data(ticker_file)
                
                if df is not None and not df.empty:
                    # Filter for date range
                    mask = (df.index >= start_date) & (df.index <= end_date)
                    df_filtered = df[mask]
                    
                    if not df_filtered.empty:
                        constituent_data[ticker] = df_filtered
                        logger.info(f"Loaded data for {ticker}: {len(df_filtered)} rows")
                    else:
                        logger.warning(f"No data for {ticker} in date range {start_date} to {end_date}")
                else:
                    logger.warning(f"Empty or invalid data for {ticker}")
            else:
                logger.warning(f"Processed data file not found for {ticker}")
        
        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {e}")
    
    return constituent_data


def create_matrix_around_anomaly(constituent_data, anomaly_date, window_days=5):
    """
    Create a matrix of constituent data around an anomaly date.
    
    Args:
        constituent_data (dict): Dictionary of DataFrames for each constituent
        anomaly_date (datetime): Date of the index anomaly
        window_days (int): Number of days before and after the anomaly
        
    Returns:
        tuple: (matrix, metadata) where matrix is shape (n_days, n_stocks, n_features)
    """
    # Define the date range
    start_date = anomaly_date - timedelta(days=window_days)
    end_date = anomaly_date + timedelta(days=window_days)
    
    # Create date range (business days only)
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Features to use for anomaly detection
    features = ['daily_return', 'volume_change', 'high_low_range']
    
    # Initialize matrix
    n_dates = len(date_range)
    n_stocks = len(constituent_data)
    n_features = len(features)
    
    matrix = np.full((n_dates, n_stocks, n_features), np.nan)
    stock_names = list(constituent_data.keys())
    
    # Fill the matrix
    for i, date in enumerate(date_range):
        for j, (ticker, df) in enumerate(constituent_data.items()):
            if date in df.index:
                for k, feature in enumerate(features):
                    if feature in df.columns:
                        matrix[i, j, k] = df.loc[date, feature]
    
    # Find the index of the anomaly date in our date range
    anomaly_idx = None
    for i, date in enumerate(date_range):
        if date.date() == anomaly_date.date():
            anomaly_idx = i
            break
    
    metadata = {
        'date_range': date_range.strftime('%Y-%m-%d').tolist(),
        'stock_names': stock_names,
        'features': features,
        'anomaly_idx': anomaly_idx,
        'start_date': date_range[0].strftime('%Y-%m-%d'),
        'end_date': date_range[-1].strftime('%Y-%m-%d')
    }
    
    return matrix, metadata


def analyze_constituent_anomalies(index_anomalies_file, output_dir):
    """
    Analyze constituent behavior around S&P 500 index anomalies.
    
    Args:
        index_anomalies_file (Path): Path to file with index anomalies
        output_dir (Path): Directory to save results
    """
    ensure_directory_exists(output_dir)
    
    # Load index anomalies
    try:
        index_anomalies = pd.read_csv(index_anomalies_file)
        
        # Convert date column
        if 'start_date' in index_anomalies.columns:
            index_anomalies['date'] = pd.to_datetime(index_anomalies['start_date'])
        elif 'date' in index_anomalies.columns:
            index_anomalies['date'] = pd.to_datetime(index_anomalies['date'])
        else:
            logger.error("No date column found in index anomalies file")
            return
    except Exception as e:
        logger.error(f"Error loading index anomalies: {e}")
        return
    
    # Get list of constituent tickers
    # For now, use the predefined top constituents from config
    # In production, this would be expanded to include additional stocks
    constituent_tickers = config.TOP_SP500_CONSTITUENTS
    
    # Initialize anomaly detector
    detector = MatrixBasedAnomalyDetector(contamination=0.1)
    
    # Analyze each index anomaly
    results = []
    
    for idx, row in index_anomalies.iterrows():
        anomaly_date = row['date']
        logger.info(f"Analyzing constituent behavior for anomaly on {anomaly_date}")
        
        # Define date range for analysis
        window_days = 5
        start_date = anomaly_date - timedelta(days=window_days)
        end_date = anomaly_date + timedelta(days=window_days)
        
        # Load constituent data
        constituent_data = load_constituent_data(start_date, end_date, constituent_tickers)
        
        if not constituent_data:
            logger.warning(f"No constituent data available for {anomaly_date}")
            continue
        
        # Create matrix around anomaly
        matrix, metadata = create_matrix_around_anomaly(constituent_data, anomaly_date, window_days)
        
        # Skip if we have too many missing values
        if np.isnan(matrix).mean() > 0.3:  # More than 30% missing
            logger.warning(f"Too many missing values for {anomaly_date}, skipping")
            continue
        
        # Handle missing values - forward fill then backward fill
        for i in range(matrix.shape[1]):  # For each stock
            for j in range(matrix.shape[2]):  # For each feature
                column = matrix[:, i, j]
                if np.isnan(column).any():
                    # Forward fill
                    mask = np.isnan(column)
                    idx = np.where(~mask, np.arange(len(mask)), 0)
                    np.maximum.accumulate(idx, out=idx)
                    column[mask] = column[idx[mask]]
                    
                    # Backward fill remaining
                    mask = np.isnan(column)
                    idx = np.where(~mask, np.arange(len(mask)), len(mask) - 1)
                    idx = np.minimum.accumulate(idx[::-1])[::-1]
                    column[mask] = column[idx[mask]]
                    
                    matrix[:, i, j] = column
        
        # Reshape matrix for anomaly detection
        # Create sliding windows of length 3 around the anomaly
        window_length = 3
        windows = []
        
        for i in range(len(metadata['date_range']) - window_length + 1):
            windows.append(matrix[i:i+window_length])
        
        windows_array = np.array(windows)
        
        if len(windows_array) < 2:
            logger.warning(f"Not enough windows for {anomaly_date}, skipping")
            continue
        
        # Detect anomalies using both algorithms
        iforest_scores, iforest_labels = detector.detect_anomalies(windows_array, algorithm='iforest')
        lof_scores, lof_labels = detector.detect_anomalies(windows_array, algorithm='lof')
        
        # Find the window that contains the anomaly date
        anomaly_window_idx = None
        if metadata['anomaly_idx'] is not None:
            # Find which window contains the anomaly date
            for i in range(len(windows_array)):
                if i <= metadata['anomaly_idx'] < i + window_length:
                    anomaly_window_idx = i
                    break
        
        if anomaly_window_idx is None:
            logger.warning(f"Could not find window containing anomaly date for {anomaly_date}")
            continue
        
        # Check if the window with the anomaly is detected as anomalous
        iforest_is_anomaly = iforest_labels[anomaly_window_idx] == -1
        lof_is_anomaly = lof_labels[anomaly_window_idx] == -1
        
        # Count how many windows are anomalous
        n_iforest_anomalies = np.sum(iforest_labels == -1)
        n_lof_anomalies = np.sum(lof_labels == -1)
        
        # Classify based on number of anomalous windows
        classification = {
            'iforest': 'isolated' if n_iforest_anomalies == 1 else 
                      'widespread' if n_iforest_anomalies > 2 else 
                      'moderate',
            'lof': 'isolated' if n_lof_anomalies == 1 else 
                  'widespread' if n_lof_anomalies > 2 else 
                  'moderate'
        }
        
        result = {
            'anomaly_date': anomaly_date.strftime('%Y-%m-%d'),
            'n_stocks': len(metadata['stock_names']),
            'stocks': metadata['stock_names'],
            'n_windows': len(windows_array),
            'anomaly_window_idx': anomaly_window_idx,
            'iforest_is_anomaly': bool(iforest_is_anomaly),
            'lof_is_anomaly': bool(lof_is_anomaly),
            'n_iforest_anomalies': int(n_iforest_anomalies),
            'n_lof_anomalies': int(n_lof_anomalies),
            'classification_iforest': classification['iforest'],
            'classification_lof': classification['lof'],
            'iforest_score': float(iforest_scores[anomaly_window_idx]),
            'lof_score': float(lof_scores[anomaly_window_idx])
        }
        
        results.append(result)
        logger.info(f"Completed analysis for {anomaly_date}")
    
    # Save results
    if results:
        results_file = output_dir / 'constituent_analysis_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved constituent analysis results to {results_file}")
        
        # Create summary
        summary = {
            'total_index_anomalies': len(index_anomalies),
            'analyzed_anomalies': len(results),
            'iforest_classifications': {
                'widespread': sum(1 for r in results if r['classification_iforest'] == 'widespread'),
                'isolated': sum(1 for r in results if r['classification_iforest'] == 'isolated'),
                'moderate': sum(1 for r in results if r['classification_iforest'] == 'moderate')
            },
            'lof_classifications': {
                'widespread': sum(1 for r in results if r['classification_lof'] == 'widespread'),
                'isolated': sum(1 for r in results if r['classification_lof'] == 'isolated'),
                'moderate': sum(1 for r in results if r['classification_lof'] == 'moderate')
            },
            'anomaly_agreement': sum(1 for r in results if r['iforest_is_anomaly'] == r['lof_is_anomaly']),
            'average_stocks_analyzed': np.mean([r['n_stocks'] for r in results])
        }
        
        summary_file = output_dir / 'constituent_analysis_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved constituent analysis summary to {summary_file}")
    else:
        logger.warning("No results to save")


def main():
    """
    Main function to run constituent anomaly analysis.
    """
    parser = argparse.ArgumentParser(description="Run constituent analysis for S&P 500 anomalies")
    parser.add_argument(
        "--index-anomalies", 
        type=str, 
        default=str(config.DATA_DIR / "subsequence_results" / "aida" / "w3_overlap" / "aida_anomalies.csv"),
        help="Path to index anomalies file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=str(config.DATA_DIR / "constituent_analysis"),
        help="Directory to save analysis results"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    index_anomalies_file = Path(args.index_anomalies)
    output_dir = Path(args.output)
    
    # Run analysis
    analyze_constituent_anomalies(index_anomalies_file, output_dir)
    
    logger.info("Constituent analysis completed")


if __name__ == "__main__":
    main()
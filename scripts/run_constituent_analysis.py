#!/usr/bin/env python
"""
Script to analyze correlations between S&P 500 index anomalies and constituent stock anomalies.
"""
import os
import sys
import logging
import argparse
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from matplotlib.dates import DateFormatter
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data.preparation import load_ticker_data
from src.utils.helpers import ensure_directory_exists, plot_time_series

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConstituentAnomalyDetector:
    """
    Detect anomalies in constituent stock data around index anomalies.
    """
    
    def __init__(self, window_size=5, contamination=0.05):
        """
        Initialize the detector.
        
        Args:
            window_size (int): Size of the window around anomalies
            contamination (float): Expected proportion of anomalies
        """
        self.window_size = window_size
        self.contamination = contamination
    
    def prepare_matrix_features(self, matrix_data):
        """
        Prepare features from matrix data.
        
        Args:
            matrix_data (numpy.ndarray): Matrix of time series data
            
        Returns:
            numpy.ndarray: Flattened feature array
        """
        # Handle different shapes of matrix_data
        if matrix_data.ndim == 3:
            # Expected shape: (n_samples, n_stocks, n_features)
            n_samples, n_stocks, n_features = matrix_data.shape
            
            # Reshape to 2D: (n_samples, n_stocks*n_features)
            features = matrix_data.reshape(n_samples, -1)
            logger.info(f"Prepared 3D matrix features with shape: {features.shape}")
            
        elif matrix_data.ndim == 2:
            # Already 2D, but log the shape
            n_samples, feature_dim = matrix_data.shape
            logger.info(f"Using 2D matrix with shape: {matrix_data.shape}")
            features = matrix_data
            
        else:
            # If there's an unexpected dimension, try to flatten or reshape
            logger.warning(f"Unexpected matrix dimension: {matrix_data.ndim}, shape: {matrix_data.shape}")
            try:
                # Try to flatten to 2D
                features = matrix_data.reshape(matrix_data.shape[0], -1)
                logger.info(f"Reshaped to 2D features with shape: {features.shape}")
            except Exception as e:
                logger.error(f"Failed to reshape matrix: {e}")
                # Return the original data as fallback
                features = matrix_data
        
        return features
    
    def detect_anomalies(self, matrix_data, algorithm='iforest'):
        """
        Detect anomalies in the matrix data.
        
        Args:
            matrix_data (numpy.ndarray): Matrix of time series data
            algorithm (str): Detection algorithm to use ('iforest' or 'lof')
            
        Returns:
            tuple: (scores, labels)
        """
        # Prepare features
        features = self.prepare_matrix_features(matrix_data)
        
        try:
            if algorithm == 'iforest':
                # Use Isolation Forest
                detector = IsolationForest(
                    n_estimators=100,
                    max_samples=min(256, features.shape[0]),
                    contamination=self.contamination,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                # Use LOF
                detector = LocalOutlierFactor(
                    n_neighbors=min(20, features.shape[0] - 1),
                    contamination=self.contamination,
                    n_jobs=-1,
                    novelty=False
                )
            
            # Fit and predict
            if algorithm == 'iforest':
                detector.fit(features)
                labels = detector.predict(features)
                scores = -detector.decision_function(features)
            else:
                # For LOF, use fit_predict and get the negative_outlier_factor_
                labels = detector.fit_predict(features)
                # LOF in scikit-learn already stores negative factor, so we don't need to negate it
                scores = detector.negative_outlier_factor_
            
            # Normalize scores
            if len(scores) > 1:
                scores = (scores - np.mean(scores)) / np.std(scores)
            
            return scores, labels
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return None, None


def load_index_anomalies(anomalies_file):
    """
    Load S&P 500 index anomalies.
    
    Args:
        anomalies_file (Path): Path to the anomalies CSV file
        
    Returns:
        pandas.DataFrame: Anomaly data
    """
    try:
        logger.info(f"Loading index anomalies from {anomalies_file}")
        anomalies_df = pd.read_csv(anomalies_file)
        
        # Ensure date columns are datetime
        for date_col in ['start_date', 'end_date']:
            if date_col in anomalies_df.columns:
                anomalies_df[date_col] = pd.to_datetime(anomalies_df[date_col])
        
        logger.info(f"Loaded {len(anomalies_df)} index anomalies")
        return anomalies_df
    except Exception as e:
        logger.error(f"Error loading index anomalies: {e}")
        return None


def load_constituent_data(tickers, start_date, end_date):
    """
    Load constituent stock data for a specific window.
    
    Args:
        tickers (list): List of ticker symbols
        start_date (datetime): Start date of the window
        end_date (datetime): End date of the window
        
    Returns:
        dict: Dictionary of DataFrames for each ticker
    """
    constituent_data = {}
    
    for ticker in tickers:
        try:
            # Load processed data for the ticker
            file_path = config.PROCESSED_DATA_DIR / f"{ticker}_processed.csv"
            if not file_path.exists():
                logger.warning(f"No data file found for {ticker}")
                continue
                
            df = load_ticker_data(file_path)
            
            if df is None or df.empty:
                logger.warning(f"Empty data for {ticker}")
                continue
            
            # Filter to the date range
            if start_date and end_date:
                try:
                    df = df.loc[start_date:end_date]
                except Exception as e:
                    logger.warning(f"Error filtering dates for {ticker}: {e}")
            
            if df.empty:
                logger.warning(f"No data in date range for {ticker}")
                continue
                
            # Keep the DataFrame
            constituent_data[ticker] = df
            logger.info(f"Loaded data for {ticker}: {len(df)} rows")
            
        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {e}")
    
    return constituent_data


def create_constituent_matrix(constituent_data, feature_columns):
    """
    Create a 3D matrix of constituent data.
    
    Args:
        constituent_data (dict): Dictionary of DataFrames for each ticker
        feature_columns (list): List of feature columns to include
        
    Returns:
        tuple: (3D matrix, list of dates, list of tickers)
    """
    if not constituent_data:
        logger.error("No constituent data to create matrix")
        return None, None, None
    
    # Get common dates
    common_dates = None
    tickers = list(constituent_data.keys())
    
    for ticker, df in constituent_data.items():
        if common_dates is None:
            common_dates = set(df.index)
        else:
            common_dates &= set(df.index)
    
    if not common_dates:
        logger.error("No common dates found across constituents")
        return None, None, None
    
    # Sort common dates
    common_dates = sorted(common_dates)
    
    # Create matrix: (n_timesteps, n_stocks, n_features)
    n_timesteps = len(common_dates)
    n_stocks = len(constituent_data)
    n_features = len(feature_columns)
    
    # Initialize matrix with NaNs
    matrix = np.empty((n_timesteps, n_stocks, n_features))
    matrix[:] = np.nan
    
    # Fill matrix
    for i, ticker in enumerate(tickers):
        df = constituent_data[ticker]
        for j, feature in enumerate(feature_columns):
            if feature in df.columns:
                # Reindex DataFrame to common dates
                feature_values = df.reindex(common_dates)[feature].values
                matrix[:, i, j] = feature_values
    
    # Check for NaNs and fill
    if np.isnan(matrix).any():
        logger.warning(f"Matrix contains {np.isnan(matrix).sum()} NaN values, filling with 0")
        matrix = np.nan_to_num(matrix, nan=0.0)
    
    return matrix, common_dates, tickers


def slide_window_over_matrix(matrix, dates, window_size=5):
    """
    Create sliding windows over the 3D matrix.
    
    Args:
        matrix (numpy.ndarray): 3D matrix of constituent data
        dates (list): List of dates corresponding to matrix timesteps
        window_size (int): Size of each window
        
    Returns:
        tuple: (windows_array, window_dates)
    """
    n_timesteps = matrix.shape[0]
    
    if n_timesteps < window_size:
        logger.warning(f"Not enough timesteps ({n_timesteps}) for window size {window_size}")
        return matrix[np.newaxis, :, :, :], [dates]
    
    # Create windows
    n_windows = n_timesteps - window_size + 1
    windows = []
    window_dates = []
    
    for i in range(n_windows):
        window = matrix[i:i+window_size, :, :]
        windows.append(window)
        window_dates.append((dates[i], dates[i+window_size-1]))
    
    # Stack windows into a 4D array: (n_windows, window_size, n_stocks, n_features)
    windows_array = np.stack(windows)
    
    return windows_array, window_dates


def visualize_constituent_anomalies(constituent_data, tickers, window_dates, anomaly_indices, scores, output_path, algorithm='iforest'):
    """
    Visualize constituent anomalies.
    
    Args:
        constituent_data (dict): Dictionary of DataFrames for each ticker
        tickers (list): List of ticker symbols
        window_dates (list): List of date pairs for each window
        anomaly_indices (list): Indices of anomalous windows
        scores (numpy.ndarray): Anomaly scores
        output_path (Path): Path to save visualizations
        algorithm (str): Algorithm used for detection
    """
    try:
        if not anomaly_indices.size:
            logger.warning(f"No anomalies detected with {algorithm}, skipping visualization")
            return
        
        # Create directory for outputs
        algorithm_dir = output_path / algorithm
        ensure_directory_exists(algorithm_dir)
        
        # Plot overview of all anomalies
        plt.figure(figsize=(16, 10))
        
        # Plot anomaly scores
        plt.subplot(2, 1, 1)
        plt.plot(scores, 'b-', alpha=0.5)
        plt.scatter(anomaly_indices, scores[anomaly_indices], color='r', label=f'Anomalies ({len(anomaly_indices)})')
        plt.title(f"Constituent Window Anomalies - {algorithm.upper()}")
        plt.xlabel("Window Index")
        plt.ylabel("Anomaly Score")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot top 5 stocks by anomaly window counts
        plt.subplot(2, 1, 2)
        
        # For each anomaly window, collect the constituent data
        anomaly_dates = [window_dates[i] for i in anomaly_indices]
        
        # Count anomalies by stock
        stock_anomaly_counts = {ticker: 0 for ticker in tickers}
        
        for start_date, end_date in anomaly_dates:
            for ticker, df in constituent_data.items():
                if ticker in tickers:
                    try:
                        window_data = df.loc[start_date:end_date]
                        if not window_data.empty:
                            stock_anomaly_counts[ticker] += 1
                    except:
                        pass
        
        # Sort by count
        stock_counts = sorted(stock_anomaly_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Plot top 5
        top_stocks = stock_counts[:5]
        if top_stocks:
            plt.bar([stock[0] for stock in top_stocks], [stock[1] for stock in top_stocks])
            plt.title("Top 5 Stocks by Anomaly Window Count")
            plt.xlabel("Stock")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(algorithm_dir / "constituent_anomalies_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save anomalies to CSV
        anomalies_df = pd.DataFrame({
            'window_index': anomaly_indices,
            'anomaly_score': scores[anomaly_indices]
        })
        
        # Add date information
        anomalies_df['start_date'] = [window_dates[i][0] for i in anomaly_indices]
        anomalies_df['end_date'] = [window_dates[i][1] for i in anomaly_indices]
        
        # Add top anomalous stocks for each window
        top_stocks_per_window = []
        
        for i in anomaly_indices:
            start_date, end_date = window_dates[i]
            stock_scores = {}
            
            for ticker, df in constituent_data.items():
                if ticker in tickers:
                    try:
                        window_data = df.loc[start_date:end_date]
                        if not window_data.empty and 'daily_return' in window_data.columns:
                            # Use simple volatility as a proxy for anomaly
                            score = window_data['daily_return'].std()
                            stock_scores[ticker] = score
                    except:
                        pass
            
            # Get top 3 stocks by volatility
            top_stocks_str = ", ".join([f"{ticker} ({score:.4f})" for ticker, score in 
                                      sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)[:3]])
            top_stocks_per_window.append(top_stocks_str)
        
        anomalies_df['top_anomalous_stocks'] = top_stocks_per_window
        
        # Save to CSV
        anomalies_df.to_csv(algorithm_dir / "constituent_anomalies.csv", index=False)
        
        # Generate detailed visualizations for top 5 anomalies
        top_anomalies = anomalies_df.sort_values('anomaly_score', ascending=False).iloc[:5]
        
        for idx, row in top_anomalies.iterrows():
            try:
                window_idx = row['window_index']
                start_date = row['start_date']
                end_date = row['end_date']
                
                # Create window visualization
                plt.figure(figsize=(16, 12))
                
                # Plot index performance
                plt.subplot(3, 1, 1)
                for ticker, df in constituent_data.items():
                    if ticker == "index_GSPC":  # S&P 500 index
                        window_data = df.loc[start_date:end_date]
                        if not window_data.empty and 'Close' in window_data.columns:
                            plt.plot(window_data.index, window_data['Close'], 'b-', linewidth=2)
                            plt.title(f"S&P 500 Index - Anomaly Window {window_idx}")
                            plt.ylabel("Price")
                            plt.grid(True, alpha=0.3)
                            break
                
                # Plot top 3 anomalous stocks
                top_stocks = row['top_anomalous_stocks'].split(", ")[:3]
                top_tickers = [stock.split(" ")[0] for stock in top_stocks]
                
                plt.subplot(3, 1, 2)
                for ticker in top_tickers:
                    if ticker in constituent_data:
                        window_data = constituent_data[ticker].loc[start_date:end_date]
                        if not window_data.empty and 'Close' in window_data.columns:
                            plt.plot(window_data.index, window_data['Close'], label=ticker)
                
                plt.title("Top Anomalous Stocks - Price")
                plt.ylabel("Price")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Plot returns of top stocks
                plt.subplot(3, 1, 3)
                for ticker in top_tickers:
                    if ticker in constituent_data:
                        window_data = constituent_data[ticker].loc[start_date:end_date]
                        if not window_data.empty and 'daily_return' in window_data.columns:
                            plt.plot(window_data.index, window_data['daily_return'], label=ticker)
                
                plt.title("Top Anomalous Stocks - Daily Returns")
                plt.ylabel("Return")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(algorithm_dir / f"anomaly_window_{window_idx}_detail.png", dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                logger.error(f"Error generating detailed visualization for window {row['window_index']}: {e}")
        
        logger.info(f"Saved constituent anomaly visualizations to {algorithm_dir}")
        
    except Exception as e:
        logger.error(f"Error visualizing constituent anomalies: {e}")


def analyze_constituent_anomalies(index_anomalies_file, output_dir):
    """
    Analyze constituent stock anomalies around index anomalies.
    
    Args:
        index_anomalies_file (str or Path): Path to the index anomalies CSV
        output_dir (str or Path): Directory to save outputs
    """
    # Load index anomalies
    index_anomalies = load_index_anomalies(Path(index_anomalies_file))
    
    if index_anomalies is None or index_anomalies.empty:
        logger.error("No index anomalies to analyze")
        return
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    ensure_directory_exists(output_path)
    
    # Top and random constituents to analyze
    constituents = config.TOP_SP500_CONSTITUENTS
    
    # For each index anomaly, analyze constituent behavior
    for idx, anomaly in index_anomalies.iterrows():
        try:
            logger.info(f"Analyzing constituents for anomaly {idx+1}/{len(index_anomalies)}")
            
            # Get anomaly dates - use start_date if available, otherwise use index
            if 'start_date' in anomaly:
                anomaly_date = pd.to_datetime(anomaly['start_date'])
            else:
                # Skip if no date available
                logger.warning(f"No date available for anomaly {idx+1}, skipping")
                continue
            
            # Define window around anomaly
            window_days = config.ANOMALY_WINDOW
            start_date = anomaly_date - pd.Timedelta(days=window_days)
            end_date = anomaly_date + pd.Timedelta(days=window_days)
            
            # Create subdirectory for this anomaly
            anomaly_dir = output_path / f"anomaly_{idx+1}_{anomaly_date.strftime('%Y%m%d')}"
            ensure_directory_exists(anomaly_dir)
            
            # Load constituent data
            all_constituents = constituents + ["index_GSPC"]  # Include S&P 500 index
            constituent_data = load_constituent_data(all_constituents, start_date, end_date)
            
            if not constituent_data:
                logger.warning(f"No constituent data available for anomaly {idx+1}, skipping")
                continue
            
            # Feature columns to analyze
            feature_columns = ['daily_return', 'volume_change', 'high_low_range']
            
            # Filter constituent data to ensure all have required features
            filtered_data = {}
            for ticker, df in constituent_data.items():
                available_features = [col for col in feature_columns if col in df.columns]
                if len(available_features) == len(feature_columns):
                    filtered_data[ticker] = df[available_features]
                else:
                    logger.warning(f"Ticker {ticker} missing required features, skipping")
            
            if not filtered_data:
                logger.warning(f"No constituent data with required features for anomaly {idx+1}, skipping")
                continue
            
            # Create matrix of constituent data
            matrix, dates, tickers = create_constituent_matrix(filtered_data, feature_columns)
            
            if matrix is None or len(dates) < 3:
                logger.warning(f"Insufficient data for matrix creation for anomaly {idx+1}, skipping")
                continue
            
            # Create sliding windows
            window_size = 3  # Size of each window
            windows_array, window_dates = slide_window_over_matrix(matrix, dates, window_size)
            
            # Debugging: Check the shape of windows_array
            logger.info(f"Windows array shape: {windows_array.shape}")
            
            # Detect anomalies using Isolation Forest
            iforest_scores, iforest_labels = detector.detect_anomalies(windows_array, algorithm='iforest')
            
            # Detect anomalies using LOF
            lof_scores, lof_labels = detector.detect_anomalies(windows_array, algorithm='lof')
            
            # Get anomaly indices
            iforest_anomaly_indices = np.where(iforest_labels == -1)[0] if iforest_labels is not None else np.array([])
            lof_anomaly_indices = np.where(lof_labels == -1)[0] if lof_labels is not None else np.array([])
            
            # Visualize constituent anomalies
            visualize_constituent_anomalies(
                filtered_data,
                tickers,
                window_dates,
                iforest_anomaly_indices,
                iforest_scores,
                anomaly_dir,
                algorithm='iforest'
            )
            
            visualize_constituent_anomalies(
                filtered_data,
                tickers,
                window_dates,
                lof_anomaly_indices,
                lof_scores,
                anomaly_dir,
                algorithm='lof'
            )
            
            # Save window data
            window_info = pd.DataFrame({
                'window_index': range(len(window_dates)),
                'start_date': [d[0] for d in window_dates],
                'end_date': [d[1] for d in window_dates],
                'iforest_score': iforest_scores if iforest_scores is not None else np.zeros(len(window_dates)),
                'lof_score': lof_scores if lof_scores is not None else np.zeros(len(window_dates))
            })
            
            window_info.to_csv(anomaly_dir / "window_scores.csv", index=False)
            
            logger.info(f"Completed analysis for anomaly {idx+1}/{len(index_anomalies)}")
            
        except Exception as e:
            logger.error(f"Error analyzing constituents for anomaly {idx+1}: {e}")
            continue
    
    # After completing all anomalies, create a summary JSON file
    try:
        # Collect anomaly statistics
        anomaly_results = []
        anomaly_dirs = list(Path(output_dir).glob("anomaly_*"))
        
        for anomaly_dir in anomaly_dirs:
            # Extract anomaly index from directory name
            anomaly_info = {}
            dir_parts = anomaly_dir.name.split('_')
            anomaly_info['anomaly_id'] = int(dir_parts[1]) if len(dir_parts) > 1 else 0
            
            # Set anomaly date
            if len(dir_parts) > 2:
                try:
                    anomaly_info['anomaly_date'] = dir_parts[2]
                except:
                    anomaly_info['anomaly_date'] = 'unknown'
            
            # Get iForest anomaly count
            iforest_dir = anomaly_dir / 'iforest'
            iforest_anomalies_file = iforest_dir / 'constituent_anomalies.csv'
            if iforest_anomalies_file.exists():
                try:
                    iforest_anomalies = pd.read_csv(iforest_anomalies_file)
                    anomaly_info['iforest_anomaly_count'] = len(iforest_anomalies)
                    
                    # Get top anomalous stocks
                    if 'top_anomalous_stocks' in iforest_anomalies.columns and not iforest_anomalies.empty:
                        top_stocks = iforest_anomalies.iloc[0]['top_anomalous_stocks'].split(',')
                        anomaly_info['top_anomalous_stocks'] = [s.strip() for s in top_stocks]
                except:
                    anomaly_info['iforest_anomaly_count'] = 0
            else:
                anomaly_info['iforest_anomaly_count'] = 0
            
            # Get LOF anomaly count
            lof_dir = anomaly_dir / 'lof'
            lof_anomalies_file = lof_dir / 'constituent_anomalies.csv'
            if lof_anomalies_file.exists():
                try:
                    lof_anomalies = pd.read_csv(lof_anomalies_file)
                    anomaly_info['lof_anomaly_count'] = len(lof_anomalies)
                except:
                    anomaly_info['lof_anomaly_count'] = 0
            else:
                anomaly_info['lof_anomaly_count'] = 0
            
            # Add to results
            anomaly_results.append(anomaly_info)
        
        # Save results to JSON
        results_file = Path(output_dir) / 'constituent_analysis_results.json'
        with open(results_file, 'w') as f:
            json.dump(anomaly_results, f, indent=2)
        
        # Create summary statistics
        summary = {
            'total_anomalies_analyzed': len(anomaly_results),
            'total_iforest_constituent_anomalies': sum(r.get('iforest_anomaly_count', 0) for r in anomaly_results),
            'total_lof_constituent_anomalies': sum(r.get('lof_anomaly_count', 0) for r in anomaly_results),
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save summary to JSON
        summary_file = Path(output_dir) / 'constituent_analysis_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved analysis results to {results_file}")
        logger.info(f"Saved analysis summary to {summary_file}")
        
    except Exception as e:
        logger.error(f"Error creating summary files: {e}")
    
    logger.info("Constituent anomaly analysis completed")


def main():
    """
    Main function to run constituent anomaly analysis.
    """
    parser = argparse.ArgumentParser(description="Analyze constituent stock anomalies around S&P 500 index anomalies")
    parser.add_argument(
        "--index-anomalies", 
        type=str, 
        required=True,
        help="Path to the index anomalies CSV file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=str(config.DATA_DIR / "constituent_analysis"),
        help="Directory to save outputs"
    )
    
    args = parser.parse_args()
    
    # Analyze constituent anomalies
    analyze_constituent_anomalies(args.index_anomalies, args.output)


if __name__ == "__main__":
    # Create detector at global scope to avoid creation in loop
    detector = ConstituentAnomalyDetector(window_size=3)
    main()
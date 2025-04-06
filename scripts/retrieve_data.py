"""
Data preparation and feature engineering functions.
"""
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import glob

from .. import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_ticker_data(file_path):
    """
    Load ticker data from a CSV file.
    
    Args:
        file_path (str or Path): Path to the CSV file
    
    Returns:
        pandas.DataFrame: Loaded data
    """
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Check if we have the expected columns
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns in {file_path}: {missing_columns}")
        
        logger.info(f"Loaded {len(df)} records from {file_path}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return None


def preprocess_data(df):
    """
    Perform basic preprocessing on financial data.
    
    Args:
        df (pandas.DataFrame): Raw financial data
    
    Returns:
        pandas.DataFrame: Preprocessed data
    """
    if df is None or df.empty:
        logger.warning("No data to preprocess")
        return None
    
    try:
        logger.info("Preprocessing data")
        
        # Make a copy to avoid modifying the original
        df_processed = df.copy()
        
        # Forward fill missing price values
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].ffill()
        
        # Fill missing volume with 0
        if 'Volume' in df_processed.columns:
            df_processed['Volume'] = df_processed['Volume'].fillna(0)
        
        # Sort by date
        df_processed = df_processed.sort_index()
        
        # Remove rows where all price columns are NaN
        price_cols = [col for col in ['Open', 'High', 'Low', 'Close'] if col in df_processed.columns]
        df_processed = df_processed.dropna(subset=price_cols, how='all')
        
        logger.info(f"Preprocessing complete, {len(df_processed)} records remaining")
        return df_processed
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        return df


def engineer_features(df, ticker=None, sp500_df=None):
    """
    Engineer features from financial data.
    
    Args:
        df (pandas.DataFrame): Preprocessed financial data
        ticker (str, optional): Ticker symbol for the data
        sp500_df (pandas.DataFrame, optional): S&P 500 index data for market-relative metrics
    
    Returns:
        pandas.DataFrame: Data with engineered features
    """
    if df is None or df.empty:
        logger.warning("No data for feature engineering")
        return None
    
    try:
        logger.info(f"Engineering features for {ticker if ticker else 'unknown ticker'}")
        
        # Make a copy to avoid modifying the original
        df_feat = df.copy()
        
        # 1. Return Metrics
        # Daily returns
        df_feat['daily_return'] = df_feat['Close'].pct_change()
        
        # Log returns
        df_feat['log_return'] = np.log(df_feat['Close'] / df_feat['Close'].shift(1))
        
        # 2. Volume Metrics
        # Volume change
        df_feat['volume_change'] = df_feat['Volume'].pct_change()
        
        # Relative volume (compared to 10-day moving average)
        df_feat['relative_volume'] = df_feat['Volume'] / df_feat['Volume'].rolling(
            window=config.WINDOW_SIZES['volume']).mean()
        
        # 3. Price Movement Characteristics
        # High-low range
        df_feat['high_low_range'] = (df_feat['High'] - df_feat['Low']) / df_feat['Low']
        
        # 4. Market-Relative Metrics (if S&P 500 data is provided)
        if sp500_df is not None and ticker != config.SP500_TICKER:
            try:
                # Ensure the indices align
                sp500_aligned = sp500_df.reindex(df_feat.index)
                
                # Calculate excess return (stock return - market return)
                df_feat['excess_return'] = df_feat['daily_return'] - sp500_aligned['daily_return']
                
                logger.info("Added market-relative metrics")
            except Exception as e:
                logger.error(f"Error calculating market-relative metrics: {e}")
        
        # 5. Statistical Normalization
        
        # Z-score normalization for return-based features within a rolling window
        return_features = ['daily_return', 'log_return']
        for feature in return_features:
            if feature in df_feat.columns:
                # Calculate rolling mean and std
                roll_mean = df_feat[feature].rolling(window=config.WINDOW_SIZES['returns']).mean()
                roll_std = df_feat[feature].rolling(window=config.WINDOW_SIZES['returns']).std()
                
                # Calculate z-score
                df_feat[f'{feature}_zscore'] = (df_feat[feature] - roll_mean) / roll_std
        
        # Z-score normalization for volume-based features
        volume_features = ['volume_change', 'relative_volume']
        for feature in volume_features:
            if feature in df_feat.columns:
                # Calculate rolling mean and std
                roll_mean = df_feat[feature].rolling(window=config.WINDOW_SIZES['volume']).mean()
                roll_std = df_feat[feature].rolling(window=config.WINDOW_SIZES['volume']).std()
                
                # Calculate z-score
                df_feat[f'{feature}_zscore'] = (df_feat[feature] - roll_mean) / roll_std
        
        # Logarithmic transformation for volume
        if 'Volume' in df_feat.columns:
            df_feat['log_volume'] = np.log1p(df_feat['Volume'])  # log(1+x) to handle zeros
        
        # Remove the first few rows that have NaN due to rolling window calculations
        df_feat = df_feat.iloc[max(config.WINDOW_SIZES.values()):]
        
        logger.info(f"Feature engineering complete, {len(df_feat)} records with {len(df_feat.columns)} features")
        return df_feat
    
    except Exception as e:
        logger.error(f"Error engineering features: {e}")
        return df


def save_processed_data(df, ticker, output_dir):
    """
    Save processed data to a CSV file.
    
    Args:
        df (pandas.DataFrame): Processed data
        ticker (str): Ticker symbol
        output_dir (Path): Directory to save the file
    
    Returns:
        Path: Path to the saved file or None if operation failed
    """
    if df is None or df.empty:
        logger.warning(f"No processed data to save for {ticker}")
        return None
    
    try:
        # Create the output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Sanitize ticker for filename
        ticker_safe = ticker.replace('^', 'index_').replace('/', '_')
        filename = output_dir / f"{ticker_safe}_processed.csv"
        
        # Save the data
        df.to_csv(filename)
        logger.info(f"Saved processed data for {ticker} to {filename}")
        return filename
    
    except Exception as e:
        logger.error(f"Error saving processed data for {ticker}: {e}")
        return None


def create_feature_dataset(df_list, ticker_list, feature_columns):
    """
    Create a combined dataset with selected features from multiple tickers.
    
    Args:
        df_list (list): List of DataFrames with processed data
        ticker_list (list): List of ticker symbols corresponding to the DataFrames
        feature_columns (list): List of feature columns to include
        
    Returns:
        pandas.DataFrame: Combined dataset with selected features
    """
    if not df_list or not ticker_list or len(df_list) != len(ticker_list):
        logger.error("Invalid inputs for creating feature dataset")
        return None
    
    try:
        logger.info("Creating combined feature dataset")
        
        # Create an empty list to store DataFrames for each ticker
        dfs = []
        
        for df, ticker in zip(df_list, ticker_list):
            if df is None or df.empty:
                logger.warning(f"Skipping {ticker} due to empty data")
                continue
                
            # Select the required features
            available_features = [col for col in feature_columns if col in df.columns]
            
            if len(available_features) != len(feature_columns):
                missing = set(feature_columns) - set(available_features)
                logger.warning(f"Missing features for {ticker}: {missing}")
            
            if not available_features:
                logger.warning(f"No required features available for {ticker}")
                continue
                
            # Create a copy with only the required features
            df_features = df[available_features].copy()
            
            # Add ticker identifier
            df_features['ticker'] = ticker
            
            dfs.append(df_features)
        
        if not dfs:
            logger.warning("No data available for combined dataset")
            return None
            
        # Combine all DataFrames
        combined_df = pd.concat(dfs, axis=0)
        
        logger.info(f"Created combined dataset with {len(combined_df)} records and {len(combined_df.columns)} columns")
        return combined_df
    
    except Exception as e:
        logger.error(f"Error creating feature dataset: {e}")
        return None


def process_all_data():
    """
    Process all downloaded ticker data.
    
    Returns:
        dict: Dictionary with paths to processed files
    """
    logger.info("Starting data processing")
    
    # Get all CSV files in the raw data directory
    raw_files = list(config.RAW_DATA_DIR.glob('*.csv'))
    
    if not raw_files:
        logger.warning("No raw data files found")
        return {}
    
    # First, process the S&P 500 index data
    sp500_file = next((f for f in raw_files if config.SP500_TICKER.replace('^', 'index_') in f.name), None)
    
    if sp500_file is None:
        logger.warning(f"S&P 500 index data not found, expected filename with {config.SP500_TICKER.replace('^', 'index_')}")
        sp500_df = None
        sp500_processed_path = None
    else:
        # Load and process S&P 500 data
        sp500_raw = load_ticker_data(sp500_file)
        sp500_preprocessed = preprocess_data(sp500_raw)
        sp500_df = engineer_features(sp500_preprocessed, ticker=config.SP500_TICKER)
        sp500_processed_path = save_processed_data(sp500_df, config.SP500_TICKER, config.PROCESSED_DATA_DIR)
    
    # Process constituent stocks
    processed_files = []
    
    for file_path in raw_files:
        # Skip the S&P 500 index file as it's already processed
        if sp500_file and file_path.sameas(sp500_file):
            continue
        
        # Extract ticker from filename
        ticker = file_path.stem
        
        # Load and process the data
        df_raw = load_ticker_data(file_path)
        df_preprocessed = preprocess_data(df_raw)
        df_features = engineer_features(df_preprocessed, ticker=ticker, sp500_df=sp500_df)
        
        # Save the processed data
        processed_path = save_processed_data(df_features, ticker, config.PROCESSED_DATA_DIR)
        
        if processed_path:
            processed_files.append(processed_path)
    
    logger.info(f"Processed {len(processed_files)} constituent stock files")
    
    # Create a feature matrix for all stocks combined
    # We'll implement this later as needed
    
    return {
        'sp500_processed': sp500_processed_path,
        'constituent_processed': processed_files
    }


def create_subsequence_dataset(df, subsequence_length=3, step=1):
    """
    Create a dataset of subsequences from time series data.
    This is preparation for anomaly detection on subsequences as mentioned in the additional notes.
    
    Args:
        df (pandas.DataFrame): Time series data
        subsequence_length (int): Length of each subsequence
        step (int): Step size between subsequences (1 for overlapping, subsequence_length for non-overlapping)
    
    Returns:
        list: List of subsequence DataFrames
    """
    if df is None or df.empty:
        logger.warning("No data for creating subsequences")
        return []
    
    try:
        logger.info(f"Creating subsequences of length {subsequence_length} with step {step}")
        
        # Get total number of possible subsequences
        n_samples = len(df)
        n_subsequences = max(0, (n_samples - subsequence_length) // step + 1)
        
        if n_subsequences == 0:
            logger.warning(f"Not enough data points ({n_samples}) for subsequence length {subsequence_length}")
            return []
        
        subsequences = []
        
        for i in range(0, n_samples - subsequence_length + 1, step):
            # Extract subsequence
            subsequence = df.iloc[i:i+subsequence_length].copy()
            
            # Add metadata
            subsequence.attrs['start_idx'] = i
            subsequence.attrs['end_idx'] = i + subsequence_length - 1
            subsequence.attrs['start_date'] = df.index[i]
            subsequence.attrs['end_date'] = df.index[i + subsequence_length - 1]
            
            subsequences.append(subsequence)
        
        logger.info(f"Created {len(subsequences)} subsequences")
        return subsequences
    
    except Exception as e:
        logger.error(f"Error creating subsequences: {e}")
        return []


def create_multi_ts_subsequences(df_list, ticker_list, feature_columns, subsequence_length=3, step=1):
    """
    Create subsequences from multiple time series (stocks) for matrix-based anomaly detection.
    
    Args:
        df_list (list): List of DataFrames with processed data
        ticker_list (list): List of ticker symbols corresponding to the DataFrames
        feature_columns (list): List of feature columns to include
        subsequence_length (int): Length of each subsequence
        step (int): Step size between subsequences
        
    Returns:
        list: List of dictionaries, each containing a subsequence matrix and metadata
    """
    if not df_list or not ticker_list or len(df_list) != len(ticker_list):
        logger.error("Invalid inputs for creating multi-TS subsequences")
        return []
    
    try:
        logger.info(f"Creating multi-TS subsequences for {len(df_list)} stocks")
        
        # Align all DataFrames to a common date range
        # This is important for creating subsequence matrices
        common_idx = None
        aligned_dfs = []
        aligned_tickers = []
        
        for df, ticker in zip(df_list, ticker_list):
            if df is None or df.empty:
                logger.warning(f"Skipping {ticker} due to empty data")
                continue
                
            # Select only the required features
            available_features = [col for col in feature_columns if col in df.columns]
            if not available_features:
                logger.warning(f"No required features available for {ticker}")
                continue
                
            df_features = df[available_features].copy()
            
            if common_idx is None:
                common_idx = df_features.index
            else:
                common_idx = common_idx.intersection(df_features.index)
            
            aligned_dfs.append(df_features)
            aligned_tickers.append(ticker)
        
        if len(common_idx) < subsequence_length:
            logger.warning(f"Not enough common dates ({len(common_idx)}) for subsequence length {subsequence_length}")
            return []
        
        # Reindex all DataFrames to the common date range
        for i in range(len(aligned_dfs)):
            aligned_dfs[i] = aligned_dfs[i].reindex(common_idx)
        
        # Create subsequences
        n_dates = len(common_idx)
        n_subsequences = max(0, (n_dates - subsequence_length) // step + 1)
        
        subsequence_matrices = []
        
        for i in range(0, n_dates - subsequence_length + 1, step):
            # Create a subsequence matrix for this time window
            # Shape: (n_stocks, subsequence_length, n_features)
            subseq_data = []
            
            for j, df in enumerate(aligned_dfs):
                # Extract subsequence for this stock
                stock_subseq = df.iloc[i:i+subsequence_length].values
                subseq_data.append(stock_subseq)
            
            # Convert to numpy array for easier manipulation
            subseq_matrix = np.array(subseq_data)
            
            # Add metadata
            metadata = {
                'start_idx': i,
                'end_idx': i + subsequence_length - 1,
                'start_date': common_idx[i],
                'end_date': common_idx[i + subsequence_length - 1],
                'tickers': aligned_tickers,
                'features': feature_columns
            }
            
            subsequence_matrices.append({
                'matrix': subseq_matrix,
                'metadata': metadata
            })
        
        logger.info(f"Created {len(subsequence_matrices)} multi-TS subsequence matrices")
        return subsequence_matrices
    
    except Exception as e:
        logger.error(f"Error creating multi-TS subsequences: {e}")
        return []


def save_subsequence_dataset(subsequences, output_path, prefix='subsequence'):
    """
    Save subsequence dataset to disk.
    
    Args:
        subsequences (list): List of subsequences
        output_path (Path): Directory to save the subsequences
        prefix (str): Prefix for the filenames
        
    Returns:
        list: List of paths to saved files
    """
    if not subsequences:
        logger.warning("No subsequences to save")
        return []
    
    try:
        # Create the output directory if it doesn't exist
        output_path.mkdir(exist_ok=True, parents=True)
        
        saved_paths = []
        
        for i, subsequence in enumerate(subsequences):
            # For simple subsequences (DataFrames)
            if isinstance(subsequence, pd.DataFrame):
                filename = output_path / f"{prefix}_{i}.csv"
                subsequence.to_csv(filename)
                saved_paths.append(filename)
            
            # For multi-TS subsequence matrices
            elif isinstance(subsequence, dict) and 'matrix' in subsequence and 'metadata' in subsequence:
                filename = output_path / f"{prefix}_{i}.npz"
                np.savez(
                    filename, 
                    matrix=subsequence['matrix'], 
                    metadata=json.dumps(subsequence['metadata'])
                )
                saved_paths.append(filename)
            
            else:
                logger.warning(f"Unknown subsequence type for index {i}, skipping")
        
        logger.info(f"Saved {len(saved_paths)} subsequences to {output_path}")
        return saved_paths
    
    except Exception as e:
        logger.error(f"Error saving subsequences: {e}")
        return []


def main():
    """
    Main function to process all data and create feature-engineered datasets.
    """
    logger.info("Starting data preparation process")
    
    # Process all ticker data
    processed_data = process_all_data()
    
    logger.info("Data preparation completed")
    return processed_data


if __name__ == "__main__":
    main()
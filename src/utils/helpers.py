"""
Helper utility functions for the anomaly detection project.
"""
import os
import logging
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from .. import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_file_list(directory, pattern='*.csv'):
    """
    Get a list of files in a directory matching a pattern.
    
    Args:
        directory (str or Path): Directory to search
        pattern (str): Glob pattern to match
        
    Returns:
        list: List of Path objects
    """
    directory = Path(directory)
    return list(directory.glob(pattern))


def load_subsequence(file_path):
    """
    Load a subsequence from a file.
    
    Args:
        file_path (str or Path): Path to the subsequence file
        
    Returns:
        dict or pandas.DataFrame: Loaded subsequence
    """
    file_path = Path(file_path)
    
    try:
        if file_path.suffix == '.csv':
            # Load CSV subsequence
            return pd.read_csv(file_path, index_col=0, parse_dates=True)
        elif file_path.suffix == '.npz':
            # Load NPZ subsequence (multi-TS)
            data = np.load(file_path, allow_pickle=True)
            matrix = data['matrix']
            
            try:
                metadata = json.loads(data['metadata'].item())
            except:
                metadata = {'info': 'Metadata parsing failed'}
            
            return {'matrix': matrix, 'metadata': metadata}
        else:
            logger.warning(f"Unsupported file format: {file_path.suffix}")
            return None
    
    except Exception as e:
        logger.error(f"Error loading subsequence from {file_path}: {e}")
        return None


def get_trading_dates(start_date, end_date):
    """
    Get a list of trading dates between start_date and end_date.
    This is a simplified version that excludes weekends but not holidays.
    
    Args:
        start_date (str or datetime): Start date
        end_date (str or datetime): End date
        
    Returns:
        list: List of datetime objects representing trading dates
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate all dates in the range
    all_dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
    
    # Filter out weekends
    trading_dates = [date for date in all_dates if date.weekday() < 5]  # Monday to Friday
    
    return trading_dates


def plot_time_series(df, columns, title=None, figsize=(12, 6), save_path=None):
    """
    Plot time series data.
    
    Args:
        df (pandas.DataFrame): DataFrame containing time series data
        columns (list): List of columns to plot
        title (str, optional): Plot title
        figsize (tuple, optional): Figure size
        save_path (str or Path, optional): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for column in columns:
        if column in df.columns:
            df[column].plot(ax=ax, label=column)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    
    if title:
        ax.set_title(title)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_subsequence_matrix(subsequence_dict, feature_idx=0, figsize=(12, 8), save_path=None):
    """
    Plot a heatmap of a subsequence matrix for visual analysis.
    
    Args:
        subsequence_dict (dict): Dictionary containing 'matrix' and 'metadata'
        feature_idx (int, optional): Index of the feature to plot
        figsize (tuple, optional): Figure size
        save_path (str or Path, optional): Path to save the figure
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if not isinstance(subsequence_dict, dict) or 'matrix' not in subsequence_dict or 'metadata' not in subsequence_dict:
        logger.error("Invalid subsequence dictionary format")
        return None
    
    matrix = subsequence_dict['matrix']
    metadata = subsequence_dict['metadata']
    
    # Extract information from metadata
    tickers = metadata.get('tickers', [f'Stock {i}' for i in range(matrix.shape[0])])
    features = metadata.get('features', [f'Feature {i}' for i in range(matrix.shape[2])])
    start_date = metadata.get('start_date', 'Unknown')
    end_date = metadata.get('end_date', 'Unknown')
    
    # Check if feature_idx is valid
    if feature_idx >= matrix.shape[2]:
        logger.warning(f"Feature index {feature_idx} out of range, using index 0")
        feature_idx = 0
    
    # Extract the matrix slice for the selected feature
    feature_matrix = matrix[:, :, feature_idx]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        feature_matrix, 
        cmap='viridis', 
        xticklabels=range(matrix.shape[1]),
        yticklabels=tickers,
        ax=ax
    )
    
    # Set labels and title
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Stock')
    feature_name = features[feature_idx] if feature_idx < len(features) else f'Feature {feature_idx}'
    ax.set_title(f'{feature_name} - {start_date} to {end_date}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def ensure_directory_exists(path):
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path (str or Path): Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def format_ticker_for_filename(ticker):
    """
    Format a ticker symbol for use in a filename.
    
    Args:
        ticker (str): Ticker symbol
        
    Returns:
        str: Formatted ticker
    """
    return ticker.replace('^', 'index_').replace('/', '_').replace('-', '_')
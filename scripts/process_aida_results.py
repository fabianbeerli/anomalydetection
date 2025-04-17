"""
Process and analyze AIDA anomaly detection results for S&P 500.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src import config

def load_aida_scores(scores_file):
    """
    Load AIDA anomaly scores from a file.
    
    Args:
        scores_file (str or Path): Path to the AIDA scores file
    
    Returns:
        pd.Series: Series of anomaly scores
    """
    try:
        # Read the scores file, skipping the first line (total count)
        scores = pd.read_csv(scores_file, skiprows=1, header=None, names=['score'])
        return scores['score']
    except Exception as e:
        print(f"Error loading AIDA scores: {e}")
        return None

def load_aida_anomalies(anomalies_file):
    """
    Load detected anomalies from AIDA results.
    
    Args:
        anomalies_file (str or Path): Path to the AIDA anomalies file
    
    Returns:
        pd.DataFrame: DataFrame of anomalies with index and score
    """
    try:
        anomalies = pd.read_csv(anomalies_file)
        return anomalies
    except Exception as e:
        print(f"Error loading AIDA anomalies: {e}")
        return None

def visualize_aida_scores(scores, output_dir):
    """
    Create visualizations of AIDA anomaly scores.
    
    Args:
        scores (pd.Series): Series of anomaly scores
        output_dir (Path): Directory to save visualizations
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Histogram of scores
    plt.figure(figsize=(10, 6))
    scores.hist(bins=50)
    plt.title('Distribution of AIDA Anomaly Scores')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(output_dir / 'aida_scores_histogram.png')
    plt.close()
    
    # Box plot of scores
    plt.figure(figsize=(10, 6))
    scores.plot(kind='box')
    plt.title('Boxplot of AIDA Anomaly Scores')
    plt.ylabel('Anomaly Score')
    plt.tight_layout()
    plt.savefig(output_dir / 'aida_scores_boxplot.png')
    plt.close()
    
    # Basic statistics
    print("AIDA Anomaly Scores Statistics:")
    print(scores.describe())

def process_aida_results(input_file):
    """
    Process AIDA results for a given input file.
    
    Args:
        input_file (str or Path): Path to the input CSV file
    """
    # Convert to Path object
    input_file = Path(input_file)
    
    # Construct score and anomaly file paths
    scores_file = input_file.with_suffix(input_file.suffix + '_AIDA_scores.dat')
    anomalies_file = input_file.with_suffix(input_file.suffix + '_AIDA_anomalies.dat')
    
    # Output directory for visualizations
    output_dir = config.DATA_DIR / 'aida_results'
    
    # Load scores
    scores = load_aida_scores(scores_file)
    
    if scores is not None:
        # Visualize scores
        visualize_aida_scores(scores, output_dir)
    
    # Load anomalies
    anomalies = load_aida_anomalies(anomalies_file)
    
    if anomalies is not None:
        print("\nDetected Anomalies:")
        print(anomalies)
        
        # Optional: Save anomalies to CSV
        anomalies.to_csv(output_dir / 'aida_anomalies.csv', index=False)

def main():
    """
    Main function to process AIDA results for S&P 500 data.
    """
    # Path to the processed S&P 500 data
    sp500_processed_file = config.PROCESSED_DATA_DIR / 'sp500_index_processed.csv'
    
    # Process AIDA results
    process_aida_results(sp500_processed_file)

if __name__ == "__main__":
    main()
def visualize_anomalies(data, anomaly_results, execution_times, output_dir):
    """
    Create a comprehensive visualization of anomalies with offset markers and connecting lines.
    
    Args:
        data (pandas.DataFrame): Original time series data
        anomaly_results (dict): Dictionary of anomaly results for each algorithm
        execution_times (dict): Dictionary of execution times
        output_dir (Path): Directory to save visualizations
    """
    # Ensure output directory exists
    ensure_directory_exists(output_dir)
    
    # Create a figure with two subplots
    plt.figure(figsize=(16, 10))
    
    # Plot 1: Time series with anomalies
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['Close'], label='S&P 500 Close Price', color='blue', alpha=0.7)
    plt.title('S&P 500 Subsequence Anomalies')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    
    # Markers and colors for each algorithm
    markers = {
        'aida': ('o', 'red', 'AIDA', -1),
        'iforest': ('^', 'green', 'Isolation Forest', 0),
        'lof': ('s', 'purple', 'LOF', 1)
    }
    
    # Calculate vertical offset
    price_range = data['Close'].max() - data['Close'].min()
    vertical_offset = price_range * 0.1  # 10% of price range
    
    # Detailed anomaly statistics
    anomaly_stats = {}
    
    # Collect anomalies for each algorithm
    for algo, (marker, color, label, position) in markers.items():
        if algo in anomaly_results and not anomaly_results[algo].empty:
            # Convert dates to datetime
            anomaly_results[algo]['date'] = pd.to_datetime(anomaly_results[algo]['start_date'])
            
            # Find the actual close prices for these dates
            matched_mask = data.index.isin(anomaly_results[algo]['date'])
            matched_prices = data.loc[matched_mask, 'Close']
            matched_dates = data.index[matched_mask]
            
            # Calculate statistics for this algorithm's anomaly scores
            all_scores = anomaly_results[algo]['score']
            mean_score = all_scores.mean()
            std_score = all_scores.std()
            
            # Store detailed statistics
            anomaly_stats[algo] = {
                'count': len(matched_dates),
                'mean_score': mean_score,
                'std_score': std_score,
                'min_score': all_scores.min(),
                'max_score': all_scores.max()
            }
            
            # Calculate vertical position
            offset_prices = matched_prices + (position * vertical_offset)
            
            # Plot connecting lines
            for date, orig_price, offset_price in zip(matched_dates, matched_prices, offset_prices):
                plt.plot([date, date], [orig_price, offset_price], 
                         color=color, linestyle=':', linewidth=1, alpha=0.5)
            
            # Plot offset markers
            plt.scatter(
                matched_dates, 
                offset_prices, 
                marker=marker, 
                color=color, 
                label=label, 
                s=100, 
                alpha=0.7
            )
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plot_file = output_dir / "subsequence_anomalies_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {plot_file}")
    
    # Detailed summary file
    summary_file = output_dir / "anomaly_detection_detailed_summary.txt"
    
    # Ensure the directory exists
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write detailed statistics to file
    with open(summary_file, 'w') as f:
        f.write("Detailed Anomaly Detection Statistics\n")
        f.write("====================================\n\n")
        
        for algo, stats in anomaly_stats.items():
            f.write(f"{algo.upper()} Anomaly Statistics:\n")
            for key, value in stats.items():
                f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
            f.write("\n")
    
    logger.info(f"Saved detailed statistics to {summary_file}")
    
    # Print to console as well
    print("\nDetailed Anomaly Detection Statistics:")
    for algo, stats in anomaly_stats.items():
        print(f"\n{algo.upper()}:")
        for key, value in stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
            
def visualize_anomalies(data, anomaly_results, execution_times, output_dir):
    """
    Create a comprehensive visualization of anomalies with offset markers and connecting lines.
    
    Args:
        data (pandas.DataFrame): Original time series data
        anomaly_results (dict): Dictionary of anomaly results for each algorithm
        execution_times (dict): Dictionary of execution times
        output_dir (Path): Directory to save visualizations
    """
    # Ensure output directory exists
    ensure_directory_exists(output_dir)
    
    # Create a figure with two subplots
    plt.figure(figsize=(16, 10))
    
    # Plot 1: Time series with anomalies
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['Close'], label='S&P 500 Close Price', color='blue', alpha=0.7)
    plt.title('S&P 500 Subsequence Anomalies')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    
    # Markers and colors for each algorithm
    markers = {
        'aida': ('o', 'red', 'AIDA', -1),
        'iforest': ('^', 'green', 'Isolation Forest', 0),
        'lof': ('s', 'purple', 'LOF', 1)
    }
    
    # Calculate vertical offset
    price_range = data['Close'].max() - data['Close'].min()
    vertical_offset = price_range * 0.1  # 10% of price range
    
    # Detailed anomaly statistics
    anomaly_stats = {}
    
    # Collect anomalies for each algorithm
    for algo, (marker, color, label, position) in markers.items():
        if algo in anomaly_results and not anomaly_results[algo].empty:
            # Convert dates to datetime
            anomaly_results[algo]['date'] = pd.to_datetime(anomaly_results[algo]['start_date'])
            
            # Find the actual close prices for these dates
            matched_mask = data.index.isin(anomaly_results[algo]['date'])
            matched_prices = data.loc[matched_mask, 'Close']
            matched_dates = data.index[matched_mask]
            
            # Calculate statistics for this algorithm's anomaly scores
            all_scores = anomaly_results[algo]['score']
            mean_score = all_scores.mean()
            std_score = all_scores.std()
            
            # Store detailed statistics
            anomaly_stats[algo] = {
                'count': len(matched_dates),
                'mean_score': mean_score,
                'std_score': std_score,
                'min_score': all_scores.min(),
                'max_score': all_scores.max()
            }
            
            # Calculate vertical position
            offset_prices = matched_prices + (position * vertical_offset)
            
            # Plot connecting lines
            for date, orig_price, offset_price in zip(matched_dates, matched_prices, offset_prices):
                plt.plot([date, date], [orig_price, offset_price], 
                         color=color, linestyle=':', linewidth=1, alpha=0.5)
            
            # Plot offset markers
            plt.scatter(
                matched_dates, 
                offset_prices, 
                marker=marker, 
                color=color, 
                label=label, 
                s=100, 
                alpha=0.7
            )
    
    plt.legend()
    
    # Detailed summary file
    summary_file = output_dir / "anomaly_detection_detailed_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Detailed Anomaly Detection Statistics\n")
        f.write("====================================\n\n")
        
        for algo, stats in anomaly_stats.items():
            f.write(f"{algo.upper()} Anomaly Statistics:\n")
            f.write(f"  Count: {stats['count']}\n")
            f.write(f"  Mean Score: {stats['mean_score']:.4f}\n")
            f.write(f"  Standard Deviation: {stats['std_score']:.4f}\n")
            f.write(f"  Min Score: {stats['min_score']:.4f}\n")
            f.write(f"  Max Score: {stats['max_score']:.4f}\n\n")
    
    logger.info(f"Detailed anomaly statistics saved to {summary_file}")
    
    # Print to console as well
    print("\nDetailed Anomaly Detection Statistics:")
    for algo, stats in anomaly_stats.items():
        print(f"\n{algo.upper()}:")
        for key, value in stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
#!/usr/bin/env python
"""
Compare and visualize anomaly detection results for subsequence analysis.
"""
import os
import sys
import json
import logging
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data.preparation import load_ticker_data
from src.utils.helpers import ensure_directory_exists

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_anomaly_results(results_dir):
    """
    Load anomaly results for different algorithms.
    
    Args:
        results_dir (Path): Directory containing algorithm results
        
    Returns:
        dict: Dictionary of anomaly results for each algorithm
    """
    algorithms = ['aida', 'iforest', 'lof']
    anomaly_results = {}
    
    for algo in algorithms:
        try:
            # Load anomalies
            anomalies_file = results_dir / algo / f"{algo}_anomalies.csv"
            if anomalies_file.exists():
                anomalies_df = pd.read_csv(anomalies_file)
                anomaly_results[algo] = anomalies_df
            else:
                logger.warning(f"No anomalies file found for {algo}")
        except Exception as e:
            logger.error(f"Error loading {algo} anomalies: {e}")
    
    return anomaly_results

def load_execution_times(results_dir):
    """
    Load execution times for different algorithms.
    
    Args:
        results_dir (Path): Directory containing algorithm results
        
    Returns:
        dict: Dictionary of execution times
    """
    try:
        execution_times_file = results_dir / "execution_times.json"
        if execution_times_file.exists():
            with open(execution_times_file, 'r') as f:
                return json.load(f)
        else:
            logger.warning("No execution times file found")
            return {}
    except Exception as e:
        logger.error(f"Error loading execution times: {e}")
        return {}

def visualize_anomalies(data, anomaly_results, execution_times, output_dir):
    """
    Create a comprehensive visualization of anomalies with offset markers and connecting lines.
    
    Args:
        data (pandas.DataFrame): Original time series data
        anomaly_results (dict): Dictionary of anomaly results for each algorithm
        execution_times (dict): Dictionary of execution times
        output_dir (Path): Directory to save visualizations
    """
    # Ensure output directory exists
    ensure_directory_exists(output_dir)
    
    # Create a figure with two subplots
    plt.figure(figsize=(16, 10))
    
    # Plot 1: Time series with anomalies
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['Close'], label='S&P 500 Close Price', color='blue', alpha=0.7)
    plt.title('S&P 500 Subsequence Anomalies')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    
    # Markers and colors for each algorithm
    markers = {
        'aida': ('o', 'red', 'AIDA', -1),
        'iforest': ('^', 'green', 'Isolation Forest', 0),
        'lof': ('s', 'purple', 'LOF', 1)
    }
    
    # Calculate vertical offset
    price_range = data['Close'].max() - data['Close'].min()
    vertical_offset = price_range * 0.1  # 10% of price range
    
    # Collect anomalies for each algorithm
    for algo, (marker, color, label, position) in markers.items():
        if algo in anomaly_results and not anomaly_results[algo].empty:
            # Convert dates to datetime
            anomaly_results[algo]['date'] = pd.to_datetime(anomaly_results[algo]['start_date'])
            
            # Find the actual close prices for these dates
            matched_mask = data.index.isin(anomaly_results[algo]['date'])
            matched_prices = data.loc[matched_mask, 'Close']
            matched_dates = data.index[matched_mask]
            
            # Calculate vertical position
            offset_prices = matched_prices + (position * vertical_offset)
            
            # Plot connecting lines
            for date, orig_price, offset_price in zip(matched_dates, matched_prices, offset_prices):
                plt.plot([date, date], [orig_price, offset_price], 
                         color=color, linestyle=':', linewidth=1, alpha=0.5)
            
            # Plot offset markers
            plt.scatter(
                matched_dates, 
                offset_prices, 
                marker=marker, 
                color=color, 
                label=label, 
                s=100, 
                alpha=0.7
            )
    
    plt.legend()
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Execution Times
    plt.subplot(2, 1, 2)
    plt.title('Algorithm Execution Times')
    plt.bar(
        list(execution_times.keys()), 
        list(execution_times.values()), 
        color=['red', 'green', 'purple']
    )
    plt.xlabel('Algorithm')
    plt.ylabel('Execution Time (seconds)')
    
    # Add exact times as text on top of bars
    for i, (algo, time_val) in enumerate(execution_times.items()):
        plt.text(i, time_val, f'{time_val:.4f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    output_file = output_dir / "subsequence_anomalies_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    logger.info(f"Saved comprehensive anomaly visualization to {output_file}")
    
    # Detailed summary of anomalies
    summary_file = output_dir / "anomaly_detection_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Subsequence Anomaly Detection Summary\n")
        f.write("====================================\n\n")
        
        f.write("Execution Times:\n")
        for algo, time_val in execution_times.items():
            f.write(f"{algo.upper()}: {time_val:.4f} seconds\n")
        
        f.write("\nAnomaly Counts:\n")
        for algo, anomalies in anomaly_results.items():
            f.write(f"{algo.upper()}: {len(anomalies)} anomalies\n")
    
    logger.info(f"Saved anomaly detection summary to {summary_file}")

def main():
    """
    Main function to compare and visualize subsequence anomaly detection results.
    """
    parser = argparse.ArgumentParser(description="Compare subsequence anomaly detection results")
    parser.add_argument(
        "--results", 
        type=str, 
        default=str(config.DATA_DIR / "subsequence_results"),
        help="Directory containing algorithm results"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default=str(config.PROCESSED_DATA_DIR / "index_GSPC_processed.csv"),
        help="Path to the original processed S&P 500 data"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=str(config.DATA_DIR / "subsequence_analysis"),
        help="Directory to save analysis results"
    )
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    results_dir = Path(args.results)
    data_file = Path(args.data)
    output_dir = Path(args.output)
    
    # Ensure output directory exists
    ensure_directory_exists(output_dir)
    
    # Load original data
    logger.info(f"Loading S&P 500 data from {data_file}")
    data = load_ticker_data(data_file)
    
    if data is None or data.empty:
        logger.error("Failed to load S&P 500 data")
        return
    
    # Load anomaly results
    logger.info(f"Loading anomaly results from {results_dir}")
    anomaly_results = load_anomaly_results(results_dir)
    
    # Load execution times
    execution_times = load_execution_times(results_dir)
    
    # Create detailed summary file even if no anomalies
    summary_file = output_dir / "anomaly_detection_detailed_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("Detailed Anomaly Detection Statistics\n")
        f.write("====================================\n\n")
        
        if not anomaly_results:
            f.write("No anomaly results found for any algorithm.\n")
        else:
            for algo, results in anomaly_results.items():
                f.write(f"{algo.upper()} Anomaly Statistics:\n")
                f.write(f"  Total Anomalies: {len(results)}\n")
                
                if 'score' in results.columns:
                    f.write(f"  Mean Score: {results['score'].mean():.4f}\n")
                    f.write(f"  Score Standard Deviation: {results['score'].std():.4f}\n")
                    f.write(f"  Min Score: {results['score'].min():.4f}\n")
                    f.write(f"  Max Score: {results['score'].max():.4f}\n")
                f.write("\n")
    
    logger.info(f"Saved detailed statistics to {summary_file}")
    
    # Visualize and compare results if anomalies exist
    if anomaly_results:
        visualize_anomalies(data, anomaly_results, execution_times, output_dir)
    else:
        logger.warning("No anomalies to visualize")

if __name__ == "__main__":
    main()
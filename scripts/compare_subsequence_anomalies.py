#!/usr/bin/env python
"""
Compare and visualize anomaly detection results for subsequence analysis.
Supports selecting specific window sizes and overlap settings.
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
from matplotlib.dates import DateFormatter

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


def load_original_data(data_file):
    """
    Load the original S&P 500 data.
    
    Args:
        data_file (Path): Path to the original data file
        
    Returns:
        pandas.DataFrame: Original time series data
    """
    try:
        logger.info(f"Loading original data from {data_file}")
        df = load_ticker_data(data_file)
        
        if df is None or df.empty:
            logger.error(f"Failed to load data from {data_file}")
            return None
        
        # Ensure we have a 'Close' column for visualization
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        
        logger.info(f"Loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading original data: {e}")
        return None


def load_anomaly_results(results_dir, window_size, overlap_type):
    """
    Load anomaly results for different algorithms.
    
    Args:
        results_dir (Path): Base directory for results
        window_size (int): Window size to analyze
        overlap_type (str): 'overlap' or 'nonoverlap'
        
    Returns:
        dict: Dictionary of anomaly results for each algorithm
    """
    algorithms = ['aida', 'iforest', 'lof']
    anomaly_results = {}
    
    for algo in algorithms:
        try:
            # Corrected path structure: results_dir/algo/w{window_size}_{overlap_type}
            config_dir = results_dir / algo / f"w{window_size}_{overlap_type}"
            
            # Load anomalies
            anomalies_file = config_dir / f"{algo}_anomalies.csv"
            if anomalies_file.exists():
                anomalies_df = pd.read_csv(anomalies_file)
                # Convert date columns to datetime
                for date_col in ['start_date', 'end_date']:
                    if date_col in anomalies_df.columns:
                        anomalies_df[date_col] = pd.to_datetime(anomalies_df[date_col])
                anomaly_results[algo] = anomalies_df
                logger.info(f"Loaded {len(anomalies_df)} anomalies for {algo}")
            else:
                logger.warning(f"No anomalies file found for {algo} at {anomalies_file}")
        except Exception as e:
            logger.error(f"Error loading {algo} anomalies: {e}")
    
    return anomaly_results


def load_execution_times(results_dir, window_size, overlap_type):
    """
    Load execution times for different algorithms.
    
    Args:
        results_dir (Path): Base directory for results
        window_size (int): Window size to analyze
        overlap_type (str): 'overlap' or 'nonoverlap'
        
    Returns:
        dict: Dictionary of execution times
    """
    algorithms = ['aida', 'iforest', 'lof']
    execution_times = {}
    
    for algo in algorithms:
        try:
            # Corrected path structure: results_dir/algo/w{window_size}_{overlap_type}
            config_dir = results_dir / algo / f"w{window_size}_{overlap_type}"
            
            # Try JSON first
            execution_times_file = config_dir / "execution_times.json"
            if execution_times_file.exists():
                with open(execution_times_file, 'r') as f:
                    return json.load(f)
            
            # Fallback to individual time files
            time_file = config_dir / f"{algo}_execution_time.txt"
            if time_file.exists():
                with open(time_file, 'r') as f:
                    execution_times[algo] = float(f.read().strip())
            
        except Exception as e:
            logger.error(f"Error loading {algo} execution time: {e}")
    
    return execution_times


def visualize_anomalies(data, anomaly_results, execution_times, window_size, overlap_type, output_dir):
    """
    Create a comprehensive visualization of anomalies with offset markers and connecting lines.
    
    Args:
        data (pandas.DataFrame): Original time series data
        anomaly_results (dict): Dictionary of anomaly results for each algorithm
        execution_times (dict): Dictionary of execution times
        window_size (int): Window size used
        overlap_type (str): 'overlap' or 'nonoverlap'
        output_dir (Path): Directory to save visualizations
    """
    # Ensure output directory exists
    ensure_directory_exists(output_dir)
    
    # Create a figure with two subplots
    plt.figure(figsize=(16, 10))
    
    # Plot 1: Time series with anomalies
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['Close'], label='S&P 500 Close Price', color='blue', alpha=0.7)
    plt.title(f'S&P 500 Subsequence Anomalies (Window Size: {window_size}, {overlap_type})')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    # Markers and colors for each algorithm
    markers = {
        'aida': ('o', 'red', 'AIDA', -1),
        'iforest': ('^', 'green', 'Isolation Forest', 0),
        'lof': ('s', 'purple', 'LOF', 1)
    }
    
    # Calculate vertical offset
    price_range = data['Close'].max() - data['Close'].min()
    vertical_offset = price_range * 0.05  # 5% of price range
    
    # Detailed anomaly statistics
    anomaly_stats = {}
    
    # Collect anomalies for each algorithm
    for algo, (marker, color, label, position) in markers.items():
        if algo not in anomaly_results or anomaly_results[algo].empty:
            continue
            
        anomaly_df = anomaly_results[algo]
        
        # Ensure we have start_date column
        if 'start_date' not in anomaly_df.columns:
            logger.warning(f"No 'start_date' column in {algo} anomalies, skipping visualization")
            continue
        
        # Find the actual close prices for these dates
        date_values = []
        prices = []
        
        for _, row in anomaly_df.iterrows():
            date = row['start_date']
            
            # Find closest date in data
            if date in data.index:
                matched_date = date
            else:
                # Find closest date (for cases where anomaly date is not exactly in data)
                closest_idx = (data.index - date).abs().argmin()
                matched_date = data.index[closest_idx]
            
            date_values.append(matched_date)
            prices.append(data.loc[matched_date, 'Close'])
        
        # Calculate statistics for this algorithm's anomaly scores
        all_scores = anomaly_df['score']
        mean_score = all_scores.mean()
        std_score = all_scores.std()
        
        # Store detailed statistics
        anomaly_stats[algo] = {
            'count': len(date_values),
            'mean_score': mean_score,
            'std_score': std_score,
            'min_score': all_scores.min(),
            'max_score': all_scores.max()
        }
        
        # Calculate vertical position
        offset_prices = [price + (position * vertical_offset) for price in prices]
        
        # Plot connecting lines
        for date, orig_price, offset_price in zip(date_values, prices, offset_prices):
            plt.plot([date, date], [orig_price, offset_price], 
                     color=color, linestyle=':', linewidth=1, alpha=0.5)
        
        # Plot offset markers
        plt.scatter(
            date_values, 
            offset_prices, 
            marker=marker, 
            color=color, 
            label=f"{label} ({len(date_values)} anomalies)", 
            s=100, 
            alpha=0.7
        )
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Execution Times
    plt.subplot(2, 1, 2)
    plt.title(f'Algorithm Execution Times (Window Size: {window_size}, {overlap_type})')
    
    if execution_times:
        colors = {'aida': 'red', 'iforest': 'green', 'lof': 'purple'}
        bars = plt.bar(
            list(execution_times.keys()), 
            list(execution_times.values()), 
            color=[colors.get(algo, 'gray') for algo in execution_times.keys()]
        )
        plt.xlabel('Algorithm')
        plt.ylabel('Execution Time (seconds)')
        
        # Add exact times as text on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.2f}s',
                     ha='center', va='bottom')
    else:
        plt.text(0.5, 0.5, 'No execution time data available', 
                 ha='center', va='center', transform=plt.gca().transAxes,
                 fontsize=14, color='gray')
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    output_file = output_dir / f"subsequence_anomalies_w{window_size}_{overlap_type}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comprehensive anomaly visualization to {output_file}")
    
    # Create a more detailed visualization for each algorithm
    for algo in anomaly_results:
        if anomaly_results[algo].empty:
            continue
            
        try:
            plt.figure(figsize=(16, 8))
            
            # Plot the whole time series
            plt.plot(data.index, data['Close'], label='S&P 500 Close Price', color='blue', alpha=0.5)
            
            # Get anomaly dates
            anomaly_df = anomaly_results[algo]
            start_dates = pd.to_datetime(anomaly_df['start_date'])
            
            if 'end_date' in anomaly_df.columns:
                end_dates = pd.to_datetime(anomaly_df['end_date'])
            else:
                # If no end date, use start date + window size
                end_dates = [date + pd.Timedelta(days=window_size) for date in start_dates]
            
            # Highlight anomaly windows
            for i, (start, end) in enumerate(zip(start_dates, end_dates)):
                # Find closest dates in the data
                if start not in data.index:
                    closest_start_idx = (data.index - start).abs().argmin()
                    start = data.index[closest_start_idx]
                
                if end not in data.index:
                    closest_end_idx = (data.index - end).abs().argmin()
                    end = data.index[closest_end_idx]
                
                # Get price range for this window
                window_data = data.loc[start:end]
                if not window_data.empty:
                    plt.axvspan(start, end, alpha=0.2, color='red')
                    
                    # Annotate with anomaly score
                    score = anomaly_df.iloc[i]['score']
                    mid_point = start + (end - start) / 2
                    y_pos = window_data['Close'].max() * 1.02
                    plt.annotate(f"Score: {score:.2f}", (mid_point, y_pos), 
                                ha='center', fontsize=8, color='red')
            
            plt.title(f'{algo.upper()} Anomalies (Window Size: {window_size}, {overlap_type})')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the detailed plot
            detailed_file = output_dir / f"{algo}_anomalies_w{window_size}_{overlap_type}.png"
            plt.savefig(detailed_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved detailed {algo} visualization to {detailed_file}")
        except Exception as e:
            logger.error(f"Error creating detailed visualization for {algo}: {e}")
    
    # Create a detailed time series visualization showing specific anomaly windows
    try:
        # Collect all anomaly dates from all algorithms
        all_anomaly_dates = []
        for algo in anomaly_results:
            if 'start_date' in anomaly_results[algo].columns:
                dates = pd.to_datetime(anomaly_results[algo]['start_date'])
                all_anomaly_dates.extend(dates)
        
        if all_anomaly_dates:
            # Sort and get unique dates
            all_anomaly_dates = sorted(set(all_anomaly_dates))
            
            # Create multiple plots of anomaly windows
            for i, anomaly_date in enumerate(all_anomaly_dates[:min(10, len(all_anomaly_dates))]):
                # Get a window around the anomaly date (±30 days)
                start_date = anomaly_date - pd.Timedelta(days=15)
                end_date = anomaly_date + pd.Timedelta(days=15)
                
                # Get data for this window
                window_data = data.loc[start_date:end_date]
                if window_data.empty:
                    continue
                
                # Plot the anomaly window
                plt.figure(figsize=(12, 6))
                plt.plot(window_data.index, window_data['Close'], label='S&P 500 Close', marker='o')
                
                # Highlight the anomaly date
                anomaly_price = window_data.loc[window_data.index[window_data.index.get_indexer([anomaly_date], method='nearest')[0]], 'Close']
                plt.scatter([anomaly_date], [anomaly_price], color='red', s=100, zorder=5, label='Anomaly')
                
                # Add algorithm information
                algo_info = []
                for algo, results in anomaly_results.items():
                    if 'start_date' in results.columns:
                        if any(pd.to_datetime(results['start_date']).isin([anomaly_date])):
                            scores = results.loc[pd.to_datetime(results['start_date']) == anomaly_date, 'score']
                            algo_info.append(f"{algo.upper()}: Score={scores.iloc[0]:.2f}")
                
                if algo_info:
                    plt.annotate('\n'.join(algo_info), (0.02, 0.02), xycoords='axes fraction',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
                
                plt.title(f'Anomaly Window: {anomaly_date.strftime("%Y-%m-%d")}')
                plt.xlabel('Date')
                plt.ylabel('Close Price')
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.legend()
                plt.tight_layout()
                
                # Save the window plot
                window_file = output_dir / f"anomaly_window_{i+1}_w{window_size}_{overlap_type}.png"
                plt.savefig(window_file, dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info(f"Saved anomaly window visualizations to {output_dir}")
    except Exception as e:
        logger.error(f"Error creating anomaly window visualizations: {e}")
    
    # Detailed summary file
    summary_file = output_dir / f"anomaly_detection_summary_w{window_size}_{overlap_type}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Subsequence Anomaly Detection Summary\n")
        f.write(f"Window Size: {window_size}, Type: {overlap_type}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Execution Times:\n")
        for algo, time_val in execution_times.items():
            f.write(f"{algo.upper()}: {time_val:.4f} seconds\n")
        
        f.write("\nAnomaly Counts:\n")
        for algo, anomalies in anomaly_results.items():
            f.write(f"{algo.upper()}: {len(anomalies)} anomalies\n")
        
        f.write("\nDetailed Anomaly Statistics:\n")
        for algo, stats in anomaly_stats.items():
            f.write(f"\n{algo.upper()}:\n")
            for key, value in stats.items():
                f.write(f"  {key.replace('_', ' ').title()}: {value}\n")
    
    logger.info(f"Saved anomaly detection summary to {summary_file}")


def main():
    """
    Main function to compare and visualize subsequence anomaly detection results.
    """
    parser = argparse.ArgumentParser(description="Compare subsequence anomaly detection results")
    parser.add_argument(
        "--results-base", 
        type=str, 
        default=str(config.DATA_DIR / "subsequence_results"),
        help="Base directory containing all subsequence results"
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
    parser.add_argument(
        "--window-size",
        type=int,
        default=3,
        help="Window size to analyze"
    )
    parser.add_argument(
        "--overlap-type",
        choices=["overlap", "nonoverlap"],
        default="overlap",
        help="Overlap type to analyze"
    )
    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Analyze all available window sizes and overlap types"
    )
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    results_base_dir = Path(args.results_base)
    data_file = Path(args.data)
    output_dir = Path(args.output)
    
    # Ensure output directory exists
    ensure_directory_exists(output_dir)
    
    # Load original data
    data = load_original_data(data_file)
    
    if data is None or data.empty:
        logger.error("Failed to load original data. Exiting.")
        return
    
    # Determine configurations to analyze
    configs_to_analyze = []
    
    if args.all_configs:
        # Find all available configurations
        for directory in results_base_dir.iterdir():
            if directory.is_dir() and directory.name.startswith('w'):
                try:
                    # Parse window size and overlap type from directory name
                    # Format expected: w{size}_{overlap_type}
                    parts = directory.name.split('_')
                    if len(parts) == 2:
                        window_size = int(parts[0][1:])  # Remove 'w' prefix
                        overlap_type = parts[1]
                        configs_to_analyze.append((window_size, overlap_type))
                except ValueError:
                    logger.warning(f"Couldn't parse configuration from directory name: {directory.name}")
        
        logger.info(f"Found {len(configs_to_analyze)} configurations to analyze")
    else:
        # Use the single configuration specified in arguments
        configs_to_analyze = [(args.window_size, args.overlap_type)]
    
    # Process each configuration
    for window_size, overlap_type in configs_to_analyze:
        logger.info(f"Analyzing configuration: Window Size = {window_size}, Overlap Type = {overlap_type}")
        
        # Create specific output directory for this configuration
        config_output_dir = output_dir / f"w{window_size}_{overlap_type}"
        ensure_directory_exists(config_output_dir)
        
        # Load anomaly results
        anomaly_results = load_anomaly_results(results_base_dir, window_size, overlap_type)
        
        if not anomaly_results:
            logger.warning(f"No anomaly results found for w{window_size}_{overlap_type}")
            continue
        
        # Load execution times
        execution_times = load_execution_times(results_base_dir, window_size, overlap_type)
        
        # Visualize and compare results
        visualize_anomalies(data, anomaly_results, execution_times, window_size, overlap_type, config_output_dir)
        
        logger.info(f"Completed analysis for w{window_size}_{overlap_type}")
    
    # If we analyzed multiple configurations, create a comparative summary
    if len(configs_to_analyze) > 1:
        try:
            # Collect summary statistics for each configuration
            summary_data = []
            
            for window_size, overlap_type in configs_to_analyze:
                config_summary = {}
                config_summary['window_size'] = window_size
                config_summary['overlap_type'] = overlap_type
                
                # Load anomaly results to get counts
                anomaly_results = load_anomaly_results(results_base_dir, window_size, overlap_type)
                for algo in ['aida', 'iforest', 'lof']:
                    config_summary[f'{algo}_count'] = len(anomaly_results.get(algo, pd.DataFrame()))
                
                # Load execution times
                execution_times = load_execution_times(results_base_dir, window_size, overlap_type)
                for algo in ['aida', 'iforest', 'lof']:
                    config_summary[f'{algo}_time'] = execution_times.get(algo, 0)
                
                summary_data.append(config_summary)
            
            # Create summary DataFrame
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                
                # Save to CSV
                summary_csv = output_dir / "configuration_comparison.csv"
                summary_df.to_csv(summary_csv, index=False)
                logger.info(f"Saved configuration comparison to {summary_csv}")
                
                # Create comparative visualizations
                plt.figure(figsize=(14, 10))
                
                # Anomaly Count Comparison
                plt.subplot(2, 1, 1)
                plt.title("Anomaly Count Comparison by Configuration")
                
                # Prepare data for grouped bar chart
                algo_counts = []
                
                for index, row in summary_df.iterrows():
                    window_size = row['window_size']
                    overlap_type = row['overlap_type']
                    config_label = f"w{window_size}_{overlap_type}"
                    
                    for algo in ['aida', 'iforest', 'lof']:
                        algo_counts.append({
                            'Configuration': config_label,
                            'Algorithm': algo.upper(),
                            'Count': row[f'{algo}_count']
                        })
                
                counts_df = pd.DataFrame(algo_counts)
                sns.barplot(x='Configuration', y='Count', hue='Algorithm', data=counts_df)
                plt.grid(True, alpha=0.3)
                
                # Execution Time Comparison
                plt.subplot(2, 1, 2)
                plt.title("Execution Time Comparison by Configuration")
                
                # Prepare data for grouped bar chart
                algo_times = []
                
                for index, row in summary_df.iterrows():
                    window_size = row['window_size']
                    overlap_type = row['overlap_type']
                    config_label = f"w{window_size}_{overlap_type}"
                    
                    for algo in ['aida', 'iforest', 'lof']:
                        algo_times.append({
                            'Configuration': config_label,
                            'Algorithm': algo.upper(),
                            'Time (s)': row[f'{algo}_time']
                        })
                
                times_df = pd.DataFrame(algo_times)
                sns.barplot(x='Configuration', y='Time (s)', hue='Algorithm', data=times_df)
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_dir / "configuration_comparison.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Saved configuration comparison visualization to {output_dir / 'configuration_comparison.png'}")
                
                # Additional visualizations: scatter plots showing relationship between window size and anomaly count
                plt.figure(figsize=(10, 6))
                plt.title("Window Size vs. Anomaly Count by Algorithm")
                
                for algo in ['aida', 'iforest', 'lof']:
                    overlap_data = summary_df[summary_df['overlap_type'] == 'overlap']
                    plt.scatter(
                        overlap_data['window_size'], 
                        overlap_data[f'{algo}_count'],
                        label=f"{algo.upper()} (overlap)",
                        marker='o',
                        s=80,
                        alpha=0.7
                    )
                    
                    non_overlap_data = summary_df[summary_df['overlap_type'] == 'nonoverlap']
                    plt.scatter(
                        non_overlap_data['window_size'], 
                        non_overlap_data[f'{algo}_count'],
                        label=f"{algo.upper()} (non-overlap)",
                        marker='x',
                        s=80,
                        alpha=0.7
                    )
                
                plt.xlabel("Window Size")
                plt.ylabel("Anomaly Count")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_dir / "window_size_vs_anomaly_count.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Saved window size analysis to {output_dir / 'window_size_vs_anomaly_count.png'}")
        
        except Exception as e:
            logger.error(f"Error creating comparative summary: {e}")
    
    logger.info(f"All analyses completed and saved to {output_dir}")


if __name__ == "__main__":
    main()
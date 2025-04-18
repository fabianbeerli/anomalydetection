#!/usr/bin/env python
"""
Script to compare anomaly detection results between different subsequence configurations.
"""
import os
import sys
import logging
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import date2num, DateFormatter

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


def load_configuration_results(base_dir, configs):
    """
    Load results for different subsequence configurations.
    
    Args:
        base_dir (Path): Base directory containing results
        configs (list): List of configuration strings (e.g., "w3_overlap", "w5_nonoverlap")
        
    Returns:
        dict: Dictionary of configuration results
    """
    config_results = {}
    
    for config in configs:
        config_dir = base_dir / config
        if not config_dir.exists():
            logger.warning(f"Configuration directory {config_dir} does not exist. Skipping.")
            continue
        
        # Load execution times
        execution_times_file = config_dir / "execution_times.json"
        execution_times = {}
        if execution_times_file.exists():
            try:
                with open(execution_times_file, 'r') as f:
                    execution_times = json.load(f)
            except Exception as e:
                logger.error(f"Error loading execution times for {config}: {e}")
        
        # Load algorithm results
        algo_results = {}
        for algo in ["aida", "iforest", "lof"]:
            algo_dir = base_dir / algo / config
            if not algo_dir.exists():
                logger.warning(f"Algorithm directory {algo_dir} does not exist. Skipping.")
                continue
            
            # Load anomalies
            anomalies_file = algo_dir / f"{algo}_anomalies.csv"
            if anomalies_file.exists():
                try:
                    anomalies = pd.read_csv(anomalies_file)
                    if 'start_date' in anomalies.columns:
                        anomalies['start_date'] = pd.to_datetime(anomalies['start_date'])
                    if 'end_date' in anomalies.columns:
                        anomalies['end_date'] = pd.to_datetime(anomalies['end_date'])
                    algo_results[algo] = anomalies
                except Exception as e:
                    logger.error(f"Error loading anomalies for {algo} in {config}: {e}")
        
        config_results[config] = {
            "execution_times": execution_times,
            "algorithm_results": algo_results
        }
    
    return config_results


def compare_execution_times(config_results, output_dir):
    """
    Compare execution times across configurations and algorithms.
    
    Args:
        config_results (dict): Dictionary of configuration results
        output_dir (Path): Directory to save comparison results
        
    Returns:
        Path: Path to the saved comparison plot
    """
    # Extract execution times
    configs = []
    algos = set()
    times = {}
    
    for config, results in config_results.items():
        configs.append(config)
        for algo, time_val in results["execution_times"].items():
            algos.add(algo)
            if algo not in times:
                times[algo] = []
            times[algo].append(time_val)
    
    # Create DataFrame for plotting
    df_times = pd.DataFrame({algo: times.get(algo, [np.nan] * len(configs)) for algo in algos}, index=configs)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    ax = df_times.plot(kind='bar')
    plt.title('Execution Times by Algorithm and Configuration')
    plt.xlabel('Configuration')
    plt.ylabel('Execution Time (seconds)')
    plt.legend(title='Algorithm')
    plt.grid(axis='y', alpha=0.3)
    
    # Add values on top of bars
    for i, container in enumerate(ax.containers):
        ax.bar_label(container, fmt='%.2f', padding=3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / "execution_time_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data
    csv_file = output_dir / "execution_time_comparison.csv"
    df_times.to_csv(csv_file)
    
    return plot_file


def compare_anomaly_counts(config_results, output_dir):
    """
    Compare anomaly counts across configurations and algorithms.
    
    Args:
        config_results (dict): Dictionary of configuration results
        output_dir (Path): Directory to save comparison results
        
    Returns:
        Path: Path to the saved comparison plot
    """
    # Extract anomaly counts
    configs = []
    algos = set()
    counts = {}
    
    for config, results in config_results.items():
        configs.append(config)
        for algo, anomalies in results["algorithm_results"].items():
            algos.add(algo)
            if algo not in counts:
                counts[algo] = []
            counts[algo].append(len(anomalies))
    
    # Create DataFrame for plotting
    df_counts = pd.DataFrame({algo: counts.get(algo, [0] * len(configs)) for algo in algos}, index=configs)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    ax = df_counts.plot(kind='bar')
    plt.title('Anomaly Counts by Algorithm and Configuration')
    plt.xlabel('Configuration')
    plt.ylabel('Number of Anomalies')
    plt.legend(title='Algorithm')
    plt.grid(axis='y', alpha=0.3)
    
    # Add values on top of bars
    for i, container in enumerate(ax.containers):
        ax.bar_label(container, fmt='%d', padding=3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / "anomaly_count_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save data
    csv_file = output_dir / "anomaly_count_comparison.csv"
    df_counts.to_csv(csv_file)
    
    return plot_file


def visualize_anomalies_by_configuration(config_results, data_df, output_dir):
    """
    Visualize anomalies across different configurations.
    
    Args:
        config_results (dict): Dictionary of configuration results
        data_df (pandas.DataFrame): Original time series data
        output_dir (Path): Directory to save visualizations
        
    Returns:
        dict: Dictionary mapping configurations to plot files
    """
    plot_files = {}
    
    # Define colors and markers for algorithms
    algo_styles = {
        'aida': {'color': 'red', 'marker': 'o', 'label': 'AIDA'},
        'iforest': {'color': 'green', 'marker': '^', 'label': 'Isolation Forest'},
        'lof': {'color': 'blue', 'marker': 's', 'label': 'LOF'}
    }
    
    # Create visualizations for each configuration
    for config, results in config_results.items():
        plt.figure(figsize=(16, 8))
        
        # Plot the original time series
        plt.plot(data_df.index, data_df['Close'], color='gray', alpha=0.5, label='S&P 500 Close')
        
        # Plot anomalies for each algorithm
        for algo, anomalies in results["algorithm_results"].items():
            style = algo_styles.get(algo, {'color': 'black', 'marker': 'x', 'label': algo})
            
            if 'start_date' in anomalies.columns:
                # Convert dates to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(anomalies['start_date']):
                    anomalies['start_date'] = pd.to_datetime(anomalies['start_date'])
                
                # Find corresponding close prices for these dates
                anomaly_dates = anomalies['start_date'].tolist()
                anomaly_prices = []
                
                for date in anomaly_dates:
                    # Find the closest date in the original data
                    closest_idx = data_df.index.get_indexer([date], method='nearest')[0]
                    if 0 <= closest_idx < len(data_df):
                        anomaly_prices.append(data_df['Close'].iloc[closest_idx])
                    else:
                        anomaly_prices.append(np.nan)
                
                plt.scatter(
                    anomaly_dates,
                    anomaly_prices,
                    color=style['color'],
                    marker=style['marker'],
                    s=100,
                    alpha=0.7,
                    label=f"{style['label']} ({len(anomalies)} anomalies)"
                )
        
        plt.title(f'Anomalies Detected in Configuration: {config}')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        # Format date axis
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        
        # Save plot
        plot_file = output_dir / f"anomalies_{config}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        plot_files[config] = plot_file
    
    return plot_files


def create_anomaly_timeline(config_results, output_dir):
    """
    Create a timeline of anomalies across configurations.
    
    Args:
        config_results (dict): Dictionary of configuration results
        output_dir (Path): Directory to save visualizations
        
    Returns:
        Path: Path to the saved timeline
    """
    # Collect all anomaly dates across configurations
    all_dates = []
    all_configs = []
    all_algos = []
    
    for config, results in config_results.items():
        for algo, anomalies in results["algorithm_results"].items():
            if 'start_date' in anomalies.columns:
                dates = pd.to_datetime(anomalies['start_date']).tolist()
                all_dates.extend(dates)
                all_configs.extend([config] * len(dates))
                all_algos.extend([algo] * len(dates))
    
    if not all_dates:
        logger.warning("No anomaly dates found across configurations")
        return None
    
    # Create DataFrame
    timeline_df = pd.DataFrame({
        'date': all_dates,
        'config': all_configs,
        'algorithm': all_algos
    })
    
    # Sort by date
    timeline_df = timeline_df.sort_values('date')
    
    # Create a figure with shared x-axis
    fig, axes = plt.subplots(len(config_results), 1, figsize=(16, 10), sharex=True)
    
    # If only one configuration, put axes in a list for consistent indexing
    if len(config_results) == 1:
        axes = [axes]
    
    # Define colors for algorithms
    algo_colors = {
        'aida': 'red',
        'iforest': 'green',
        'lof': 'blue'
    }
    
    # Plot anomalies for each configuration
    for i, config in enumerate(sorted(config_results.keys())):
        ax = axes[i]
        
        # Filter data for this configuration
        config_data = timeline_df[timeline_df['config'] == config]
        
        # Create scatter plot grouped by algorithm
        for algo in sorted(config_data['algorithm'].unique()):
            algo_data = config_data[config_data['algorithm'] == algo]
            
            # Convert dates to numbers for plotting
            dates_as_num = date2num(algo_data['date'].tolist())
            
            ax.scatter(
                dates_as_num,
                [0.5] * len(dates_as_num),  # All points at same height
                s=100,
                color=algo_colors.get(algo, 'black'),
                marker='o',
                alpha=0.7,
                label=f"{algo.upper()} ({len(algo_data)})"
            )
        
        ax.set_title(f'Configuration: {config}')
        ax.set_yticks([])  # No y-ticks needed
        ax.grid(axis='x', alpha=0.3)
        ax.legend(loc='upper right')
        
        # Add a line at the bottom
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Format the date axis
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    plt.xlabel('Date')
    plt.tight_layout()
    
    # Save the plot
    timeline_file = output_dir / "anomaly_timeline.png"
    plt.savefig(timeline_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save timeline data
    csv_file = output_dir / "anomaly_timeline.csv"
    timeline_df.to_csv(csv_file, index=False)
    
    return timeline_file


def analyze_anomaly_overlap(config_results, output_dir):
    """
    Analyze the overlap of anomalies across configurations.
    
    Args:
        config_results (dict): Dictionary of configuration results
        output_dir (Path): Directory to save visualizations
        
    Returns:
        Path: Path to the saved overlap analysis
    """
    # Extract anomalies by date for each configuration and algorithm
    anomaly_dates = {}
    
    for config, results in config_results.items():
        anomaly_dates[config] = {}
        for algo, anomalies in results["algorithm_results"].items():
            if 'start_date' in anomalies.columns:
                dates = set(pd.to_datetime(anomalies['start_date']).dt.date)
                anomaly_dates[config][algo] = dates
    
    # Create overlap analysis
    overlap_data = []
    
    # Compare configurations for each algorithm
    for algo in ['aida', 'iforest', 'lof']:
        config_list = sorted([c for c in anomaly_dates if algo in anomaly_dates[c]])
        
        for i, config1 in enumerate(config_list):
            for j, config2 in enumerate(config_list[i+1:], i+1):
                dates1 = anomaly_dates[config1].get(algo, set())
                dates2 = anomaly_dates[config2].get(algo, set())
                
                if dates1 and dates2:
                    intersection = dates1.intersection(dates2)
                    union = dates1.union(dates2)
                    jaccard = len(intersection) / len(union) if union else 0
                    
                    overlap_data.append({
                        'algorithm': algo.upper(),
                        'config1': config1,
                        'config2': config2,
                        'anomalies1': len(dates1),
                        'anomalies2': len(dates2),
                        'overlap': len(intersection),
                        'jaccard': jaccard
                    })
    
    if not overlap_data:
        logger.warning("No overlap data to analyze")
        return None
    
    # Create DataFrame
    overlap_df = pd.DataFrame(overlap_data)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=overlap_df, x='algorithm', y='jaccard', hue='config1')
    plt.title('Anomaly Overlap Between Configurations (Jaccard Index)')
    plt.xlabel('Algorithm')
    plt.ylabel('Jaccard Index')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / "anomaly_overlap.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save overlap data
    csv_file = output_dir / "anomaly_overlap.csv"
    overlap_df.to_csv(csv_file, index=False)
    
    return plot_file


def main():
    """
    Main function to compare anomaly detection results.
    """
    parser = argparse.ArgumentParser(description="Compare subsequence anomaly detection results")
    parser.add_argument(
        "--results-dir", 
        type=str, 
        default=str(config.DATA_DIR / "subsequence_results"),
        help="Directory containing algorithm results"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["w3_overlap", "w3_nonoverlap", "w5_overlap", "w5_nonoverlap"],
        help="Configurations to compare (e.g., w3_overlap, w5_nonoverlap)"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default=str(config.PROCESSED_DATA_DIR / "index_GSPC_processed.csv"),
        help="Path to the original S&P 500 data"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=str(config.DATA_DIR / "subsequence_comparison"),
        help="Directory to save comparison results"
    )
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    results_dir = Path(args.results_dir)
    data_file = Path(args.data)
    output_dir = Path(args.output)
    
    # Ensure output directory exists
    ensure_directory_exists(output_dir)
    
    # Load original data
    logger.info(f"Loading S&P 500 data from {data_file}")
    data_df = load_ticker_data(data_file)
    
    if data_df is None or data_df.empty:
        logger.error("Failed to load S&P 500 data")
        return
    
    # Load results for configurations
    logger.info(f"Loading results for configurations: {', '.join(args.configs)}")
    config_results = load_configuration_results(results_dir, args.configs)
    
    if not config_results:
        logger.error("No configuration results loaded")
        return
    
    # Create comparison reports
    logger.info("Creating comparison reports")
    
    # Compare execution times
    logger.info("Comparing execution times")
    execution_time_plot = compare_execution_times(config_results, output_dir)
    logger.info(f"Execution time comparison saved to {execution_time_plot}")
    
    # Compare anomaly counts
    logger.info("Comparing anomaly counts")
    anomaly_count_plot = compare_anomaly_counts(config_results, output_dir)
    logger.info(f"Anomaly count comparison saved to {anomaly_count_plot}")
    
    # Visualize anomalies by configuration
    logger.info("Visualizing anomalies by configuration")
    anomaly_plots = visualize_anomalies_by_configuration(config_results, data_df, output_dir)
    logger.info(f"Created {len(anomaly_plots)} anomaly visualizations")
    
    # Create anomaly timeline
    logger.info("Creating anomaly timeline")
    timeline_plot = create_anomaly_timeline(config_results, output_dir)
    if timeline_plot:
        logger.info(f"Anomaly timeline saved to {timeline_plot}")
    
    # Analyze anomaly overlap
    logger.info("Analyzing anomaly overlap")
    overlap_plot = analyze_anomaly_overlap(config_results, output_dir)
    if overlap_plot:
        logger.info(f"Anomaly overlap analysis saved to {overlap_plot}")
    
    logger.info(f"All comparison results saved to {output_dir}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Script to visualize results from matrix-based constituent analysis.
"""
import os
import sys
import json
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.utils.helpers import ensure_directory_exists
from src.data.preparation import load_ticker_data

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def visualize_classification_distribution(summary, output_dir):
    """Create a bar chart showing the distribution of anomaly classifications."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Isolation Forest classifications
    iforest_counts = summary['iforest_classifications']
    categories = list(iforest_counts.keys())
    values = list(iforest_counts.values())
    
    bars1 = ax1.bar(categories, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_title('Isolation Forest Anomaly Classifications')
    ax1.set_ylabel('Count')
    ax1.set_xlabel('Classification Type')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    # LOF classifications
    lof_counts = summary['lof_classifications']
    values = [lof_counts[cat] for cat in categories]
    
    bars2 = ax2.bar(categories, values, color=['#d62728', '#9467bd', '#8c564b'])
    ax2.set_title('LOF Anomaly Classifications')
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Classification Type')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'classification_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_algorithm_agreement(results, output_dir):
    """Create a visualization showing agreement between algorithms."""
    dates = [r['anomaly_date'] for r in results]
    iforest_anomalies = [r['iforest_is_anomaly'] for r in results]
    lof_anomalies = [r['lof_is_anomaly'] for r in results]
    
    # Convert to datetime for better plotting
    dates_dt = [pd.to_datetime(date) for date in dates]
    
    plt.figure(figsize=(14, 6))
    
    # Plot anomalies for both algorithms
    plt.scatter(dates_dt, [0] * len(dates), s=100, 
               c=iforest_anomalies, cmap='RdYlGn_r', marker='o', 
               label='Isolation Forest')
    plt.scatter(dates_dt, [1] * len(dates), s=100, 
               c=lof_anomalies, cmap='RdYlGn_r', marker='s', 
               label='LOF')
    
    # Highlight agreements/disagreements
    agreements = [iforest_anomalies[i] == lof_anomalies[i] for i in range(len(results))]
    for i, (date, agree) in enumerate(zip(dates_dt, agreements)):
        if not agree:
            plt.axvline(x=date, color='red', alpha=0.3, linestyle='--')
    
    plt.yticks([0, 1], ['Isolation Forest', 'LOF'])
    plt.xticks(rotation=45)
    plt.title('Algorithm Agreement on Constituent Anomalies')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'algorithm_agreement.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_anomaly_scores(results, output_dir):
    """Create a scatter plot of anomaly scores from both algorithms."""
    iforest_scores = [r['iforest_score'] for r in results]
    lof_scores = [r['lof_score'] for r in results]
    dates = [r['anomaly_date'] for r in results]
    
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    scatter = plt.scatter(iforest_scores, lof_scores, 
                         c=range(len(results)), cmap='viridis', 
                         s=100, alpha=0.7)
    
    # Add diagonal line
    min_val = min(min(iforest_scores), min(lof_scores))
    max_val = max(max(iforest_scores), max(lof_scores))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Add annotations for points
    for i, date in enumerate(dates):
        plt.annotate(date, (iforest_scores[i], lof_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)
    
    plt.xlabel('Isolation Forest Score')
    plt.ylabel('LOF Score')
    plt.title('Comparison of Anomaly Scores')
    plt.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Chronological Order')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'anomaly_scores_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_temporal_pattern(results, output_dir):
    """Create a temporal visualization of anomalies and classifications."""
    # Extract data
    dates = [pd.to_datetime(r['anomaly_date']) for r in results]
    classifications_if = [r['classification_iforest'] for r in results]
    classifications_lof = [r['classification_lof'] for r in results]
    n_stocks = [r['n_stocks'] for r in results]
    
    # Create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), height_ratios=[2, 2, 1])
    
    # Plot Isolation Forest classifications over time
    colors_if = {'widespread': 'red', 'isolated': 'blue', 'moderate': 'orange'}
    for i, (date, classification) in enumerate(zip(dates, classifications_if)):
        ax1.scatter(date, 1, c=colors_if[classification], s=100, 
                   marker='o', edgecolors='black')
    
    ax1.set_yticks([])
    ax1.set_title('Isolation Forest Classifications Over Time')
    
    # Create legend for classifications
    handles = [plt.Line2D([0], [0], marker='o', color='w', 
                         markerfacecolor=color, markersize=10, 
                         label=classification.capitalize())
              for classification, color in colors_if.items()]
    ax1.legend(handles=handles, loc='upper right')
    
    # Plot LOF classifications over time
    colors_lof = {'widespread': 'red', 'isolated': 'blue', 'moderate': 'orange'}
    for i, (date, classification) in enumerate(zip(dates, classifications_lof)):
        ax2.scatter(date, 1, c=colors_lof[classification], s=100, 
                   marker='s', edgecolors='black')
    
    ax2.set_yticks([])
    ax2.set_title('LOF Classifications Over Time')
    ax2.legend(handles=handles, loc='upper right')
    
    # Plot number of stocks analyzed
    ax3.plot(dates, n_stocks, 'g-', marker='o')
    ax3.set_ylabel('Number of Stocks')
    ax3.set_title('Number of Stocks Analyzed per Anomaly')
    ax3.grid(True, alpha=0.3)
    
    # Format x-axis
    for ax in [ax1, ax2, ax3]:
        ax.grid(True, axis='x', alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_pattern.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_detailed_summary(results, summary, output_dir):
    """Create a detailed text summary of the analysis."""
    with open(output_dir / 'detailed_summary.txt', 'w') as f:
        f.write("Constituent Anomaly Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total index anomalies analyzed: {summary['analyzed_anomalies']}\n")
        f.write(f"Average number of stocks analyzed per anomaly: {summary['average_stocks_analyzed']:.1f}\n\n")
        
        f.write("Classification Results:\n")
        f.write("-" * 30 + "\n")
        f.write("Isolation Forest:\n")
        for classification, count in summary['iforest_classifications'].items():
            f.write(f"  {classification.capitalize()}: {count} ({count/summary['analyzed_anomalies']*100:.1f}%)\n")
        
        f.write("\nLOF:\n")
        for classification, count in summary['lof_classifications'].items():
            f.write(f"  {classification.capitalize()}: {count} ({count/summary['analyzed_anomalies']*100:.1f}%)\n")
        
        f.write(f"\nAlgorithm Agreement: {summary['anomaly_agreement']} out of {summary['analyzed_anomalies']} "
                f"({summary['anomaly_agreement']/summary['analyzed_anomalies']*100:.1f}%)\n\n")
        
        f.write("Individual Anomaly Details:\n")
        f.write("-" * 30 + "\n")
        for result in results:
            f.write(f"\nDate: {result['anomaly_date']}\n")
            f.write(f"Stocks analyzed: {result['n_stocks']}\n")
            f.write(f"Isolation Forest: {result['classification_iforest']} "
                    f"(Anomaly: {'Yes' if result['iforest_is_anomaly'] else 'No'}, "
                    f"Score: {result['iforest_score']:.3f})\n")
            f.write(f"LOF: {result['classification_lof']} "
                    f"(Anomaly: {'Yes' if result['lof_is_anomaly'] else 'No'}, "
                    f"Score: {result['lof_score']:.3f})\n")
            f.write(f"Number of windows analyzed: {result['n_windows']}\n")
            f.write(f"Anomalous windows - IF: {result['n_iforest_anomalies']}, LOF: {result['n_lof_anomalies']}\n")
            if 'stocks' in result:
                f.write(f"Stocks: {', '.join(result['stocks'])}\n")


def create_stock_frequency_analysis(results, output_dir):
    """Analyze which stocks are most frequently involved in anomalies."""
    if not results or 'stocks' not in results[0]:
        logger.warning("No stock information available for frequency analysis")
        return
    
    # Count stock occurrences
    stock_counts = {}
    for result in results:
        for stock in result['stocks']:
            stock_counts[stock] = stock_counts.get(stock, 0) + 1
    
    # Sort by frequency
    sorted_stocks = sorted(stock_counts.items(), key=lambda x: x[1], reverse=True)
    stocks, counts = zip(*sorted_stocks)
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(stocks, counts)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Stock Ticker')
    plt.ylabel('Number of Anomalies')
    plt.title('Stock Frequency in Anomalous Events')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'stock_frequency_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_classification_heatmap(results, output_dir):
    """Create a heatmap showing classification patterns over time."""
    # Extract data
    dates = [r['anomaly_date'] for r in results]
    classifications_if = [r['classification_iforest'] for r in results]
    classifications_lof = [r['classification_lof'] for r in results]
    
    # Create classification mapping
    class_map = {'isolated': 0, 'moderate': 1, 'widespread': 2}
    
    # Convert to numeric
    data_if = [class_map[c] for c in classifications_if]
    data_lof = [class_map[c] for c in classifications_lof]
    
    # Create matrix for heatmap
    matrix = np.array([data_if, data_lof])
    
    # Create heatmap
    plt.figure(figsize=(14, 6))
    sns.heatmap(matrix, annot=False, cmap='RdYlGn_r', 
                yticklabels=['Isolation Forest', 'LOF'],
                xticklabels=dates, cbar=False)
    
    # Add custom colorbar
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    
    colors = ['#2ca02c', '#ff7f0e', '#d62728']  # green, orange, red
    labels = ['Isolated', 'Moderate', 'Widespread']
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
    plt.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.xticks(rotation=45, ha='right')
    plt.title('Classification Patterns Over Time')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'classification_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_results(results_file, summary_file, output_dir):
    """Main function to create all visualizations."""
    ensure_directory_exists(output_dir)
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Create visualizations
    logger.info("Creating classification distribution chart")
    visualize_classification_distribution(summary, output_dir)
    
    logger.info("Creating algorithm agreement visualization")
    visualize_algorithm_agreement(results, output_dir)
    
    logger.info("Creating anomaly scores comparison")
    visualize_anomaly_scores(results, output_dir)
    
    logger.info("Creating temporal pattern visualization")
    visualize_temporal_pattern(results, output_dir)
    
    logger.info("Creating detailed summary")
    create_detailed_summary(results, summary, output_dir)
    
    logger.info("Creating stock frequency analysis")
    create_stock_frequency_analysis(results, output_dir)
    
    logger.info("Creating classification heatmap")
    create_classification_heatmap(results, output_dir)
    
    logger.info(f"All visualizations saved to {output_dir}")


def main():
    """Main function to visualize constituent analysis results."""
    parser = argparse.ArgumentParser(description="Visualize constituent analysis results")
    parser.add_argument(
        "--results", 
        type=str, 
        default=str(config.DATA_DIR / "constituent_analysis" / "constituent_analysis_results.json"),
        help="Path to results JSON file"
    )
    parser.add_argument(
        "--summary", 
        type=str, 
        default=str(config.DATA_DIR / "constituent_analysis" / "constituent_analysis_summary.json"),
        help="Path to summary JSON file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=str(config.DATA_DIR / "constituent_analysis" / "visualizations"),
        help="Directory to save visualizations"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    results_file = Path(args.results)
    summary_file = Path(args.summary)
    output_dir = Path(args.output)
    
    # Create visualizations
    visualize_results(results_file, summary_file, output_dir)
    
    logger.info("Visualization completed")


if __name__ == "__main__":
    main()
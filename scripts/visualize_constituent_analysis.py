#!/usr/bin/env python
"""
Script to visualize constituent analysis results.
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
from datetime import datetime
from matplotlib.dates import DateFormatter

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.utils.helpers import ensure_directory_exists

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_summary_visualizations(summary_data, output_dir):
    """
    Create summary visualizations from the analysis summary.
    
    Args:
        summary_data (dict): Summary data from analysis
        output_dir (Path): Directory to save visualizations
    """
    try:
        # Create a summary figure
        plt.figure(figsize=(10, 8))
        
        # Plot total anomalies by algorithm
        labels = ['Isolation Forest', 'LOF']
        anomaly_counts = [
            summary_data.get('total_iforest_constituent_anomalies', 0),
            summary_data.get('total_lof_constituent_anomalies', 0)
        ]
        
        plt.bar(labels, anomaly_counts, color=['green', 'purple'])
        plt.title("Total Constituent Anomalies Detected by Algorithm")
        plt.ylabel("Number of Anomalies")
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add counts as text
        for i, count in enumerate(anomaly_counts):
            plt.text(i, count + max(anomaly_counts) * 0.02, str(count), 
                    ha='center', va='bottom', fontsize=11)
        
        # Add summary text
        analysis_date = summary_data.get('analysis_date', 'Unknown')
        plt.figtext(0.5, 0.01, 
                   f"Analysis Date: {analysis_date}\nTotal Anomalies Analyzed: {summary_data.get('total_anomalies_analyzed', 0)}", 
                   ha='center', fontsize=10)
        
        # Save the figure
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to make room for the text
        plt.savefig(output_dir / "anomaly_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created summary visualization at {output_dir / 'anomaly_summary.png'}")
        
    except Exception as e:
        logger.error(f"Error creating summary visualizations: {e}")


def create_detailed_visualizations(results_data, output_dir):
    """
    Create detailed visualizations from the analysis results.
    
    Args:
        results_data (list): List of anomaly results
        output_dir (Path): Directory to save visualizations
    """
    try:
        # Convert to DataFrame for easier manipulation
        results_df = pd.DataFrame(results_data)
        
        if results_df.empty:
            logger.warning("No results data to visualize")
            return
        
        # Add date field for sorting
        try:
            results_df['date'] = pd.to_datetime(results_df['anomaly_date'], format='%Y%m%d')
            results_df = results_df.sort_values('date')
        except:
            logger.warning("Could not parse anomaly dates, using order in file")
        
        # 1. Timeline of Anomalies
        plt.figure(figsize=(14, 8))
        
        # Plot iForest and LOF anomaly counts by date/ID
        x = range(len(results_df))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], results_df['iforest_anomaly_count'], width, label='Isolation Forest', color='green', alpha=0.7)
        plt.bar([i + width/2 for i in x], results_df['lof_anomaly_count'], width, label='LOF', color='purple', alpha=0.7)
        
        # Use dates as labels if available, otherwise use IDs
        if 'date' in results_df.columns:
            plt.xticks(x, results_df['date'].dt.strftime('%Y-%m-%d'), rotation=45)
        else:
            plt.xticks(x, [f"Anomaly {id}" for id in results_df['anomaly_id']], rotation=45)
        
        plt.title("Constituent Anomalies Timeline")
        plt.ylabel("Number of Anomalies")
        plt.xlabel("Index Anomaly Date")
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / "anomalies_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Top Anomalous Stocks
        # Extract all mentioned stocks
        all_stocks = []
        
        for _, row in results_df.iterrows():
            if 'top_anomalous_stocks' in row and isinstance(row['top_anomalous_stocks'], list):
                stocks = [s.split(' ')[0] for s in row['top_anomalous_stocks']]
                all_stocks.extend(stocks)
        
        if all_stocks:
            # Count occurrences
            stock_counts = pd.Series(all_stocks).value_counts()
            
            # Plot top 10 stocks
            plt.figure(figsize=(12, 6))
            stock_counts.head(10).plot(kind='bar', color='darkblue')
            plt.title("Top 10 Anomalous Stocks")
            plt.ylabel("Frequency in Anomalous Windows")
            plt.xlabel("Stock")
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(output_dir / "top_anomalous_stocks.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Comparative Algorithm Performance
        plt.figure(figsize=(10, 6))
        
        # Create a scatter plot comparing iForest vs LOF anomaly counts
        plt.scatter(results_df['iforest_anomaly_count'], results_df['lof_anomaly_count'], 
                   alpha=0.7, s=80, c='darkblue')
        
        # Add a diagonal line representing equal performance
        max_val = max(results_df['iforest_anomaly_count'].max(), results_df['lof_anomaly_count'].max())
        plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Equal performance')
        
        # Add anomaly IDs as labels
        for idx, row in results_df.iterrows():
            plt.annotate(str(row['anomaly_id']), 
                        (row['iforest_anomaly_count'], row['lof_anomaly_count']),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title("Algorithm Comparison: iForest vs. LOF")
        plt.xlabel("Isolation Forest Anomaly Count")
        plt.ylabel("LOF Anomaly Count")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "algorithm_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Created detailed visualizations in {output_dir}")
        
    except Exception as e:
        logger.error(f"Error creating detailed visualizations: {e}")


def visualize_results(results_file, summary_file, output_dir):
    """
    Visualize constituent analysis results.
    
    Args:
        results_file (str): Path to results JSON file
        summary_file (str): Path to summary JSON file
        output_dir (str): Directory to save visualizations
    """
    try:
        # Ensure output directory exists
        output_path = Path(output_dir)
        ensure_directory_exists(output_path)
        
        # Load results data
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        # Load summary data
        with open(summary_file, 'r') as f:
            summary_data = json.load(f)
        
        # Create summary visualizations
        create_summary_visualizations(summary_data, output_path)
        
        # Create detailed visualizations
        create_detailed_visualizations(results_data, output_path)
        
        logger.info("Visualization of constituent analysis results completed")
        
    except Exception as e:
        logger.error(f"Error visualizing results: {e}")


def main():
    """
    Main function to visualize constituent analysis results.
    """
    parser = argparse.ArgumentParser(description="Visualize constituent analysis results")
    parser.add_argument(
        "--results", 
        type=str, 
        required=True,
        help="Path to the constituent analysis results JSON file"
    )
    parser.add_argument(
        "--summary", 
        type=str, 
        required=True,
        help="Path to the constituent analysis summary JSON file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=str(config.DATA_DIR / "constituent_analysis/visualizations"),
        help="Directory to save visualizations"
    )
    
    args = parser.parse_args()
    
    # Visualize results
    visualize_results(args.results, args.summary, args.output)


if __name__ == "__main__":
    main()
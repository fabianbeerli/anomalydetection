#!/usr/bin/env python
"""
Script to analyze the relationship between anomalies detected in the S&P 500 index
and its constituent stocks, including AIDA, Isolation Forest, and LOF algorithms.

This script:
1. Loads anomaly detection results for the S&P 500 index and constituent stocks
2. Analyzes which constituents show anomalies during index anomaly periods
3. Provides sector-specific and large-cap vs. small-cap analysis
4. Visualizes the findings with comprehensive plots and summaries
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
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import yfinance as yf  # For sector information

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


def load_index_anomalies(results_dir, algorithms=None):
    """
    Load anomaly detection results for the S&P 500 index.
    
    Args:
        results_dir (Path): Directory containing algorithm results for the index
        algorithms (list): List of algorithms to include (default: all)
        
    Returns:
        dict: Dictionary of anomaly DataFrames for each algorithm
    """
    if algorithms is None:
        algorithms = ['aida', 'iforest', 'lof']
        
    anomaly_results = {}
    
    for algo in algorithms:
        try:
            algo_dir = results_dir / algo / "w3_nonoverlap"
            
            if not algo_dir.exists() or not algo_dir.is_dir():
                logger.warning(f"No directory found for {algo} at {algo_dir}")
                continue
                
            # Find anomalies CSV file
            anomalies_files = list(algo_dir.glob(f"{algo}_anomalies.csv"))
            
            if not anomalies_files:
                logger.warning(f"No anomalies file found for {algo}")
                logger.warning(algo_dir)
                continue
                
            # Load the first anomalies file found
            anomalies_df = pd.read_csv(anomalies_files[0])
            
            # Convert date columns to datetime
            for date_col in ['start_date', 'end_date']:
                if date_col in anomalies_df.columns:
                    anomalies_df[date_col] = pd.to_datetime(anomalies_df[date_col])
                    
            anomaly_results[algo] = anomalies_df
            logger.info(f"Loaded {len(anomalies_df)} index anomalies for {algo}")
            
        except Exception as e:
            logger.error(f"Error loading {algo} index anomalies: {e}")
    
    return anomaly_results


def load_constituent_anomalies(constituents_dir, algorithms=None):
    """
    Load anomaly detection results for constituent stocks.
    
    Args:
        constituents_dir (Path): Directory containing constituent results
        algorithms (list): List of algorithms to include (default: all)
        
    Returns:
        dict: Dictionary of anomaly results by algorithm and ticker
    """
    if algorithms is None:
        algorithms = ['aida', 'iforest', 'lof']
        
    anomaly_results = {algo: {} for algo in algorithms}
    
    for algo in algorithms:
        algo_dir = constituents_dir / algo
        
        if not algo_dir.exists() or not algo_dir.is_dir():
            logger.warning(f"No directory found for {algo} at {algo_dir}")
            continue
            
        # Loop through ticker directories
        for ticker_dir in algo_dir.iterdir():
            if not ticker_dir.is_dir():
                continue
                
            ticker = ticker_dir.name
            
            # Find anomalies CSV file
            anomalies_files = list(ticker_dir.glob(f"{algo}_anomalies.csv"))
            
            if not anomalies_files:
                logger.warning(f"No anomalies file found for {ticker} using {algo}")
                continue
                
            # Load anomalies file
            try:
                anomalies_df = pd.read_csv(anomalies_files[0])
                
                # Convert date columns to datetime
                for date_col in ['start_date', 'end_date']:
                    if date_col in anomalies_df.columns:
                        anomalies_df[date_col] = pd.to_datetime(anomalies_df[date_col])
                        
                anomaly_results[algo][ticker] = anomalies_df
                
            except Exception as e:
                logger.error(f"Error loading {algo} anomalies for {ticker}: {e}")
    
    # Count loaded anomalies
    for algo in algorithms:
        total_constituents = len(anomaly_results[algo])
        total_anomalies = sum(len(df) for df in anomaly_results[algo].values())
        logger.info(f"Loaded {algo} anomalies for {total_constituents} constituents, total {total_anomalies} anomalies")
    
    return anomaly_results


def get_ticker_sector_info(tickers):
    """
    Get sector information for a list of tickers.
    
    Args:
        tickers (list): List of ticker symbols
        
    Returns:
        dict: Dictionary mapping tickers to their sectors
    """
    sector_info = {}
    
    for ticker in tickers:
        try:
            # Try to get ticker info from yfinance
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            if 'sector' in info:
                sector_info[ticker] = info['sector']
            else:
                logger.warning(f"No sector information found for {ticker}")
                sector_info[ticker] = "Unknown"
                
        except Exception as e:
            logger.error(f"Error getting sector info for {ticker}: {e}")
            sector_info[ticker] = "Unknown"
    
    # Print summary of sectors
    sector_counts = {}
    for sector in sector_info.values():
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        
    logger.info("Sector distribution:")
    for sector, count in sector_counts.items():
        logger.info(f"  {sector}: {count} companies")
        
    return sector_info


def analyze_index_constituent_relationship(index_anomalies, constituent_anomalies, sector_info, 
                                          window_days=3, output_dir=None, algorithms=None):
    """
    Analyze the relationship between index anomalies and constituent anomalies.
    
    Args:
        index_anomalies (dict): Dictionary of index anomaly DataFrames by algorithm
        constituent_anomalies (dict): Dictionary of constituent anomaly DataFrames by algorithm and ticker
        sector_info (dict): Dictionary mapping tickers to their sectors
        window_days (int): Number of days to look around each index anomaly
        output_dir (Path): Directory to save analysis results
        algorithms (list): List of algorithms to include (default: all available)
        
    Returns:
        dict: Dictionary containing analysis results
    """
    if algorithms is None:
        algorithms = list(index_anomalies.keys())
        
    # Create output directory
    if output_dir is not None:
        ensure_directory_exists(output_dir)
        
    # Initialize results
    analysis_results = {algo: [] for algo in algorithms}
    
    # Set of all tickers for which we have data
    all_tickers = set()
    for algo in algorithms:
        all_tickers.update(constituent_anomalies[algo].keys())
    all_tickers = sorted(all_tickers)
    
    # Process each algorithm
    for algo in algorithms:
        if algo not in index_anomalies or algo not in constituent_anomalies:
            logger.warning(f"Missing data for {algo}, skipping")
            continue
            
        index_df = index_anomalies[algo]
        constituent_data = constituent_anomalies[algo]
        
        # Process each index anomaly
        for idx, index_anomaly in index_df.iterrows():
            if 'start_date' not in index_anomaly:
                logger.warning(f"No start_date for index anomaly {idx}, skipping")
                continue
                
            anomaly_date = index_anomaly['start_date']
            anomaly_score = index_anomaly['score']
            
            # Define the window around the anomaly
            window_start = anomaly_date - pd.Timedelta(days=window_days)
            window_end = anomaly_date + pd.Timedelta(days=window_days)
            
            # Check which constituents have anomalies in this window
            constituent_anomalies_in_window = {}
            
            for ticker, ticker_anomalies in constituent_data.items():
                if 'start_date' not in ticker_anomalies.columns:
                    continue
                    
                # Find anomalies within the window
                window_mask = (
                    (ticker_anomalies['start_date'] >= window_start) & 
                    (ticker_anomalies['start_date'] <= window_end)
                )
                
                matching_anomalies = ticker_anomalies[window_mask]
                
                if not matching_anomalies.empty:
                    constituent_anomalies_in_window[ticker] = matching_anomalies
            
            # Calculate summary statistics
            affected_constituents = list(constituent_anomalies_in_window.keys())
            num_affected = len(affected_constituents)
            num_total = len(constituent_data)
            percentage_affected = (num_affected / num_total * 100) if num_total > 0 else 0
            
            # Get sectors of affected constituents
            affected_sectors = [sector_info.get(ticker, "Unknown") for ticker in affected_constituents]
            sector_counts = pd.Series(affected_sectors).value_counts().to_dict()
            
            # Determine if the anomaly is sector-specific
            sector_threshold = 0.5  # If 50% of affected constituents are from the same sector
            is_sector_specific = False
            dominant_sector = None
            
            if num_affected > 0:
                max_sector = max(sector_counts.items(), key=lambda x: x[1])
                if max_sector[1] / num_affected >= sector_threshold:
                    is_sector_specific = True
                    dominant_sector = max_sector[0]
            
            # Check if top 10 constituents (by weight) are affected
            top_constituents = config.TOP_SP500_CONSTITUENTS[:10]
            affected_top_constituents = [ticker for ticker in affected_constituents if ticker in top_constituents]
            num_top_affected = len(affected_top_constituents)
            pct_top_affected = (num_top_affected / len(top_constituents) * 100) if top_constituents else 0
            
            # Categorize the anomaly based on propagation patterns
            anomaly_type = "Unknown"
            if num_affected >= 15:  # Widespread anomaly affecting many constituents
                anomaly_type = "Widespread"
            elif is_sector_specific and num_affected >= 3:  # Sector-specific anomaly
                anomaly_type = f"Sector-specific ({dominant_sector})"
            elif 2 <= num_affected <= 5:  # Isolated anomaly affecting few constituents
                anomaly_type = "Isolated"
            elif num_affected <= 1:  # Index-only anomaly with little constituent impact
                anomaly_type = "Index-only"
            
            # Store results
            result = {
                'algorithm': algo,
                'index_anomaly_date': anomaly_date,
                'index_anomaly_score': anomaly_score,
                'affected_constituents': affected_constituents,
                'num_affected': num_affected,
                'percentage_affected': percentage_affected,
                'sector_counts': sector_counts,
                'is_sector_specific': is_sector_specific,
                'dominant_sector': dominant_sector,
                'affected_top_constituents': affected_top_constituents,
                'num_top_affected': num_top_affected,
                'pct_top_affected': pct_top_affected,
                'anomaly_type': anomaly_type
            }
            
            analysis_results[algo].append(result)
            
            # Log information about this anomaly
            logger.info(f"{algo} anomaly on {anomaly_date.strftime('%Y-%m-%d')}: {anomaly_type}, {num_affected}/{num_total} constituents affected ({percentage_affected:.1f}%)")
            
            if num_top_affected > 0:
                logger.info(f"  Top constituents affected: {', '.join(affected_top_constituents)} ({num_top_affected}/{len(top_constituents)})")
                
            if is_sector_specific:
                logger.info(f"  Sector-specific anomaly in {dominant_sector}: {sector_counts.get(dominant_sector, 0)} constituents")
    
    # If output directory is provided, save detailed results
    if output_dir is not None:
        save_analysis_results(analysis_results, output_dir)
        visualize_analysis_results(analysis_results, output_dir)
    
    return analysis_results


def save_analysis_results(analysis_results, output_dir):
    """
    Save analysis results to files.
    
    Args:
        analysis_results (dict): Dictionary containing analysis results
        output_dir (Path): Directory to save analysis results
    """
    for algo, results in analysis_results.items():
        if not results:
            continue
            
        # Create a DataFrame from the results
        results_data = []
        
        for result in results:
            row = {
                'index_anomaly_date': result['index_anomaly_date'],
                'index_anomaly_score': result['index_anomaly_score'],
                'num_affected': result['num_affected'],
                'percentage_affected': result['percentage_affected'],
                'is_sector_specific': result['is_sector_specific'],
                'dominant_sector': result['dominant_sector'],
                'num_top_affected': result['num_top_affected'],
                'pct_top_affected': result['pct_top_affected'],
                'anomaly_type': result['anomaly_type'],
                'affected_constituents': ','.join(result['affected_constituents'])
            }
            
            # Add sector counts
            for sector, count in result['sector_counts'].items():
                row[f'sector_{sector.replace(" ", "_")}'] = count
                
            results_data.append(row)
            
        # Create DataFrame
        results_df = pd.DataFrame(results_data)
        
        # Save to CSV
        csv_file = output_dir / f"{algo}_index_constituent_analysis.csv"
        results_df.to_csv(csv_file, index=False)
        logger.info(f"Saved {algo} analysis results to {csv_file}")
        
        # Save detailed anomaly report
        report_file = output_dir / f"{algo}_anomaly_report.txt"
        with open(report_file, 'w') as f:
            f.write(f"{algo.upper()} Index-Constituent Anomaly Analysis\n")
            f.write(f"==================================================\n\n")
            
            f.write(f"Total anomalies analyzed: {len(results)}\n")
            f.write(f"Results summary:\n")
            
            # Summarize anomaly types
            anomaly_types = [r['anomaly_type'] for r in results]
            type_counts = pd.Series(anomaly_types).value_counts().to_dict()
            
            for anomaly_type, count in type_counts.items():
                f.write(f"  {anomaly_type}: {count} anomalies\n")
                
            f.write("\n")
            
            # Write details for each anomaly
            for i, result in enumerate(results):
                f.write(f"Anomaly {i+1}: {result['index_anomaly_date'].strftime('%Y-%m-%d')}\n")
                f.write(f"  Type: {result['anomaly_type']}\n")
                f.write(f"  Index Score: {result['index_anomaly_score']:.4f}\n")
                f.write(f"  Affected Constituents: {result['num_affected']} ({result['percentage_affected']:.1f}%)\n")
                
                if result['is_sector_specific']:
                    f.write(f"  Dominant Sector: {result['dominant_sector']} "
                            f"({result['sector_counts'].get(result['dominant_sector'], 0)} constituents)\n")
                
                f.write(f"  Top Constituents Affected: {result['num_top_affected']}/{len(config.TOP_SP500_CONSTITUENTS[:10])} "
                       f"({result['pct_top_affected']:.1f}%)\n")
                
                f.write(f"  Affected Constituents: {', '.join(result['affected_constituents'])}\n\n")
                
        logger.info(f"Saved {algo} detailed report to {report_file}")
        
        # Save summary statistics
        summary_file = output_dir / f"{algo}_summary_statistics.txt"
        with open(summary_file, 'w') as f:
            f.write(f"{algo.upper()} Summary Statistics\n")
            f.write(f"=========================\n\n")
            
            f.write(f"Total index anomalies: {len(results)}\n\n")
            
            # Anomaly type distribution
            f.write("Anomaly Type Distribution:\n")
            for anomaly_type, count in type_counts.items():
                f.write(f"  {anomaly_type}: {count} ({count/len(results)*100:.1f}%)\n")
            
            # Affected constituents statistics
            affected_counts = [r['num_affected'] for r in results]
            if affected_counts:
                f.write("\nConstituent Impact Statistics:\n")
                f.write(f"  Average constituents affected: {np.mean(affected_counts):.2f}\n")
                f.write(f"  Median constituents affected: {np.median(affected_counts):.2f}\n")
                f.write(f"  Max constituents affected: {np.max(affected_counts)}\n")
                f.write(f"  Min constituents affected: {np.min(affected_counts)}\n")
            
            # Sector specificity
            sector_specific_count = sum(1 for r in results if r['is_sector_specific'])
            f.write(f"\nSector-specific anomalies: {sector_specific_count} ({sector_specific_count/len(results)*100:.1f}%)\n")
            
            if sector_specific_count > 0:
                dominant_sectors = [r['dominant_sector'] for r in results if r['is_sector_specific']]
                sector_counts = pd.Series(dominant_sectors).value_counts().to_dict()
                
                f.write("Dominant sectors in sector-specific anomalies:\n")
                for sector, count in sector_counts.items():
                    f.write(f"  {sector}: {count} ({count/sector_specific_count*100:.1f}%)\n")
            
            # Top constituent impact
            top_affected_counts = [r['num_top_affected'] for r in results]
            if top_affected_counts:
                f.write("\nTop Constituent Impact Statistics:\n")
                f.write(f"  Average top constituents affected: {np.mean(top_affected_counts):.2f}\n")
                f.write(f"  Median top constituents affected: {np.median(top_affected_counts):.2f}\n")
                f.write(f"  Max top constituents affected: {np.max(top_affected_counts)}\n")
                
        logger.info(f"Saved {algo} summary statistics to {summary_file}")


def visualize_analysis_results(analysis_results, output_dir):
    """
    Create visualizations of analysis results.
    
    Args:
        analysis_results (dict): Dictionary containing analysis results
        output_dir (Path): Directory to save visualizations
    """
    algorithms = list(analysis_results.keys())
    
    # Skip if no results
    if not any(analysis_results.values()):
        logger.warning("No analysis results to visualize")
        return
    
    # 1. Anomaly Type Distribution
    plt.figure(figsize=(12, 8))
    
    # Create data for stacked bar chart
    algo_types = []
    
    for algo in algorithms:
        results = analysis_results[algo]
        if not results:
            continue
            
        for anomaly_type in set(r['anomaly_type'] for r in results):
            count = sum(1 for r in results if r['anomaly_type'] == anomaly_type)
            algo_types.append({
                'Algorithm': algo.upper(),
                'Anomaly Type': anomaly_type,
                'Count': count
            })
    
    if algo_types:
        # Create DataFrame
        type_df = pd.DataFrame(algo_types)
        
        # Create stacked bar chart
        ax = plt.subplot(111)
        type_pivot = type_df.pivot(index='Algorithm', columns='Anomaly Type', values='Count')
        type_pivot.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
        
        plt.title('Anomaly Type Distribution by Algorithm')
        plt.xlabel('Algorithm')
        plt.ylabel('Number of Anomalies')
        plt.legend(title='Anomaly Type')
        plt.grid(True, alpha=0.3)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_dir / "anomaly_type_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved anomaly type distribution visualization to {output_dir / 'anomaly_type_distribution.png'}")
    
    # 2. Affected Constituents Boxplot
    plt.figure(figsize=(12, 8))
    
    affected_data = []
    
    for algo in algorithms:
        results = analysis_results[algo]
        if not results:
            continue
            
        for result in results:
            affected_data.append({
                'Algorithm': algo.upper(),
                'Affected Constituents (%)': result['percentage_affected']
            })
    
    if affected_data:
        affected_df = pd.DataFrame(affected_data)
        
        ax = plt.subplot(111)
        sns.boxplot(x='Algorithm', y='Affected Constituents (%)', data=affected_df, ax=ax)
        sns.swarmplot(x='Algorithm', y='Affected Constituents (%)', data=affected_df, color='black', size=4, alpha=0.7, ax=ax)
        
        plt.title('Percentage of Constituents Affected by Index Anomalies')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "affected_constituents_boxplot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved affected constituents visualization to {output_dir / 'affected_constituents_boxplot.png'}")
    
    # 3. Sector Impact Heatmap
    for algo in algorithms:
        results = analysis_results[algo]
        if not results:
            continue
            
        # Collect all sectors
        all_sectors = set()
        for result in results:
            all_sectors.update(result['sector_counts'].keys())
            
        # Create data for heatmap
        sectors_data = []
        
        for i, result in enumerate(results):
            anomaly_date = result['index_anomaly_date'].strftime('%Y-%m-%d')
            
            for sector in all_sectors:
                count = result['sector_counts'].get(sector, 0)
                sectors_data.append({
                    'Anomaly': f"{i+1}: {anomaly_date}",
                    'Sector': sector,
                    'Count': count
                })
        
        if sectors_data:
            # Create DataFrame and pivot for heatmap
            sectors_df = pd.DataFrame(sectors_data)
            pivot_df = sectors_df.pivot(index='Anomaly', columns='Sector', values='Count')
            
            # Fill NAs
            pivot_df = pivot_df.fillna(0)
            
            # Create heatmap
            plt.figure(figsize=(14, 10))
            sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='g')
            
            plt.title(f'{algo.upper()} - Sector Impact per Anomaly')
            plt.ylabel('Anomaly')
            plt.xlabel('Sector')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"{algo}_sector_impact_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved {algo} sector impact heatmap to {output_dir / f'{algo}_sector_impact_heatmap.png'}")
    
    # 4. Timeline of Anomalies
    plt.figure(figsize=(14, 8))
    
    timeline_data = []
    
    for algo in algorithms:
        results = analysis_results[algo]
        if not results:
            continue
            
        for result in results:
            timeline_data.append({
                'Algorithm': algo.upper(),
                'Date': result['index_anomaly_date'],
                'Anomaly Type': result['anomaly_type'],
                'Affected (%)': result['percentage_affected']
            })
    
    if timeline_data:
        timeline_df = pd.DataFrame(timeline_data)
        
        # Sort by date
        timeline_df = timeline_df.sort_values('Date')
        
        # Create scatter plot
        ax = plt.subplot(111)
        
        # Plot by algorithm with different markers
        markers = {'AIDA': 'o', 'IFOREST': '^', 'LOF': 's'}
        
        for algo in timeline_df['Algorithm'].unique():
            algo_data = timeline_df[timeline_df['Algorithm'] == algo]
            
            scatter = ax.scatter(
                algo_data['Date'], 
                algo_data['Affected (%)'],
                label=algo,
                marker=markers.get(algo, 'o'),
                s=100,
                alpha=0.7
            )
        
        plt.title('Timeline of Index Anomalies and Constituent Impact')
        plt.xlabel('Date')
        plt.ylabel('Affected Constituents (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "anomaly_timeline.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved anomaly timeline visualization to {output_dir / 'anomaly_timeline.png'}")
    
    # 5. Comprehensive comparison chart
    plt.figure(figsize=(16, 12))
    
    # Create a GridSpec
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    
    # Plot 1: Anomaly Type Distribution
    ax1 = plt.subplot(gs[0, 0])
    if algo_types:
        type_pivot = type_df.pivot(index='Algorithm', columns='Anomaly Type', values='Count')
        type_pivot.plot(kind='bar', stacked=True, ax=ax1, colormap='viridis')
        ax1.set_title('Anomaly Type Distribution')
        ax1.set_ylabel('Number of Anomalies')
        ax1.grid(True, alpha=0.3)
        ax1.legend(title='Anomaly Type', loc='upper right')
    
    # Plot 2: Affected Constituents Boxplot
    ax2 = plt.subplot(gs[0, 1])
    if affected_data:
        sns.boxplot(x='Algorithm', y='Affected Constituents (%)', data=affected_df, ax=ax2)
        ax2.set_title('Constituent Impact Distribution')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Sector Specificity
    ax3 = plt.subplot(gs[1, 0])
    sector_specific_data = []
    
    for algo in algorithms:
        results = analysis_results[algo]
        if not results:
            continue
            
        # Count sector-specific vs non-sector-specific
        sector_specific = sum(1 for r in results if r['is_sector_specific'])
        non_specific = len(results) - sector_specific
        
        sector_specific_data.append({
            'Algorithm': algo.upper(),
            'Type': 'Sector-specific',
            'Count': sector_specific
        })
        
        sector_specific_data.append({
            'Algorithm': algo.upper(),
            'Type': 'Non-sector-specific',
            'Count': non_specific
        })
    
    if sector_specific_data:
        specific_df = pd.DataFrame(sector_specific_data)
        specific_pivot = specific_df.pivot(index='Algorithm', columns='Type', values='Count')
        specific_pivot.plot(kind='bar', stacked=True, ax=ax3, colormap='Paired')
        ax3.set_title('Sector-specific vs Non-sector-specific Anomalies')
        ax3.set_ylabel('Number of Anomalies')
        ax3.grid(True, alpha=0.3)
        ax3.legend(title='Type')
    
    # Plot 4: Top Constituent Impact
    ax4 = plt.subplot(gs[1, 1])
    top_impact_data = []
    
    for algo in algorithms:
        results = analysis_results[algo]
        if not results:
            continue
            
        top_affected = [r['pct_top_affected'] for r in results]
        
        if top_affected:
            top_impact_data.append({
                'Algorithm': algo.upper(),
                'Mean': np.mean(top_affected),
                'Median': np.median(top_affected),
                'Max': np.max(top_affected)
            })
    
    if top_impact_data:
        impact_df = pd.DataFrame(top_impact_data)
        
        # Create grouped bar chart
        x = np.arange(len(impact_df))
        width = 0.25
        
        ax4.bar(x - width, impact_df['Mean'], width, label='Mean', color='skyblue')
        ax4.bar(x, impact_df['Median'], width, label='Median', color='lightgreen')
        ax4.bar(x + width, impact_df['Max'], width, label='Max', color='salmon')
        
        ax4.set_title('Top Constituent Impact Statistics')
        ax4.set_ylabel('% of Top Constituents Affected')
        ax4.set_xticks(x)
        ax4.set_xticklabels(impact_df['Algorithm'])
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "comprehensive_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comprehensive comparison to {output_dir / 'comprehensive_comparison.png'}")


def main():
    """
    Main function to analyze constituent anomalies.
    """
    parser = argparse.ArgumentParser(
        description="Analyze the relationship between S&P 500 index anomalies and constituent anomalies"
    )
    parser.add_argument(
        "--index-results", 
        type=str, 
        default=str(config.DATA_DIR / "subsequence_results"),
        help="Directory containing S&P 500 index anomaly detection results"
    )
    parser.add_argument(
        "--constituent-results", 
        type=str, 
        default=str(config.DATA_DIR / "constituent_results"),
        help="Directory containing constituent anomaly detection results"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=str(config.DATA_DIR / "constituent_analysis"),
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--window-days", 
        type=int, 
        default=3,
        help="Number of days to look around each index anomaly"
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["aida", "iforest", "lof", "all"],
        default=["all"],
        help="Algorithms to include in the analysis"
    )
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    index_results_dir = Path(args.index_results)
    constituent_results_dir = Path(args.constituent_results)
    output_dir = Path(args.output)
    
    # Determine algorithms to include
    algorithms = args.algorithms
    if "all" in algorithms:
        algorithms = ["aida", "iforest", "lof"]
    
    # Ensure output directory exists
    ensure_directory_exists(output_dir)
    
    # Load S&P 500 index anomalies
    logger.info("Loading S&P 500 index anomalies...")
    index_anomalies = load_index_anomalies(index_results_dir, algorithms)
    
    # Check if we have any data
    if not index_anomalies:
        logger.error("No index anomalies found. Exiting.")
        return
    
    # Load constituent anomalies
    logger.info("Loading constituent anomalies...")
    constituent_anomalies = load_constituent_anomalies(constituent_results_dir, algorithms)
    
    # Get list of tickers
    all_tickers = set()
    for algo, ticker_dict in constituent_anomalies.items():
        all_tickers.update(ticker_dict.keys())
    
    logger.info(f"Found data for {len(all_tickers)} constituent stocks")
    
    # Get sector information for tickers
    logger.info("Getting sector information for tickers...")
    sector_info = get_ticker_sector_info(all_tickers)
    
    # Analyze the relationship between index and constituent anomalies
    logger.info(f"Analyzing index-constituent relationship with {args.window_days}-day window...")
    analysis_results = analyze_index_constituent_relationship(
        index_anomalies,
        constituent_anomalies,
        sector_info,
        window_days=args.window_days,
        output_dir=output_dir,
        algorithms=algorithms
    )
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
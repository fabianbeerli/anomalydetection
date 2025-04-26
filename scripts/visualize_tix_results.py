#!/usr/bin/env python
"""
Script to visualize TIX analysis results.
Creates comprehensive visualizations that combine anomaly detection and feature importance.
"""
import os
import sys
import logging
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.dates import DateFormatter
import matplotlib.gridspec as gridspec

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.utils.helpers import ensure_directory_exists, load_ticker_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_price_data(data_file):
    """
    Load the price data for visualization.
    
    Args:
        data_file (Path): Path to the data file
        
    Returns:
        pandas.DataFrame: Price data
    """
    try:
        df = load_ticker_data(data_file)
        
        if df is None or df.empty:
            logger.error(f"Failed to load data from {data_file}")
            return None
        
        # Ensure we have a 'Close' column for visualization
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading price data: {e}")
        return None


def load_anomalies_and_tix(tix_results_dir, subsequence_results_dir, algorithm='aida', window_size=3, overlap_type='overlap'):
    """
    Load anomalies and their TIX explanations.
    
    Args:
        tix_results_dir (Path): Directory containing TIX results
        subsequence_results_dir (Path): Directory containing subsequence results
        algorithm (str): Algorithm name ('aida', 'iforest', 'lof')
        window_size (int): Window size
        overlap_type (str): 'overlap' or 'nonoverlap'
        
    Returns:
        tuple: (anomalies, tix_results)
    """
    try:
        # Load anomalies
        anomalies_file = subsequence_results_dir / algorithm / f"w{window_size}_{overlap_type}" / f"{algorithm}_anomalies.csv"
        
        if not anomalies_file.exists():
            logger.error(f"Anomalies file not found: {anomalies_file}")
            return None, None
        
        anomalies = pd.read_csv(anomalies_file)
        
        if anomalies.empty:
            logger.warning(f"No anomalies found for {algorithm} (w{window_size}_{overlap_type})")
            return anomalies, None
        
        # Convert date columns to datetime
        for date_col in ['start_date', 'end_date']:
            if date_col in anomalies.columns:
                anomalies[date_col] = pd.to_datetime(anomalies[date_col])
        
        # Load TIX results
        tix_results = {}
        tix_base_dir = tix_results_dir / "subsequence" / algorithm / f"w{window_size}_{overlap_type}" / "tix_analysis"
        
        if not tix_base_dir.exists():
            logger.warning(f"TIX results directory not found: {tix_base_dir}")
            return anomalies, None
        
        # For each anomaly, try to load its TIX results
        for _, anomaly in anomalies.iterrows():
            if 'subsequence_idx' not in anomaly:
                continue
                
            subsequence_idx = int(anomaly['subsequence_idx'])
            anomaly_dir = tix_base_dir / f"{algorithm}_anomaly_{subsequence_idx}"
            
            importance_file = list(anomaly_dir.glob("tix_results_point_*.csv"))
            
            if importance_file:
                importance_df = pd.read_csv(importance_file[0])
                # Convert to dictionary
                importance = dict(zip(importance_df['feature_name'], importance_df['importance_score']))
                tix_results[subsequence_idx] = {
                    'feature_importance': importance,
                    'top_features': importance_df.sort_values('importance_score', ascending=False)['feature_name'].head(5).tolist()
                }
        
        return anomalies, tix_results
        
    except Exception as e:
        logger.error(f"Error loading anomalies and TIX results: {e}")
        return None, None


def visualize_anomalies_with_explanations(price_data, anomalies, tix_results, output_dir, algorithm='aida', window_size=3, overlap_type='overlap'):
    """
    Create a visualization that combines anomalies and their explanations.
    
    Args:
        price_data (pandas.DataFrame): Price data
        anomalies (pandas.DataFrame): Anomalies
        tix_results (dict): TIX results
        output_dir (Path): Directory to save output
        algorithm (str): Algorithm name
        window_size (int): Window size
        overlap_type (str): 'overlap' or 'nonoverlap'
    """
    try:
        if anomalies is None or anomalies.empty:
            logger.warning("No anomalies to visualize")
            return
        
        # Ensure output directory exists
        ensure_directory_exists(output_dir)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
        
        # Plot 1: Price with anomalies
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(price_data.index, price_data['Close'], label='S&P 500 Close Price', color='blue', alpha=0.7)
        
        # Mark anomalies
        if 'start_date' in anomalies.columns and 'end_date' in anomalies.columns:
            for _, anomaly in anomalies.iterrows():
                start_date = anomaly['start_date']
                end_date = anomaly['end_date']
                
                if pd.notna(start_date) and pd.notna(end_date):
                    # Find the price at this date
                    if start_date in price_data.index:
                        price = price_data.loc[start_date, 'Close']
                    else:
                        # Find closest date
                        closest_idx = (price_data.index - start_date).abs().argmin()
                        price = price_data['Close'].iloc[closest_idx]
                    
                    # Highlight anomaly period
                    ax1.axvspan(start_date, end_date, alpha=0.2, color='red')
                    
                    # Mark anomaly point
                    ax1.scatter([start_date], [price], color='red', s=100, zorder=5)
                    
                    # Add label with subsequence index
                    if 'subsequence_idx' in anomaly:
                        idx = int(anomaly['subsequence_idx'])
                        label = f"#{idx}"
                        if tix_results and idx in tix_results:
                            # Add top feature
                            top_feature = tix_results[idx]['top_features'][0] if tix_results[idx]['top_features'] else "Unknown"
                            label += f"\nTop: {top_feature}"
                        
                        ax1.annotate(
                            label, 
                            (start_date, price), 
                            xytext=(10, 10),
                            textcoords='offset points',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
                        )
        
        ax1.set_title(f'S&P 500 with {algorithm.upper()} Anomalies and TIX Explanations (w{window_size}, {overlap_type})')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Close Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Feature importance for each anomaly
        ax2 = fig.add_subplot(gs[1])
        
        if tix_results:
            # Collect top features across all anomalies
            all_features = set()
            for idx, result in tix_results.items():
                all_features.update(result['top_features'])
            
            # Create a matrix of importance scores
            feature_list = sorted(all_features)
            anomaly_indices = sorted(tix_results.keys())
            
            importance_matrix = np.zeros((len(anomaly_indices), len(feature_list)))
            
            for i, idx in enumerate(anomaly_indices):
                for j, feature in enumerate(feature_list):
                    importance_matrix[i, j] = tix_results[idx]['feature_importance'].get(feature, 0)
            
            # Create heatmap
            sns.heatmap(
                importance_matrix, 
                annot=True, 
                fmt=".2f", 
                xticklabels=feature_list, 
                yticklabels=[f"Anomaly #{idx}" for idx in anomaly_indices],
                cmap="YlGnBu",
                ax=ax2
            )
            
            ax2.set_title('Feature Importance for Each Anomaly')
            ax2.set_xlabel('Features')
            ax2.set_ylabel('Anomalies')
        else:
            ax2.text(0.5, 0.5, 'No TIX results available', ha='center', va='center', fontsize=14)
        
        # Save figure
        output_file = output_dir / f"{algorithm}_tix_visualization_w{window_size}_{overlap_type}.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {output_file}")
        
        # Also create individual visualizations for each anomaly with detailed TIX analysis
        if tix_results:
            for idx, result in tix_results.items():
                # Find this anomaly
                anomaly = anomalies[anomalies['subsequence_idx'] == idx].iloc[0] if idx in anomalies['subsequence_idx'].values else None
                
                if anomaly is None:
                    continue
                
                # Get dates
                start_date = anomaly['start_date'] if 'start_date' in anomaly else None
                end_date = anomaly['end_date'] if 'end_date' in anomaly else None
                
                if start_date is None or end_date is None:
                    continue
                
                # Create a window around the anomaly
                buffer_days = 10
                window_start = start_date - pd.Timedelta(days=buffer_days)
                window_end = end_date + pd.Timedelta(days=buffer_days)
                
                # Get price data for this window
                window_data = price_data.loc[window_start:window_end]
                
                if window_data.empty:
                    continue
                
                # Create figure
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                
                # Plot price
                ax1.plot(window_data.index, window_data['Close'], label='S&P 500 Close Price', color='blue', alpha=0.7)
                
                # Highlight anomaly period
                ax1.axvspan(start_date, end_date, alpha=0.2, color='red')
                
                # Find price at anomaly
                if start_date in window_data.index:
                    price = window_data.loc[start_date, 'Close']
                else:
                    # Find closest date
                    closest_idx = (window_data.index - start_date).abs().argmin()
                    price = window_data['Close'].iloc[closest_idx]
                
                # Mark anomaly point
                ax1.scatter([start_date], [price], color='red', s=100, zorder=5)
                
                ax1.set_title(f'Anomaly #{idx} ({start_date.strftime("%Y-%m-%d")})')
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Close Price')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot feature importance
                importance = sorted(result['feature_importance'].items(), key=lambda x: x[1], reverse=True)
                features, scores = zip(*importance[:10])  # Top 10 features
                
                ax2.barh(features, scores)
                ax2.set_title('Feature Importance')
                ax2.set_xlabel('Importance Score')
                ax2.set_ylabel('Feature')
                
                # Save figure
                output_file = output_dir / f"{algorithm}_anomaly_{idx}_detail.png"
                plt.tight_layout()
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                
    except Exception as e:
        logger.error(f"Error visualizing anomalies with explanations: {e}")


def visualize_constituent_tix_results(tix_results_dir, constituent_results_dir, processed_dir, output_dir):
    """
    Visualize TIX results for constituent anomalies.
    
    Args:
        tix_results_dir (Path): Directory containing TIX results
        constituent_results_dir (Path): Directory containing constituent results
        processed_dir (Path): Directory containing processed data
        output_dir (Path): Directory to save output
    """
    try:
        # Ensure output directory exists
        ensure_directory_exists(output_dir)
        
        # Find all constituents with TIX results
        tix_constituent_dir = tix_results_dir / "constituent"
        
        if not tix_constituent_dir.exists():
            logger.warning(f"Constituent TIX directory not found: {tix_constituent_dir}")
            return
        
        # Find all tickers with TIX analysis
        tickers_with_tix = []
        
        for ticker_dir in tix_constituent_dir.iterdir():
            if ticker_dir.is_dir() and (ticker_dir / "tix_analysis").exists():
                tickers_with_tix.append(ticker_dir.name)
        
        if not tickers_with_tix:
            logger.warning("No constituents with TIX analysis found")
            return
        
        logger.info(f"Found {len(tickers_with_tix)} constituents with TIX analysis")
        
        # For each ticker, create a visualization
        for ticker in tickers_with_tix:
            # Load price data
            price_file = processed_dir / f"{ticker}_processed.csv"
            
            if not price_file.exists():
                logger.warning(f"Price data not found for {ticker}: {price_file}")
                continue
            
            price_data = load_price_data(price_file)
            
            if price_data is None:
                continue
            
            # Load anomalies
            anomalies_file = constituent_results_dir / "aida" / ticker / "aida_anomalies.csv"
            
            if not anomalies_file.exists():
                logger.warning(f"Anomalies file not found for {ticker}: {anomalies_file}")
                continue
            
            anomalies = pd.read_csv(anomalies_file)
            
            if anomalies.empty:
                logger.warning(f"No anomalies found for {ticker}")
                continue
            
            # Convert date columns to datetime
            for date_col in ['date', 'start_date', 'end_date']:
                if date_col in anomalies.columns:
                    anomalies[date_col] = pd.to_datetime(anomalies[date_col])
            
            # Load TIX results
            tix_results = {}
            tix_analysis_dir = tix_constituent_dir / ticker / "tix_analysis"
            
            for anomaly_dir in tix_analysis_dir.iterdir():
                if not anomaly_dir.is_dir() or not anomaly_dir.name.startswith("anomaly_"):
                    continue
                
                # Extract anomaly index
                try:
                    anomaly_idx = int(anomaly_dir.name.split("_")[1])
                except:
                    continue
                
                # Find importance file
                importance_files = list(anomaly_dir.glob("tix_results_point_*.csv"))
                
                if not importance_files:
                    continue
                
                # Load importance
                importance_df = pd.read_csv(importance_files[0])
                
                # Convert to dictionary
                importance = dict(zip(importance_df['feature_name'], importance_df['importance_score']))
                
                tix_results[anomaly_idx] = {
                    'feature_importance': importance,
                    'top_features': importance_df.sort_values('importance_score', ascending=False)['feature_name'].head(5).tolist()
                }
            
            if not tix_results:
                logger.warning(f"No TIX results found for {ticker}")
                continue
            
            # Create a visualization
            ticker_output_dir = output_dir / ticker
            ensure_directory_exists(ticker_output_dir)
            
            # Create figure with subplots
            fig = plt.figure(figsize=(16, 10))
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)
            
            # Plot 1: Price with anomalies
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(price_data.index, price_data['Close'], label=f'{ticker} Close Price', color='blue', alpha=0.7)
            
            # Mark anomalies
            date_col = 'date' if 'date' in anomalies.columns else ('start_date' if 'start_date' in anomalies.columns else None)
            
            if date_col is not None:
                for _, anomaly in anomalies.iterrows():
                    anomaly_date = anomaly[date_col]
                    
                    if pd.notna(anomaly_date):
                        # Find the price at this date
                        if anomaly_date in price_data.index:
                            price = price_data.loc[anomaly_date, 'Close']
                        else:
                            # Find closest date
                            closest_idx = (price_data.index - anomaly_date).abs().argmin()
                            price = price_data['Close'].iloc[closest_idx]
                        
                        # Mark anomaly point
                        ax1.scatter([anomaly_date], [price], color='red', s=100, zorder=5)
                        
                        # Add label with anomaly index
                        idx = int(anomaly['index']) if 'index' in anomaly else None
                        
                        if idx is not None and idx in tix_results:
                            # Add top feature
                            top_feature = tix_results[idx]['top_features'][0] if tix_results[idx]['top_features'] else "Unknown"
                            label = f"#{idx}\nTop: {top_feature}"
                            
                            ax1.annotate(
                                label, 
                                (anomaly_date, price), 
                                xytext=(10, 10),
                                textcoords='offset points',
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
                            )
            
            ax1.set_title(f'{ticker} with AIDA Anomalies and TIX Explanations')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Close Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Feature importance for each anomaly
            ax2 = fig.add_subplot(gs[1])
            
            # Collect top features across all anomalies
            all_features = set()
            for idx, result in tix_results.items():
                all_features.update(result['top_features'])
            
            # Create a matrix of importance scores
            feature_list = sorted(all_features)
            anomaly_indices = sorted(tix_results.keys())
            
            importance_matrix = np.zeros((len(anomaly_indices), len(feature_list)))
            
            for i, idx in enumerate(anomaly_indices):
                for j, feature in enumerate(feature_list):
                    importance_matrix[i, j] = tix_results[idx]['feature_importance'].get(feature, 0)
            
            # Create heatmap
            sns.heatmap(
                importance_matrix, 
                annot=True, 
                fmt=".2f", 
                xticklabels=feature_list, 
                yticklabels=[f"Anomaly #{idx}" for idx in anomaly_indices],
                cmap="YlGnBu",
                ax=ax2
            )
            
            ax2.set_title('Feature Importance for Each Anomaly')
            ax2.set_xlabel('Features')
            ax2.set_ylabel('Anomalies')
            
            # Save figure
            output_file = ticker_output_dir / f"{ticker}_tix_visualization.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved for {ticker}")
            
            # Also create individual visualizations for each anomaly with detailed TIX analysis
            for idx, result in tix_results.items():
                # Find this anomaly
                anomaly = anomalies[anomalies['index'] == idx].iloc[0] if 'index' in anomalies.columns and idx in anomalies['index'].values else None
                
                if anomaly is None:
                    continue
                
                # Get date
                anomaly_date = anomaly[date_col] if date_col in anomaly and pd.notna(anomaly[date_col]) else None
                
                if anomaly_date is None:
                    continue
                
                # Create a window around the anomaly
                buffer_days = 10
                window_start = anomaly_date - pd.Timedelta(days=buffer_days)
                window_end = anomaly_date + pd.Timedelta(days=buffer_days)
                
                # Get price data for this window
                window_data = price_data.loc[window_start:window_end]
                
                if window_data.empty:
                    continue
                
                # Create figure
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                
                # Plot price
                ax1.plot(window_data.index, window_data['Close'], label=f'{ticker} Close Price', color='blue', alpha=0.7)
                
                # Find price at anomaly
                if anomaly_date in window_data.index:
                    price = window_data.loc[anomaly_date, 'Close']
                else:
                    # Find closest date
                    closest_idx = (window_data.index - anomaly_date).abs().argmin()
                    price = window_data['Close'].iloc[closest_idx]
                
                # Mark anomaly point
                ax1.scatter([anomaly_date], [price], color='red', s=100, zorder=5)
                
                ax1.set_title(f'{ticker} Anomaly #{idx} ({anomaly_date.strftime("%Y-%m-%d")})')
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Close Price')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot feature importance
                importance = sorted(result['feature_importance'].items(), key=lambda x: x[1], reverse=True)
                features, scores = zip(*importance[:10])  # Top 10 features
                
                ax2.barh(features, scores)
                ax2.set_title('Feature Importance')
                ax2.set_xlabel('Importance Score')
                ax2.set_ylabel('Feature')
                
                # Save figure
                output_file = ticker_output_dir / f"{ticker}_anomaly_{idx}_detail.png"
                plt.tight_layout()
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
        
        logger.info(f"Constituent TIX visualizations completed for {len(tickers_with_tix)} tickers")
        
    except Exception as e:
        logger.error(f"Error visualizing constituent TIX results: {e}")


def visualize_multi_ts_tix_results(tix_results_dir, multi_ts_results_dir, sp500_data_file, output_dir):
    """
    Visualize TIX results for multi-TS anomalies.
    
    Args:
        tix_results_dir (Path): Directory containing TIX results
        multi_ts_results_dir (Path): Directory containing multi-TS results
        sp500_data_file (Path): Path to S&P 500 data file
        output_dir (Path): Directory to save output
    """
    try:
        # Ensure output directory exists
        ensure_directory_exists(output_dir)
        
        # Load S&P 500 data
        sp500_data = load_price_data(sp500_data_file)
        
        if sp500_data is None:
            logger.error("Failed to load S&P 500 data")
            return
        
        # Find all window sizes and overlap types with TIX results
        tix_multi_ts_dir = tix_results_dir / "multi_ts"
        
        if not tix_multi_ts_dir.exists():
            logger.warning(f"Multi-TS TIX directory not found: {tix_multi_ts_dir}")
            return
        
        # Find all configurations
        configs = []
        
        for config_dir in tix_multi_ts_dir.iterdir():
            if config_dir.is_dir() and config_dir.name.startswith("w"):
                try:
                    parts = config_dir.name.split("_")
                    window_size = int(parts[0][1:])
                    overlap_type = parts[1]
                    configs.append((window_size, overlap_type))
                except:
                    continue
        
        if not configs:
            logger.warning("No multi-TS configurations with TIX analysis found")
            return
        
        logger.info(f"Found {len(configs)} multi-TS configurations with TIX analysis")
        
        # For each configuration, create a visualization
        for window_size, overlap_type in configs:
            # Load anomalies
            anomalies_file = multi_ts_results_dir / f"multi_ts_w{window_size}_{overlap_type}" / "aida" / "aida_multi_ts_anomalies.csv"
            
            if not anomalies_file.exists():
                logger.warning(f"Anomalies file not found: {anomalies_file}")
                continue
            
            anomalies = pd.read_csv(anomalies_file)
            
            if anomalies.empty:
                logger.warning(f"No anomalies found for w{window_size}_{overlap_type}")
                continue
            
            # Convert date columns to datetime
            for date_col in ['start_date', 'end_date']:
                if date_col in anomalies.columns:
                    anomalies[date_col] = pd.to_datetime(anomalies[date_col])
            
            # Load TIX results
            tix_results = {}
            tix_analysis_dir = tix_multi_ts_dir / f"w{window_size}_{overlap_type}" / "tix_analysis"
            
            # Look for anomaly directories
            for anomaly_dir in tix_analysis_dir.glob("multi_ts_anomaly_*"):
                if not anomaly_dir.is_dir():
                    continue
                
                # Extract anomaly index
                try:
                    anomaly_idx = int(anomaly_dir.name.split("_")[-1])
                except:
                    continue
                
                # Find stock importance files
                stock_importance = {}
                
                for stock_dir in anomaly_dir.iterdir():
                    if not stock_dir.is_dir():
                        continue
                    
                    stock_name = stock_dir.name
                    
                    # Find importance file
                    importance_files = list(stock_dir.glob("tix_results_point_*.csv"))
                    
                    if not importance_files:
                        continue
                    
                    # Load importance
                    importance_df = pd.read_csv(importance_files[0])
                    
                    # Convert to dictionary
                    importance = dict(zip(importance_df['feature_name'], importance_df['importance_score']))
                    
                    stock_importance[stock_name] = {
                        'feature_importance': importance,
                        'top_features': importance_df.sort_values('importance_score', ascending=False)['feature_name'].head(5).tolist()
                    }
                
                if stock_importance:
                    tix_results[anomaly_idx] = {
                        'stock_importance': stock_importance,
                        'stocks_analyzed': len(stock_importance)
                    }
            
            if not tix_results:
                logger.warning(f"No TIX results found for w{window_size}_{overlap_type}")
                continue
            
            # Create a visualization
            config_output_dir = output_dir / f"w{window_size}_{overlap_type}"
            ensure_directory_exists(config_output_dir)
            
            # Create figure with subplots
            fig = plt.figure(figsize=(16, 12))
            gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 2], hspace=0.3)
            
            # Plot 1: Price with anomalies
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(sp500_data.index, sp500_data['Close'], label='S&P 500 Close Price', color='blue', alpha=0.7)
            
            # Mark anomalies
            if 'start_date' in anomalies.columns and 'end_date' in anomalies.columns:
                for _, anomaly in anomalies.iterrows():
                    if 'time_period_idx' not in anomaly:
                        continue
                        
                    time_period_idx = int(anomaly['time_period_idx'])
                    
                    if time_period_idx not in tix_results:
                        continue
                    
                    start_date = anomaly['start_date']
                    end_date = anomaly['end_date']
                    
                    if pd.notna(start_date) and pd.notna(end_date):
                        # Find the price at this date
                        if start_date in sp500_data.index:
                            price = sp500_data.loc[start_date, 'Close']
                        else:
                            # Find closest date
                            closest_idx = (sp500_data.index - start_date).abs().argmin()
                            price = sp500_data['Close'].iloc[closest_idx]
                        
                        # Highlight anomaly period
                        ax1.axvspan(start_date, end_date, alpha=0.2, color='red')
                        
                        # Mark anomaly point
                        ax1.scatter([start_date], [price], color='red', s=100, zorder=5)
                        
                        # Add label with number of stocks analyzed
                        stocks_analyzed = tix_results[time_period_idx]['stocks_analyzed']
                        label = f"#{time_period_idx}\n{stocks_analyzed} stocks"
                        
                        ax1.annotate(
                            label, 
                            (start_date, price), 
                            xytext=(10, 10),
                            textcoords='offset points',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8)
                        )
            
            ax1.set_title(f'S&P 500 with Multi-TS Anomalies and TIX Explanations (w{window_size}, {overlap_type})')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Close Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Summary of stock importance across anomalies
            ax2 = fig.add_subplot(gs[1])
            
            # Count number of times each stock appears in top explanations
            stock_counts = {}
            
            for idx, result in tix_results.items():
                for stock in result['stock_importance'].keys():
                    stock_counts[stock] = stock_counts.get(stock, 0) + 1
            
            # Sort by count
            sorted_stocks = sorted(stock_counts.items(), key=lambda x: x[1], reverse=True)
            stocks, counts = zip(*sorted_stocks[:20])  # Top 20 stocks
            
            ax2.barh(stocks, counts)
            ax2.set_title('Stocks Most Frequently Contributing to Anomalies')
            ax2.set_xlabel('Count')
            ax2.set_ylabel('Stock')
            
            # Plot 3: Feature importance across stocks for the first anomaly
            ax3 = fig.add_subplot(gs[2])
            
            if tix_results:
                # Get first anomaly
                first_anomaly_idx = list(tix_results.keys())[0]
                result = tix_results[first_anomaly_idx]
                
                # Create a separate visualization for each anomaly
                visualize_multi_ts_anomaly(
                    sp500_data,
                    anomalies.loc[anomalies['time_period_idx'] == first_anomaly_idx].iloc[0] if first_anomaly_idx in anomalies['time_period_idx'].values else None,
                    result,
                    config_output_dir,
                    first_anomaly_idx,
                    ax3
                )
            else:
                ax3.text(0.5, 0.5, 'No detailed TIX results available', ha='center', va='center', fontsize=14)
            
            # Save figure
            output_file = config_output_dir / f"multi_ts_tix_visualization_w{window_size}_{overlap_type}.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Multi-TS visualization saved for w{window_size}_{overlap_type}")
            
            # Create individual visualizations for each anomaly
            for idx in list(tix_results.keys()):
                anomaly = anomalies.loc[anomalies['time_period_idx'] == idx].iloc[0] if idx in anomalies['time_period_idx'].values else None
                result = tix_results[idx]
                
                visualize_multi_ts_anomaly(
                    sp500_data,
                    anomaly,
                    result,
                    config_output_dir,
                    idx
                )
        
        logger.info(f"Multi-TS TIX visualizations completed for {len(configs)} configurations")
        
    except Exception as e:
        logger.error(f"Error visualizing multi-TS TIX results: {e}")


def visualize_multi_ts_anomaly(sp500_data, anomaly, tix_result, output_dir, anomaly_idx, ax=None):
    """
    Visualize a single multi-TS anomaly with its TIX explanation.
    
    Args:
        sp500_data (pandas.DataFrame): S&P 500 price data
        anomaly (pandas.Series): Anomaly information
        tix_result (dict): TIX result for this anomaly
        output_dir (Path): Directory to save output
        anomaly_idx (int): Anomaly index
        ax (matplotlib.axes.Axes): Axes to plot on (optional)
    """
    try:
        if anomaly is None or not tix_result.get('stock_importance'):
            return
            
        # Extract dates
        start_date = anomaly['start_date'] if 'start_date' in anomaly and pd.notna(anomaly['start_date']) else None
        end_date = anomaly['end_date'] if 'end_date' in anomaly and pd.notna(anomaly['end_date']) else None
        
        if start_date is None or end_date is None:
            return
            
        # Create a window around the anomaly for individual visualization
        if ax is None:
            buffer_days = 10
            window_start = start_date - pd.Timedelta(days=buffer_days)
            window_end = end_date + pd.Timedelta(days=buffer_days)
            
            # Get price data for this window
            window_data = sp500_data.loc[window_start:window_end]
            
            if window_data.empty:
                return
                
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [1, 2]})
            
            # Plot price
            ax1.plot(window_data.index, window_data['Close'], label='S&P 500 Close Price', color='blue', alpha=0.7)
            
            # Highlight anomaly period
            ax1.axvspan(start_date, end_date, alpha=0.2, color='red')
            
            # Find price at anomaly start
            if start_date in window_data.index:
                price = window_data.loc[start_date, 'Close']
            else:
                # Find closest date
                closest_idx = (window_data.index - start_date).abs().argmin()
                price = window_data['Close'].iloc[closest_idx]
            
            # Mark anomaly point
            ax1.scatter([start_date], [price], color='red', s=100, zorder=5)
            
            ax1.set_title(f'Multi-TS Anomaly #{anomaly_idx} ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")})')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Close Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        else:
            # Use provided axes for the heatmap only
            ax2 = ax
        
        # Create a heatmap of stock-feature importance
        stock_importance = tix_result['stock_importance']
        
        # Collect all features across stocks
        all_features = set()
        for stock, importance in stock_importance.items():
            if 'top_features' in importance:
                all_features.update(importance['top_features'])
        
        # Sort features by overall importance
        feature_importance = {}
        for feature in all_features:
            importance_sum = sum(stock['feature_importance'].get(feature, 0) for stock in stock_importance.values())
            feature_importance[feature] = importance_sum
        
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        top_feature_names = [f[0] for f in top_features]
        
        # Sort stocks by overall importance
        stock_list = list(stock_importance.keys())
        stock_overall_importance = {}
        
        for stock in stock_list:
            importance_sum = sum(stock_importance[stock]['feature_importance'].get(feature, 0) for feature in top_feature_names)
            stock_overall_importance[stock] = importance_sum
        
        top_stocks = sorted(stock_overall_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        top_stock_names = [s[0] for s in top_stocks]
        
        # Create importance matrix
        importance_matrix = np.zeros((len(top_stock_names), len(top_feature_names)))
        
        for i, stock in enumerate(top_stock_names):
            for j, feature in enumerate(top_feature_names):
                if stock in stock_importance and 'feature_importance' in stock_importance[stock]:
                    importance_matrix[i, j] = stock_importance[stock]['feature_importance'].get(feature, 0)
        
        # Create heatmap
        sns.heatmap(
            importance_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=top_feature_names,
            yticklabels=top_stock_names,
            cmap="YlGnBu",
            ax=ax2
        )
        
        ax2.set_title('Feature Importance Across Top Stocks')
        ax2.set_xlabel('Features')
        ax2.set_ylabel('Stocks')
        
        # Save individual visualization if we created a new figure
        if ax is None:
            output_file = output_dir / f"multi_ts_anomaly_{anomaly_idx}_detail.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
        
    except Exception as e:
        logger.error(f"Error visualizing multi-TS anomaly {anomaly_idx}: {e}")

def run_tix_analysis_workflow(config_args):
    """
    Run TIX analysis workflow to explain anomalies detected by AIDA.
    
    Args:
        config_args (argparse.Namespace): Configuration arguments
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Starting TIX Analysis Workflow")
    
    try:
        # Create output directory for TIX results
        tix_results_dir = Path(config_args.output_dir) / "tix_results"
        ensure_directory_exists(tix_results_dir)
        
        # Setup window sizes and overlap settings for subsequence analysis
        window_sizes = [3, 5]  # Default window sizes
        if config_args.window_sizes:
            window_sizes = [int(size) for size in config_args.window_sizes.split(',')]
        
        overlap_settings = []
        if not config_args.only_overlap and not config_args.only_nonoverlap:
            overlap_settings = ["overlap", "nonoverlap"]
        elif config_args.only_overlap:
            overlap_settings = ["overlap"]
        elif config_args.only_nonoverlap:
            overlap_settings = ["nonoverlap"]
        
        # Determine TIX analysis types to run
        tix_analysis_types = []
        if config_args.run_all or (not config_args.tix_subsequence_only and not config_args.tix_constituent_only and not config_args.tix_multi_ts_only):
            tix_analysis_types = ["--run-all-tix"]
        else:
            if config_args.tix_subsequence_only:
                tix_analysis_types.append("--run-subsequence-tix")
            if config_args.tix_constituent_only:
                tix_analysis_types.append("--run-constituent-tix")
            if config_args.tix_multi_ts_only:
                tix_analysis_types.append("--run-multi-ts-tix")
        
        # Run TIX analysis script
        cmd = [
            sys.executable,
            str(Path(__file__).parent / "run_tix_analysis.py"),
            "--output-dir", str(tix_results_dir)
        ]
        
        # Add analysis types
        cmd.extend(tix_analysis_types)
        
        # Add window sizes
        if config_args.window_sizes:
            cmd.extend(["--window-sizes", config_args.window_sizes])
        
        # Add overlap flags
        if config_args.only_overlap:
            cmd.append("--only-overlap")
        elif config_args.only_nonoverlap:
            cmd.append("--only-nonoverlap")
        
        # Add constituent list if specified
        if hasattr(config_args, 'constituents') and config_args.constituents:
            cmd.extend(["--constituents", config_args.constituents])
        
        logger.info(f"Running TIX analysis command: {' '.join(cmd)}")
        
        # Execute command
        result = subprocess.run(cmd, check=True)
        
        # Run TIX visualization
        vis_cmd = [
            sys.executable,
            str(Path(__file__).parent / "visualize_tix_results.py"),
            "--tix-results-dir", str(tix_results_dir),
            "--output-dir", str(Path(config_args.output_dir) / "tix_visualizations"),
            "--sp500-data", str(Path(config_args.processed_dir) / "index_GSPC_processed.csv"),
            "--visualize-all"
        ]
        
        logger.info(f"Running TIX visualization command: {' '.join(vis_cmd)}")
        
        # Execute visualization command
        vis_result = subprocess.run(vis_cmd, check=True)
        
        logger.info("TIX Analysis Workflow completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error in TIX Analysis Workflow: {e}")
        return False
    
def main():
    """
    Main function to visualize TIX results.
    """
    parser = argparse.ArgumentParser(description="Visualize TIX analysis results")
    
    # General arguments
    parser.add_argument(
        "--tix-results-dir", 
        type=str, 
        default=str(config.DATA_DIR / "tix_results"),
        help="Directory containing TIX results"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=str(config.DATA_DIR / "tix_visualizations"),
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--sp500-data", 
        type=str, 
        default=str(config.PROCESSED_DATA_DIR / "index_GSPC_processed.csv"),
        help="Path to S&P 500 processed data"
    )
    
    # Visualization type selection
    parser.add_argument(
        "--visualize-subsequence", 
        action="store_true",
        help="Visualize subsequence TIX results"
    )
    parser.add_argument(
        "--visualize-constituent", 
        action="store_true",
        help="Visualize constituent TIX results"
    )
    parser.add_argument(
        "--visualize-multi-ts", 
        action="store_true",
        help="Visualize multi-TS TIX results"
    )
    parser.add_argument(
        "--visualize-all", 
        action="store_true",
        help="Visualize all TIX results"
    )
    
    # Subsequence visualization arguments
    parser.add_argument(
        "--window-size", 
        type=int, 
        default=3,
        help="Window size for subsequence visualization"
    )
    parser.add_argument(
        "--overlap-type", 
        choices=["overlap", "nonoverlap"],
        default="overlap",
        help="Overlap type for subsequence visualization"
    )
    parser.add_argument(
        "--algorithm", 
        type=str, 
        default="aida",
        help="Algorithm for subsequence visualization"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    ensure_directory_exists(args.output_dir)
    
    # Determine visualizations to create
    visualizations_to_create = []
    
    if args.visualize_all:
        visualizations_to_create = ["subsequence", "constituent", "multi_ts"]
    else:
        if args.visualize_subsequence:
            visualizations_to_create.append("subsequence")
        if args.visualize_constituent:
            visualizations_to_create.append("constituent")
        if args.visualize_multi_ts:
            visualizations_to_create.append("multi_ts")
    
    if not visualizations_to_create:
        logger.warning("No visualizations selected. Use --visualize-all or specific visualization flags.")
        return
    
    # Run selected visualizations
    if "subsequence" in visualizations_to_create:
        logger.info("= Creating Subsequence TIX Visualizations =")
        
        # Load price data
        price_data = load_price_data(args.sp500_data)
        
        if price_data is not None:
            # Load anomalies and TIX results
            anomalies, tix_results = load_anomalies_and_tix(
                Path(args.tix_results_dir),
                Path(config.DATA_DIR) / "subsequence_results",
                args.algorithm,
                args.window_size,
                args.overlap_type
            )
            
            if anomalies is not None:
                # Create visualizations
                subsequence_output_dir = Path(args.output_dir) / "subsequence"
                ensure_directory_exists(subsequence_output_dir)
                
                visualize_anomalies_with_explanations(
                    price_data,
                    anomalies,
                    tix_results,
                    subsequence_output_dir,
                    args.algorithm,
                    args.window_size,
                    args.overlap_type
                )
    
    if "constituent" in visualizations_to_create:
        logger.info("= Creating Constituent TIX Visualizations =")
        
        constituent_output_dir = Path(args.output_dir) / "constituent"
        ensure_directory_exists(constituent_output_dir)
        
        visualize_constituent_tix_results(
            Path(args.tix_results_dir),
            Path(config.DATA_DIR) / "constituent_results",
            config.PROCESSED_DATA_DIR,
            constituent_output_dir
        )
    
    if "multi_ts" in visualizations_to_create:
        logger.info("= Creating Multi-TS TIX Visualizations =")
        
        multi_ts_output_dir = Path(args.output_dir) / "multi_ts"
        ensure_directory_exists(multi_ts_output_dir)
        
        visualize_multi_ts_tix_results(
            Path(args.tix_results_dir),
            Path(config.DATA_DIR) / "multi_ts_results",
            args.sp500_data,
            multi_ts_output_dir
        )
    
    logger.info(f"All visualizations completed and saved to {args.output_dir}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Script to run anomaly detection on individual constituent stocks and cross-reference with S&P 500 anomalies.
This script:
1. Detects anomalies in S&P 500 index data using subsequence analysis
2. For each S&P 500 anomaly, analyzes the same time period in constituent stocks
3. Cross-references to identify which constituents show anomalies during S&P 500 anomalies
"""
import os
import sys
import logging
import argparse
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import concurrent.futures

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.isolation_forest import IForest
from src.models.lof import LOF
from src import config
from src.data.preparation import load_ticker_data, create_subsequence_dataset
from src.utils.helpers import ensure_directory_exists, get_file_list
from src.models.aida_helper import run_aida_on_constituent


# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Custom JSON encoder to handle datetime and pandas Timestamp objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        # Add handling for numpy integer types
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # Add handling for numpy boolean type
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(DateTimeEncoder, self).default(obj)


def load_sp500_anomalies(results_dir, algorithm, window_size, overlap):
    """
    Load anomalies detected in the S&P 500 index.
    
    Args:
        results_dir (Path): Directory containing results
        algorithm (str): Algorithm name ('aida', 'iforest', or 'lof')
        window_size (int): Window size used
        overlap (bool): Whether overlapping subsequences were used
        
    Returns:
        pandas.DataFrame: DataFrame of S&P 500 anomalies
    """
    try:
        overlap_str = "overlap" if overlap else "nonoverlap"
        anomalies_file = results_dir / algorithm / f"w{window_size}_{overlap_str}" / f"{algorithm}_anomalies.csv"
        
        if not anomalies_file.exists():
            logger.error(f"No anomalies file found at {anomalies_file}")
            return None
        
        anomalies_df = pd.read_csv(anomalies_file)
        
        # Ensure start_date and end_date are datetime
        for date_col in ['start_date', 'end_date']:
            if date_col in anomalies_df.columns:
                anomalies_df[date_col] = pd.to_datetime(anomalies_df[date_col])
        
        logger.info(f"Loaded {len(anomalies_df)} S&P 500 anomalies for {algorithm}")
        return anomalies_df
        
    except Exception as e:
        logger.error(f"Error loading S&P 500 anomalies: {e}")
        return None


def load_constituent_data(constituent_dir):
    """
    Load processed constituent stock data.
    
    Args:
        constituent_dir (Path): Directory containing processed constituent data
        
    Returns:
        dict: Dictionary mapping ticker symbols to DataFrames
    """
    try:
        # Get list of processed constituent files
        constituent_files = get_file_list(constituent_dir, "*_processed.csv")
        
        if not constituent_files:
            logger.error(f"No processed constituent files found in {constituent_dir}")
            return None
        
        logger.info(f"Found {len(constituent_files)} processed constituent files")
        
        # Load each constituent's data
        constituent_data = {}
        
        for file_path in constituent_files:
            # Extract ticker from filename
            ticker = file_path.stem.replace("_processed", "")
            
            # Load data
            df = load_ticker_data(file_path)
            
            if df is not None and not df.empty:
                constituent_data[ticker] = df
        
        logger.info(f"Loaded data for {len(constituent_data)} constituents")
        return constituent_data
        
    except Exception as e:
        logger.error(f"Error loading constituent data: {e}")
        return None



def detect_anomalies_in_constituent(ticker, subsequences, algorithm):
    try:
        if algorithm.lower() == 'aida':
            # Convert subsequences to proper format for AIDA
            feature_data = pd.concat(subsequences)
            
            # Create temporary directory for this analysis
            temp_dir = Path(config.DATA_DIR) / "temp" / "constituent_analysis" / ticker
            ensure_directory_exists(temp_dir)
            
            # Run AIDA using the existing implementation
            result = run_aida_on_constituent(ticker, feature_data, temp_dir)
            
            if not result or not isinstance(result, dict):
                logger.error(f"Invalid result format from AIDA for {ticker}")
                return ticker, 0, []
                
            if not result.get('success', False):
                logger.error(f"AIDA analysis failed for {ticker}: {result.get('error', 'Unknown error')}")
                return ticker, 0, []
            
            # Use the output_dir from the AIDA result
            output_dir = result['output_dir']
            anomalies_file = output_dir / "aida_anomalies.csv"
            scores_file = output_dir / "aida_scores.dat"
            
            if not anomalies_file.exists():
                logger.error(f"Anomalies file not found for {ticker}: {anomalies_file}")
                return ticker, 0, []
            
            try:
                anomalies_df = pd.read_csv(anomalies_file)
                anomaly_count = len(anomalies_df)
                
                # Read scores if they exist
                scores = []
                if scores_file.exists():
                    with open(scores_file, 'r') as f:
                        next(f)  # Skip header
                        scores = [float(line.strip()) for line in f]
                
                logger.info(f"Successfully processed AIDA results for {ticker}: {anomaly_count} anomalies found")
                return ticker, anomaly_count, scores
                
            except Exception as e:
                logger.error(f"Error processing AIDA results for {ticker}: {e}")
                return ticker, 0, []

                
        # Extract features from subsequences
        feature_vectors = []
        for subseq in subsequences:
            feature_vector = subseq.values.flatten()
            feature_vectors.append(feature_vector)
        
        # Convert to numpy array
        feature_array = np.array(feature_vectors)
        
        if feature_array.size == 0:
            logger.warning(f"No features extracted for {ticker}")
            return ticker, 0, []
        
        # Detect anomalies using specified algorithm
        if algorithm.lower() == 'iforest':
            n_samples = feature_array.shape[0]
            if n_samples < 2:  # Need at least 2 samples for IForest
                return ticker, 0, [0] * n_samples
                
            model = IForest(
                n_estimators=100,
                max_samples=min(256, n_samples),
                contamination=0.05
            )
        elif algorithm.lower() == 'lof':
            n_samples = feature_array.shape[0]
            if n_samples < 2:  # Need at least 2 samples for LOF
                return ticker, 0, [0] * n_samples
                
            n_neighbors = max(1, min(20, n_samples - 1))  # Ensure at least 1 neighbor
            model = LOF(
                n_neighbors=n_neighbors,
                p=2,
                contamination=0.05
            )
        else:
            logger.error(f"Unknown algorithm: {algorithm}")
            return ticker, 0, []
        
        # Run the model
        scores, labels = model.fit_predict(feature_array)
        
        # Count anomalies
        anomaly_count = np.sum(labels == -1)
        
        return ticker, anomaly_count, scores.tolist()
        
    except Exception as e:
        logger.error(f"Error detecting anomalies in {ticker}: {str(e)}")
        return ticker, 0, []


def run_constituent_analysis(sp500_anomalies, subsequences, algorithm, output_dir):
    try:
        results = []
        
        # For each SP500 anomaly detected by the algorithm
        for idx, anomaly in sp500_anomalies.iterrows():
            start_date = pd.to_datetime(anomaly['start_date'])
            end_date = pd.to_datetime(anomaly['end_date'])
            
            logger.info(f"Analyzing constituents for anomaly {idx} from {start_date.date()} to {end_date.date()}")
            
            # Initialize results for this anomaly period
            constituent_results = []
            
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = []
                
                # For each constituent, check the same date range
                for ticker, ticker_subsequences in subsequences.items():
                    # Filter subsequences that overlap with the anomaly period
                    period_subsequences = [
                        subseq for subseq in ticker_subsequences 
                        if not (
                            (subseq.index[-1] < start_date) or  # Ends before anomaly starts
                            (subseq.index[0] > end_date)        # Starts after anomaly ends
                        )
                    ]
                    
                    if not period_subsequences:
                        logger.debug(f"No subsequences found for {ticker} in period {start_date} to {end_date}")
                        continue
                    
                    # Submit analysis task
                    future = executor.submit(
                        detect_anomalies_in_constituent,
                        ticker,
                        period_subsequences,
                        algorithm
                    )
                    futures.append((ticker, future))
                
                # Process results
                for ticker, future in futures:
                    try:
                        ticker, anomaly_count, scores = future.result()
                        constituent_results.append({
                            'ticker': ticker,
                            'anomaly_count': anomaly_count,
                            'has_anomaly': anomaly_count > 0
                        })
                    except Exception as e:
                        logger.error(f"Error processing results for {ticker}: {e}")
            
            # Add results for this anomaly period
            anomaly_result = {
                'sp500_anomaly_index': idx,
                'start_date': start_date,
                'end_date': end_date,
                'constituent_anomalies': {r['ticker']: {
                    'anomaly_count': r['anomaly_count'],
                    'has_anomaly': r['has_anomaly']
                } for r in constituent_results},
                'summary': {
                    'total_constituents': len(constituent_results),
                    'anomalous_constituents': sum(1 for r in constituent_results if r['has_anomaly']),
                }
            }
            anomaly_result['summary']['anomaly_percentage'] = (
                (anomaly_result['summary']['anomalous_constituents'] / 
                anomaly_result['summary']['total_constituents']) * 100 
                if anomaly_result['summary']['total_constituents'] > 0 else 0
            )
            
            results.append(anomaly_result)
            
        return results
        
    except Exception as e:
        logger.error(f"Error running constituent analysis: {e}")
        return []

def load_precomputed_subsequences(subsequence_dir, window_size, overlap):
    """
    Load precomputed subsequences for constituents.
    
    Args:
        subsequence_dir (Path): Directory containing precomputed subsequences
        window_size (int): Size of subsequence window
        overlap (bool): Whether overlapping subsequences were used
        
    Returns:
        dict: Dictionary mapping tickers to lists of subsequence DataFrames
    """
    try:
        overlap_str = "overlap" if overlap else "nonoverlap"
        subsequence_files = get_file_list(subsequence_dir, f"*len{window_size}_{overlap_str}_*.csv")

        if not subsequence_files:
            logger.error(f"No precomputed subsequences found in {subsequence_dir}")
            return None

        subsequences = {}
        for file_path in subsequence_files:
            ticker = file_path.stem.split("_len")[0]
            
            # Load the CSV with the first column as index
            df = pd.read_csv(file_path, index_col=0)
            
            # Convert index to datetime if it contains dates
            if df.index.dtype == 'object':
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    pass
            
            # Initialize list for this ticker if it doesn't exist
            if ticker not in subsequences:
                subsequences[ticker] = []
            
            # Reshape the data into window_size chunks if necessary
            n_rows = len(df)
            n_chunks = n_rows // window_size
            
            for i in range(n_chunks):
                start_idx = i * window_size
                end_idx = start_idx + window_size
                chunk = df.iloc[start_idx:end_idx].copy()
                subsequences[ticker].append(chunk)

        logger.info(f"Loaded subsequences for {len(subsequences)} constituents")
        return subsequences

    except Exception as e:
        logger.error(f"Error loading precomputed subsequences: {e}")
        return None
    

def create_visualizations(results, output_dir, algorithm):
    """
    Create visualizations of constituent analysis results.
    
    Args:
        results (list): List of analysis results
        output_dir (Path): Directory to save visualizations
        algorithm (str): Algorithm used
    """
    try:
        # Ensure output directory exists
        ensure_directory_exists(output_dir)
        
        # 1. Histogram of anomaly percentages
        percentages = [result['summary']['anomaly_percentage'] for result in results]
        
        plt.figure(figsize=(10, 6))
        plt.hist(percentages, bins=10, alpha=0.7, color='blue')
        plt.axvline(np.mean(percentages), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(percentages):.1f}%')
        plt.xlabel('Percentage of Constituents with Anomalies')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Constituent Anomaly Percentages ({algorithm.upper()})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / f"{algorithm}_anomaly_percentage_histogram.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Bar chart showing constituent anomaly counts
        if results:
            # Count how many times each constituent had an anomaly
            constituent_anomaly_counts = {}
            
            for result in results:
                for ticker, data in result['constituent_anomalies'].items():
                    if data['has_anomaly']:
                        constituent_anomaly_counts[ticker] = constituent_anomaly_counts.get(ticker, 0) + 1
            
            # Sort by count
            sorted_constituents = sorted(constituent_anomaly_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Take top 20
            top_constituents = sorted_constituents[:20]
            
            # Create bar chart
            plt.figure(figsize=(12, 8))
            tickers = [item[0] for item in top_constituents]
            counts = [item[1] for item in top_constituents]
            
            plt.bar(tickers, counts, color='green', alpha=0.7)
            plt.xlabel('Constituent Ticker')
            plt.ylabel('Number of Anomalies')
            plt.title(f'Top 20 Constituents by Anomaly Count ({algorithm.upper()})')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(output_dir / f"{algorithm}_top_constituents_bar_chart.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Timeline of S&P 500 anomalies with constituent anomaly percentages
        if results:
            # Sort results by date - ensure we're working with datetime objects for sorting
            for result in results:
                if isinstance(result['start_date'], str):
                    result['start_date'] = pd.to_datetime(result['start_date'])
            
            sorted_results = sorted(results, key=lambda x: x['start_date'])
            
            dates = [result['start_date'] for result in sorted_results]
            percentages = [result['summary']['anomaly_percentage'] for result in sorted_results]
            
            plt.figure(figsize=(14, 8))
            plt.plot(dates, percentages, marker='o', linestyle='-', color='purple', alpha=0.7)
            plt.xlabel('S&P 500 Anomaly Date')
            plt.ylabel('Percentage of Constituents with Anomalies')
            plt.title(f'Constituent Anomaly Percentages Over Time ({algorithm.upper()})')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / f"{algorithm}_anomaly_timeline.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Created visualizations in {output_dir}")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")


def main():
    """
    Main function to run constituent analysis for S&P 500 anomalies.
    """

    parser = argparse.ArgumentParser(description="Run constituent analysis for S&P 500 anomalies")
    parser.add_argument(
        "--subsequence-results", 
        type=str, 
        default=str(config.DATA_DIR / "subsequence_results"),
        help="Directory containing S&P 500 subsequence analysis results"
    )
    parser.add_argument(
        "--constituent-dir", 
        type=str, 
        default=str(config.PROCESSED_DATA_DIR),
        help="Directory containing processed constituent data"
    )
    parser.add_argument(
        "--window-size", 
        type=int, 
        default=3,
        help="Size of subsequence window (e.g., 3, 5, 10)"
    )
    parser.add_argument(
        "--overlap", 
        action="store_true",
        help="Use overlapping subsequences"
    )
    parser.add_argument(
        "--no-overlap", 
        action="store_true",
        help="Use non-overlapping subsequences"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=str(config.DATA_DIR / "constituent_analysis"),
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["aida", "iforest", "lof", "all"],
        default=["all"],
        help="Algorithms to run"
    )
    
    args = parser.parse_args()
    
    # Determine which algorithms to run
    algorithms = args.algorithms
    if "all" in algorithms:
        algorithms = ["aida", "iforest", "lof"]
    
    # Determine overlap setting
    overlap = True
    if args.no_overlap:
        overlap = False
    elif args.overlap:
        overlap = True
    
    # Convert paths to Path objects
    subsequence_results_dir = Path(args.subsequence_results)
    constituent_dir = Path(args.constituent_dir)
    output_dir = Path(args.output)
    
    # Create output directory
    ensure_directory_exists(output_dir)
    
    # Create a specific output dir for this configuration
    config_dir = output_dir / f"w{args.window_size}_{'overlap' if overlap else 'nonoverlap'}"
    ensure_directory_exists(config_dir)
    
    # Load constituent data
    #constituent_data = load_constituent_data(constituent_dir)
    
    #if constituent_data is None or not constituent_data:
    #    logger.error("Failed to load constituent data. Exiting.")
    #    return
    
    # Load precomputed subsequences instead of creating them
    subsequences = load_precomputed_subsequences(
        subsequence_dir=Path(config.PROCESSED_DATA_DIR) / "subsequences",
        window_size=args.window_size,
        overlap=overlap
    )

    if subsequences is None:
        logger.error("Failed to load precomputed subsequences. Exiting.")
        return
    
    # Process each algorithm
    for algorithm in algorithms:
        logger.info(f"Processing {algorithm} algorithm")
        
        # Load S&P 500 anomalies
        sp500_anomalies = load_sp500_anomalies(
            subsequence_results_dir, 
            algorithm, 
            args.window_size, 
            overlap
        )
        
        if sp500_anomalies is None or sp500_anomalies.empty:
            logger.warning(f"No S&P 500 anomalies found for {algorithm}. Skipping.")
            continue
        
        # Create algorithm-specific output directory
        algo_dir = config_dir / algorithm
        ensure_directory_exists(algo_dir)
        
        # Run constituent analysis with precomputed subsequences
        start_time = time.time()
        
        results = run_constituent_analysis(
            sp500_anomalies,
            subsequences,  # Pass subsequences instead of constituent_data
            algorithm,
            algo_dir
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Save execution time
        with open(algo_dir / "execution_time.txt", "w") as f:
            f.write(f"{execution_time:.2f}")
        
        # Create visualizations
        if results:
            create_visualizations(results, algo_dir, algorithm)
        
        logger.info(f"Completed {algorithm} analysis in {execution_time:.2f} seconds")
    
    logger.info(f"All constituent analyses completed. Results saved to {config_dir}")


if __name__ == "__main__":
    main()
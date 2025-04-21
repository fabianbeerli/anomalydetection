"""
Multi-TS Subsequence Analysis for anomaly detection.
This module detects anomalies in matrices of multiple stock time series.
"""
import os
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pathlib import Path
import subprocess
import tempfile
import json
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_multi_ts_data(multi_ts_dir, prefix='multi_ts_len5_overlap'):
    """
    Load multi-dimensional time series data for anomaly detection.
    
    Args:
        multi_ts_dir (Path): Directory containing multi-TS files
        prefix (str): Prefix for multi-TS files
        
    Returns:
        tuple: (multi_ts_data, metadata_list)
            - multi_ts_data: List of 3D matrices [n_stocks, seq_length, n_features]
            - metadata_list: List of metadata dictionaries for each matrix
    """
    try:
        logger.info(f"Loading multi-TS data from {multi_ts_dir} with prefix {prefix}")
        
        # Get list of multi-TS files
        multi_ts_files = list(Path(multi_ts_dir).glob(f"{prefix}*.npz"))
        
        if not multi_ts_files:
            logger.error(f"No multi-TS files found with prefix {prefix}")
            return None, None
        
        logger.info(f"Found {len(multi_ts_files)} multi-TS files")
        
        # Load matrices and metadata
        multi_ts_data = []
        metadata_list = []
        
        for file_path in sorted(multi_ts_files):
            try:
                # Load NPZ file
                npz_data = np.load(file_path, allow_pickle=True)
                
                # Extract matrix and metadata
                matrix = npz_data['matrix']
                metadata = json.loads(npz_data['metadata'].item())
                
                # Add to lists
                multi_ts_data.append(matrix)
                metadata_list.append(metadata)
                
            except Exception as e:
                logger.error(f"Error loading multi-TS file {file_path}: {e}")
                continue
        
        logger.info(f"Loaded {len(multi_ts_data)} multi-TS matrices")
        
        if not multi_ts_data:
            return None, None
            
        # Check the structure
        first_matrix = multi_ts_data[0]
        logger.info(f"Matrix shape: {first_matrix.shape} - [n_stocks, seq_length, n_features]")
        
        return multi_ts_data, metadata_list
        
    except Exception as e:
        logger.error(f"Error loading multi-TS data: {e}")
        return None, None


def flatten_multi_ts_matrices(multi_ts_data):
    """
    Flatten multi-TS 3D matrices to 2D feature vectors for anomaly detection.
    Each 3D matrix [n_stocks, seq_length, n_features] becomes a single flattened vector.
    
    Args:
        multi_ts_data (list): List of 3D matrices
        
    Returns:
        numpy.ndarray: 2D array where each row is a flattened matrix
    """
    try:
        flattened_data = []
        
        for matrix in multi_ts_data:
            # Flatten the entire 3D matrix into a 1D vector
            flattened = matrix.flatten()
            flattened_data.append(flattened)
        
        # Convert to numpy array
        feature_array = np.array(flattened_data)
        
        logger.info(f"Flattened multi-TS data to shape {feature_array.shape}")
        return feature_array
        
    except Exception as e:
        logger.error(f"Error flattening multi-TS data: {e}")
        return None


def detect_anomalies_iforest(feature_array, metadata_list, contamination=0.05):
    """
    Run Isolation Forest on multi-TS data and return anomaly scores and periods.
    """
    from sklearn.ensemble import IsolationForest
    import numpy as np

    model = IsolationForest(contamination=contamination, random_state=42)
    labels = model.fit_predict(feature_array)
    scores = -model.decision_function(feature_array)

    anomaly_periods = []
    for i, (score, label, metadata) in enumerate(zip(scores, labels, metadata_list)):
        if label == -1:
            # Calculate per-ticker contributions for this window
            # Assume feature_array[i] is a flattened matrix: [stock1_feat1, stock1_feat2, ..., stockN_featM]
            # You need to know n_stocks and n_features to reshape
            n_stocks = len(metadata['tickers'])
            n_features = int(len(feature_array[i]) / n_stocks)
            matrix = feature_array[i].reshape((n_stocks, n_features))
            # Example: use sum of absolute values as "contribution"
            contributions = np.sum(np.abs(matrix), axis=1)
            top_indices = np.argsort(contributions)[-10:][::-1]
            top_tickers = [metadata['tickers'][j] for j in top_indices]

            anomaly_periods.append({
                'time_period_idx': i,
                'score': float(score),
                'start_date': metadata['start_date'],
                'end_date': metadata['end_date'],
                'tickers': top_tickers
            })
    return scores, np.where(labels == -1)[0], anomaly_periods, None  # add timing if needed



def detect_anomalies_lof(feature_array, metadata_list, contamination=0.05):
    """
    Detect anomalies in flattened multi-TS data using Local Outlier Factor.
    Treats each time period as a potential anomaly.
    """
    from sklearn.neighbors import LocalOutlierFactor
    import numpy as np

    lof = LocalOutlierFactor(
        n_neighbors=min(20, feature_array.shape[0] - 1),
        contamination=contamination,
        n_jobs=-1,
        novelty=False
    )
    labels = lof.fit_predict(feature_array)
    scores = -lof.negative_outlier_factor_

    anomaly_periods = []
    for i, (score, label, metadata) in enumerate(zip(scores, labels, metadata_list)):
        if label == -1:
            n_stocks = len(metadata['tickers'])
            n_features = int(len(feature_array[i]) / n_stocks)
            matrix = feature_array[i].reshape((n_stocks, n_features))
            contributions = np.sum(np.abs(matrix), axis=1)
            top_indices = np.argsort(contributions)[-10:][::-1]
            top_tickers = [metadata['tickers'][j] for j in top_indices]

            anomaly_periods.append({
                'time_period_idx': i,
                'score': float(score),
                'start_date': metadata['start_date'],
                'end_date': metadata['end_date'],
                'tickers': top_tickers
            })
    return scores, np.where(labels == -1)[0], anomaly_periods, None

def detect_anomalies_aida(feature_array, metadata_list, multi_ts_dir, contamination=0.05):
    """
    Detect anomalies in flattened multi-TS data using AIDA.
    Treats each time period as a potential anomaly.
    """
    try:
        logger.info(f"Running AIDA on {len(feature_array)} multi-TS matrices")
        
        # Create temporary input file for AIDA
        temp_dir = Path(multi_ts_dir) / "temp_aida"
        temp_dir.mkdir(exist_ok=True, parents=True)
        temp_input_file = temp_dir / "multi_ts_features.csv"
        
        # Save feature array to CSV for AIDA
        with open(temp_input_file, 'w') as f:
            f.write(','.join([f'feature_{i}' for i in range(feature_array.shape[1])]) + '\n')
            for i in range(feature_array.shape[0]):
                f.write(','.join([str(val) for val in feature_array[i]]) + '\n')
        
        # Set up paths for AIDA
        root_dir = Path(multi_ts_dir).parent.parent.parent  # Adjust based on your project structure
        aida_cpp_dir = root_dir / "AIDA" / "C++"
        aida_executable = aida_cpp_dir / "build" / "aida_subsequence_detection"
        
        # Execute AIDA
        start_time = time.time()
        cmd = [str(aida_executable), str(temp_input_file)]
        subprocess.run(cmd, check=True)
        
        # Read results
        scores_file = Path(str(temp_input_file) + "_AIDA_scores.dat")
        anomalies_file = Path(str(temp_input_file) + "_AIDA_anomalies.csv")
        
        # Read scores
        scores = None
        if scores_file.exists():
            with open(scores_file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    scores = np.array([float(line.strip()) for line in lines[1:]])
        
        # Read anomalies
        anomaly_indices = []
        if anomalies_file.exists():
            anomalies_df = pd.read_csv(anomalies_file)
            if 'index' in anomalies_df.columns:
                anomaly_indices = anomalies_df['index'].values
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Get default scores if AIDA failed
        if scores is None:
            logger.warning("AIDA scores not found, using default values")
            scores = np.zeros(len(feature_array))
            anomaly_indices = []
        
        # Extract anomaly information with dynamic top tickers
        anomaly_periods = []
        for idx in anomaly_indices:
            if idx < len(metadata_list):
                metadata = metadata_list[idx]
                n_stocks = len(metadata['tickers'])
                n_features = int(len(feature_array[idx]) / n_stocks)
                matrix = feature_array[idx].reshape((n_stocks, n_features))
                contributions = np.sum(np.abs(matrix), axis=1)
                top_indices = np.argsort(contributions)[-10:][::-1]
                top_tickers = [metadata['tickers'][j] for j in top_indices]
                period_info = {
                    'time_period_idx': idx,
                    'score': float(scores[idx]) if idx < len(scores) else 0,
                    'start_date': metadata.get('start_date', 'Unknown'),
                    'end_date': metadata.get('end_date', 'Unknown'),
                    'tickers': top_tickers,
                }
                anomaly_periods.append(period_info)
        
        logger.info(f"AIDA execution time: {execution_time:.2f} seconds")
        
        return scores, anomaly_indices, anomaly_periods, execution_time
        
    except Exception as e:
        logger.error(f"Error detecting anomalies with AIDA: {e}")
        return None, None, None, -1


def save_multi_ts_results(algorithm, scores, anomaly_periods, execution_time, output_dir):
    """
    Save multi-TS anomaly detection results.
    
    Args:
        algorithm (str): Algorithm name ('aida', 'iforest', 'lof')
        scores (numpy.ndarray): Anomaly scores
        anomaly_periods (list): Information about anomalous periods
        execution_time (float): Execution time in seconds
        output_dir (Path): Directory to save results
        
    Returns:
        dict: Paths to saved files
    """
    try:
        # Create output directory
        algo_dir = Path(output_dir) / algorithm
        algo_dir.mkdir(exist_ok=True, parents=True)
        
        # Save scores
        scores_file = algo_dir / f"{algorithm}_multi_ts_scores.dat"
        with open(scores_file, 'w') as f:
            f.write(f"{len(scores)}\n")
            for score in scores:
                f.write(f"{score}\n")
        
        # Save anomalies
        anomalies_file = algo_dir / f"{algorithm}_multi_ts_anomalies.csv"
        if anomaly_periods:
            # Prepare data for CSV
            anomaly_data = []
            for i, period in enumerate(anomaly_periods):
                anomaly_data.append({
                    'index': i,
                    'time_period_idx': period['time_period_idx'],
                    'score': period['score'],
                    'start_date': period['start_date'],
                    'end_date': period['end_date'],
                    'num_tickers': len(period.get('tickers', [])),
                    'top_tickers': ','.join(period.get('tickers', [])[:5]) if period.get('tickers') else 'Unknown'
                })
            
            # Save as CSV
            pd.DataFrame(anomaly_data).to_csv(anomalies_file, index=False)
        else:
            # Create empty file
            with open(anomalies_file, 'w') as f:
                f.write("index,time_period_idx,score,start_date,end_date,num_tickers,top_tickers\n")
        
        # Save execution time
        time_file = algo_dir / f"{algorithm}_multi_ts_execution_time.txt"
        # Save execution time
        time_file = algo_dir / f"{algorithm}_multi_ts_execution_time.txt"
        with open(time_file, 'w') as f:
            if execution_time is not None:
                f.write(f"{execution_time:.6f}")
            else:
                f.write("NA")
        
        logger.info(f"Saved {algorithm} multi-TS results to {algo_dir}")
        
        return {
            'scores': str(scores_file),
            'anomalies': str(anomalies_file),
            'time': str(time_file)
        }
        
    except Exception as e:
        logger.error(f"Error saving {algorithm} multi-TS results: {e}")
        return {}


def run_multi_ts_analysis(multi_ts_dir, output_dir, window_size=5, overlap=True, algorithms=None):
    """
    Run multi-TS anomaly detection analysis.
    
    Args:
        multi_ts_dir (Path): Directory containing multi-TS data
        output_dir (Path): Directory to save results
        window_size (int): Window size for multi-TS analysis
        overlap (bool): Whether subsequences are overlapping
        algorithms (list): List of algorithms to run (default: all)
        
    Returns:
        dict: Results summary
    """
    if algorithms is None:
        algorithms = ['aida', 'iforest', 'lof']  # Added AIDA to default algorithms
    
    # Create output directory
    overlap_str = "overlap" if overlap else "nonoverlap"
    config_dir = f"multi_ts_w{window_size}_{overlap_str}"
    results_dir = Path(output_dir) / config_dir
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Load multi-TS data
    prefix = f"multi_ts_len{window_size}_{'overlap' if overlap else 'nonoverlap'}"
    multi_ts_data, metadata_list = load_multi_ts_data(multi_ts_dir, prefix)
    
    if multi_ts_data is None or not multi_ts_data:
        logger.error(f"Failed to load multi-TS data. Exiting.")
        return {'status': 'error', 'message': 'Failed to load multi-TS data'}
    
    # Flatten multi-TS matrices for anomaly detection
    feature_array = flatten_multi_ts_matrices(multi_ts_data)
    
    if feature_array is None:
        logger.error(f"Failed to flatten multi-TS data. Exiting.")
        return {'status': 'error', 'message': 'Failed to flatten multi-TS data'}
    
    # Initialize results
    results = {'status': 'success', 'algorithms': {}}
    
    # Run each algorithm
    if 'iforest' in algorithms:
        scores, indices, periods, time_taken = detect_anomalies_iforest(feature_array, metadata_list)
        if scores is not None:
            files = save_multi_ts_results('iforest', scores, periods, time_taken, results_dir)
            results['algorithms']['iforest'] = {
                'success': True,
                'anomaly_count': len(indices) if indices is not None else 0,
                'execution_time': time_taken,
                'files': files
            }
    
    if 'lof' in algorithms:
        scores, indices, periods, time_taken = detect_anomalies_lof(feature_array, metadata_list)
        if scores is not None:
            files = save_multi_ts_results('lof', scores, periods, time_taken, results_dir)
            results['algorithms']['lof'] = {
                'success': True,
                'anomaly_count': len(indices) if indices is not None else 0,
                'execution_time': time_taken,
                'files': files
            }
    
    if 'aida' in algorithms:
        scores, indices, periods, time_taken = detect_anomalies_aida(feature_array, metadata_list, multi_ts_dir)
        if scores is not None:
            files = save_multi_ts_results('aida', scores, periods, time_taken, results_dir)
            results['algorithms']['aida'] = {
                'success': True,
                'anomaly_count': len(indices) if indices is not None else 0,
                'execution_time': time_taken,
                'files': files
            }
    
    # Create a summary of results
    summary_file = results_dir / "multi_ts_analysis_summary.json"
    
    # Convert Path objects to strings for JSON serialization
    json_results = {}
    json_results['status'] = results['status']
    json_results['algorithms'] = {}
    
    for algo, algo_data in results.get('algorithms', {}).items():
        json_results['algorithms'][algo] = {
            'success': algo_data.get('success', False),
            'anomaly_count': algo_data.get('anomaly_count', 0),
            'execution_time': algo_data.get('execution_time', -1),
            'files': {k: str(v) for k, v in algo_data.get('files', {}).items()}
        }
    
    with open(summary_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Multi-TS analysis complete. Results saved to {results_dir}")
    
    return results
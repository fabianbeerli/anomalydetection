#!/usr/bin/env python
"""
Script to run anomaly detection on matrix of S&P 500 and constituent data.
This script processes multi-TS subsequences containing both S&P 500 and constituent data.
"""
import os
import sys
import logging
import argparse
import subprocess
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.isolation_forest import IForest
from src.models.lof import LOF
from src import config
from src.utils.helpers import load_subsequence, get_file_list, ensure_directory_exists

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_multi_ts_data(multi_ts_dir, window_size=3, overlap=True):
    """
    Load multi-TS data containing both S&P 500 and constituent stocks.
    
    Args:
        multi_ts_dir (Path): Directory containing multi-TS files
        window_size (int): Size of the subsequence window
        overlap (bool): Whether to use overlapping or non-overlapping subsequences
        
    Returns:
        tuple: (data_matrices, metadata_list)
            data_matrices: List of numpy arrays with shape (n_stocks, window_size, n_features)
            metadata_list: List of metadata dictionaries
    """
    try:
        # Determine prefix based on parameters
        overlap_str = "overlap" if overlap else "nonoverlap"
        prefix = f"multi_ts_len{window_size}_{overlap_str}"
        
        logger.info(f"Loading multi-TS data with prefix: {prefix}")
        
        # Get list of multi-TS files
        ts_files = get_file_list(multi_ts_dir, f"{prefix}*.npz")
        
        if not ts_files:
            logger.error(f"No multi-TS files found with prefix {prefix} in {multi_ts_dir}")
            return None, None
        
        logger.info(f"Found {len(ts_files)} multi-TS files")
        
        # Load data matrices and metadata
        data_matrices = []
        metadata_list = []
        
        for file_path in sorted(ts_files):
            # Load NPZ file
            data = np.load(file_path, allow_pickle=True)
            
            # Extract matrix and metadata
            matrix = data['matrix']
            try:
                metadata = json.loads(data['metadata'].item())
                metadata_list.append(metadata)
                data_matrices.append(matrix)
            except Exception as e:
                logger.error(f"Error parsing metadata from {file_path}: {e}")
                continue
        
        logger.info(f"Loaded {len(data_matrices)} multi-TS matrices")
        return data_matrices, metadata_list
        
    except Exception as e:
        logger.error(f"Error loading multi-TS data: {e}")
        return None, None


def prepare_matrix_features(data_matrices):
    """
    Prepare matrix features for anomaly detection.
    Transforms 3D matrices into 2D feature vectors.
    
    Args:
        data_matrices (list): List of 3D numpy arrays
            Each array has shape (n_stocks, window_size, n_features)
        
    Returns:
        numpy.ndarray: Array of feature vectors
    """
    try:
        # Convert matrices to feature vectors
        feature_vectors = []
        
        for matrix in data_matrices:
            # Flatten the matrix into a feature vector
            # We preserve the stock dimension but flatten time and features
            n_stocks, window_size, n_features = matrix.shape
            
            # Reshape to (n_stocks, window_size * n_features)
            reshaped = matrix.reshape(n_stocks, -1)
            feature_vectors.append(reshaped)
        
        # Stack along the first dimension to create 
        # shape (n_matrices * n_stocks, window_size * n_features)
        feature_array = np.vstack(feature_vectors)
        
        logger.info(f"Prepared feature array with shape {feature_array.shape}")
        return feature_array
        
    except Exception as e:
        logger.error(f"Error preparing matrix features: {e}")
        return None


def run_aida_on_matrix(feature_array, metadata_list, output_dir, window_size, overlap):
    """
    Run AIDA algorithm on matrix data.
    
    Args:
        feature_array (numpy.ndarray): Array of feature vectors
        metadata_list (list): List of metadata dictionaries
        output_dir (Path): Directory to save results
        window_size (int): Size of the subsequence window
        overlap (bool): Whether subsequences are overlapping
        
    Returns:
        tuple: (success, execution_time, output_files)
            success: Boolean indicating if execution was successful
            execution_time: Model execution time from C++ (in seconds)
            output_files: Dictionary with paths to output files
    """
    # Create subdirectory name based on parameters
    subdir_name = f"w{window_size}_{'overlap' if overlap else 'nonoverlap'}"
    logger.info(f"Running AIDA algorithm on matrix data ({subdir_name})...")
    
    # Ensure output directory exists
    aida_output_dir = output_dir / "aida" / subdir_name
    ensure_directory_exists(aida_output_dir)
    
    # Create temporary input file for AIDA
    temp_input_file = aida_output_dir / "matrix_features.csv"
    
    try:
        # Save feature array to CSV
        with open(temp_input_file, 'w') as f:
            f.write(','.join([f'feature_{i}' for i in range(feature_array.shape[1])]) + '\n')
            for i in range(feature_array.shape[0]):
                f.write(','.join([str(val) for val in feature_array[i]]) + '\n')
        
        logger.info(f"Saved feature array to {temp_input_file}")
        
        # Paths to AIDA executable
        aida_cpp_dir = config.ROOT_DIR / "AIDA" / "C++"
        aida_executable = aida_cpp_dir / "build" / "aida_matrix_detection"
        
        # Check if executable exists, if not attempt to compile
        if not aida_executable.exists():
            logger.info("AIDA matrix executable not found. Attempting to compile...")
            src_models_cpp_dir = config.ROOT_DIR / "src" / "models" / "cpp"
            os.makedirs(src_models_cpp_dir, exist_ok=True)
            analysis_source_file = src_models_cpp_dir / "aida_matrix_detection.cpp"
            
            # Basic C++ code for AIDA matrix detection
            with open(analysis_source_file, "w") as f:
                f.write("""/* AIDA Anomaly Detection for Matrix Data */
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <chrono>
#include "aida_class.h"
using namespace std;
using namespace std::chrono;

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_csv_file>" << endl;
        return 1;
    }
    string input_file = argv[1];
    string output_scores_file = input_file + "_AIDA_scores.dat";
    string output_anomalies_file = input_file + "_AIDA_anomalies.csv";
    ifstream file(input_file);
    if (!file.is_open()) {
        cerr << "Error: Could not open input file " << input_file << endl;
        return 1;
    }
    string header_line;
    getline(file, header_line);
    stringstream header_stream(header_line);
    string feature_name;
    int nFnum = 0;
    while (getline(header_stream, feature_name, ',')) {
        nFnum++;
    }
    vector<vector<double>> numerical_data;
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<double> row;
        while (getline(ss, cell, ',')) {
            try {
                row.push_back(stod(cell));
            } catch (const std::invalid_argument& e) {
                row.push_back(0.0);
            }
        }
        if (!row.empty()) {
            numerical_data.push_back(row);
        }
    }
    file.close();
    int n = numerical_data.size();
    int nFnom = 1;
    cout << "Data loaded: " << n << " rows, " << nFnum << " features per row" << endl;
    double* Xnum = new double[n * nFnum];
    int* Xnom = new int[n * nFnom];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < nFnum; ++j) {
            Xnum[j + i * nFnum] = numerical_data[i][j];
        }
        Xnom[i] = 0;
    }
    int N = 100;
    string aggregate_type = "aom";
    string score_function = "variance";
    double alpha_min = 1.0;
    double alpha_max = 1.0;
    string distance_metric = "euclidean";
    int subsample_min = 50;
    int subsample_max = min(256, n);
    int dmin = min(nFnum, max(2, nFnum / 2));
    int dmax = nFnum;
    double* scoresAIDA = new double[n];
    try {
        cout << "Training AIDA..." << endl;
        auto start_time = high_resolution_clock::now();
        AIDA aida(N, aggregate_type, score_function, alpha_min, alpha_max, distance_metric);
        aida.fit(n, nFnum, Xnum, nFnom, Xnom, subsample_min, subsample_max, dmin, dmax, nFnom, nFnom);
        cout << "Computing anomaly scores..." << endl;
        aida.score_samples(n, scoresAIDA, Xnum, Xnom);
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time);
        ofstream fres(output_scores_file);
        fres << n << endl;
        for (int i = 0; i < n; ++i) {
            fres << scoresAIDA[i] << endl;
        }
        fres.close();
        double mean_score = 0.0;
        double std_score = 0.0;
        for (int i = 0; i < n; ++i) {
            mean_score += scoresAIDA[i];
        }
        mean_score /= n;
        for (int i = 0; i < n; ++i) {
            std_score += (scoresAIDA[i] - mean_score) * (scoresAIDA[i] - mean_score);
        }
        std_score = sqrt(std_score / n);
        double threshold = mean_score + 2 * std_score;
        ofstream fanom(output_anomalies_file);
        int anomaly_count = 0;
        fanom << "index,stock_idx,score" << endl;
        for (int i = 0; i < n; ++i) {
            if (scoresAIDA[i] > threshold) {
                fanom << i << "," << i % 30 << "," << scoresAIDA[i] << endl;
                anomaly_count++;
            }
        }
        fanom.close();
        cout << "AIDA Matrix Analysis Complete:" << endl;
        cout << "Total rows: " << n << endl;
        cout << "Anomalies detected: " << anomaly_count << endl;
        cout << "Model execution time: " << duration.count() / 1000.0 << " seconds" << endl;
        cout << "Scores saved to: " << output_scores_file << endl;
        cout << "Anomalies saved to: " << output_anomalies_file << endl;
        ofstream ftime(input_file + "_AIDA_time.txt");
        ftime << duration.count() / 1000.0 << endl;
        ftime.close();
    }
    catch (const std::exception& e) {
        cerr << "Error during AIDA processing: " << e.what() << endl;
        delete[] Xnum;
        delete[] Xnom;
        delete[] scoresAIDA;
        return 1;
    }
    delete[] Xnum;
    delete[] Xnom;
    delete[] scoresAIDA;
    return 0;
}
""")
            
            logger.info(f"Created source file at {analysis_source_file}")
            
            # Platform-specific compilation
            if sys.platform == 'darwin':  # macOS
                try:
                    subprocess.run(["brew", "--version"], check=True, capture_output=True)
                    try:
                        result = subprocess.run(["brew", "list", "libomp"], check=True, capture_output=True)
                        logger.info("OpenMP found via Homebrew")
                    except subprocess.CalledProcessError:
                        logger.info("Installing OpenMP via Homebrew...")
                        subprocess.run(["brew", "install", "libomp"], check=True)
                    libomp_prefix = subprocess.run(["brew", "--prefix", "libomp"], 
                                                check=True, 
                                                capture_output=True, 
                                                text=True).stdout.strip()
                    compile_cmd = [
                        "g++", "-std=c++11", "-O3", "-Xpreprocessor", "-fopenmp",
                        f"-I{aida_cpp_dir/'include'}",
                        f"-I{libomp_prefix}/include",
                        f"-L{libomp_prefix}/lib",
                        "-lomp",
                        str(analysis_source_file),
                        str(aida_cpp_dir/"src"/"aida_class.cpp"),
                        str(aida_cpp_dir/"src"/"distance_metrics.cpp"),
                        str(aida_cpp_dir/"src"/"isolation_formulas.cpp"),
                        str(aida_cpp_dir/"src"/"aggregation_functions.cpp"),
                        str(aida_cpp_dir/"src"/"rng_class.cpp"),
                        "-o", str(aida_executable)
                    ]
                except (subprocess.CalledProcessError, FileNotFoundError):
                    logger.warning("Homebrew not found, using basic macOS compilation")
                    compile_cmd = [
                        "g++", "-std=c++11", "-O3",
                        f"-I{aida_cpp_dir/'include'}",
                        str(analysis_source_file),
                        str(aida_cpp_dir/"src"/"aida_class.cpp"),
                        str(aida_cpp_dir/"src"/"distance_metrics.cpp"),
                        str(aida_cpp_dir/"src"/"isolation_formulas.cpp"),
                        str(aida_cpp_dir/"src"/"aggregation_functions.cpp"),
                        str(aida_cpp_dir/"src"/"rng_class.cpp"),
                        "-o", str(aida_executable)
                    ]
            elif sys.platform.startswith('win'):  # Windows
                compile_cmd = [
                    "g++", "-std=c++11", "-O3", "-fopenmp",
                    f"-I{aida_cpp_dir/'include'}",
                    str(analysis_source_file),
                    str(aida_cpp_dir/"src"/"aida_class.cpp"),
                    str(aida_cpp_dir/"src"/"distance_metrics.cpp"),
                    str(aida_cpp_dir/"src"/"isolation_formulas.cpp"),
                    str(aida_cpp_dir/"src"/"aggregation_functions.cpp"),
                    str(aida_cpp_dir/"src"/"rng_class.cpp"),
                    "-o", str(aida_executable)
                ]
            else:  # Linux
                compile_cmd = [
                    "g++", "-std=c++11", "-O3", "-fopenmp",
                    f"-I{aida_cpp_dir/'include'}",
                    str(analysis_source_file),
                    str(aida_cpp_dir/"src"/"aida_class.cpp"),
                    str(aida_cpp_dir/"src"/"distance_metrics.cpp"),
                    str(aida_cpp_dir/"src"/"isolation_formulas.cpp"),
                    str(aida_cpp_dir/"src"/"aggregation_functions.cpp"),
                    str(aida_cpp_dir/"src"/"rng_class.cpp"),
                    "-o", str(aida_executable)
                ]
            
            logger.info(f"Compilation command: {' '.join(compile_cmd)}")
            result = subprocess.run(compile_cmd, check=True)
            logger.info("AIDA compilation successful")
        
        # Execute AIDA
        logger.info(f"Running AIDA on {temp_input_file}")
        env = os.environ.copy()
        if sys.platform == 'darwin':
            try:
                result = subprocess.run(["brew", "--prefix", "libomp"], 
                                     check=True, 
                                     capture_output=True, 
                                     text=True)
                libomp_prefix = result.stdout.strip()
                env["DYLD_LIBRARY_PATH"] = f"{libomp_prefix}/lib"
                logger.info(f"Set DYLD_LIBRARY_PATH to {libomp_prefix}/lib for OpenMP support")
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("Could not set OpenMP library path")
        
        cmd = [str(aida_executable), str(temp_input_file)]
        result = subprocess.run(cmd, env=env, check=True)
        
        # Read execution time from C++ output
        time_file = Path(str(temp_input_file) + "_AIDA_time.txt")
        if time_file.exists():
            with open(time_file, 'r') as f:
                execution_time = float(f.read())
            logger.info(f"AIDA model execution time: {execution_time:.6f} seconds")
        else:
            logger.error("AIDA time file not found")
            return False, -1, {}
        
        # Check output files
        scores_file = Path(str(temp_input_file) + "_AIDA_scores.dat")
        anomalies_file = Path(str(temp_input_file) + "_AIDA_anomalies.csv")
        
        if scores_file.exists() and anomalies_file.exists():
            output_scores = aida_output_dir / "aida_scores.dat"
            output_anomalies = aida_output_dir / "aida_anomalies.csv"
            import shutil
            shutil.copy2(scores_file, output_scores)
            shutil.copy2(anomalies_file, output_anomalies)
            
            try:
                # Process anomaly results to add stock information
                anomalies_df = pd.read_csv(output_anomalies)
                if 'stock_idx' in anomalies_df.columns and len(metadata_list) > 0:
                    # Extract stock tickers from metadata
                    all_tickers = []
                    for metadata in metadata_list:
                        if 'tickers' in metadata:
                            all_tickers.extend(metadata['tickers'])
                    
                    # Deduplicate tickers
                    unique_tickers = list(dict.fromkeys(all_tickers))
                    
                    # Add stock information
                    anomalies_df['stock_ticker'] = anomalies_df['stock_idx'].apply(
                        lambda idx: unique_tickers[idx] if idx < len(unique_tickers) else f"Unknown-{idx}"
                    )
                    
                    # Map index to time period
                    anomalies_df['time_period'] = anomalies_df['index'].apply(
                        lambda idx: idx // len(unique_tickers)
                    )
                    
                    # Save updated results
                    anomalies_df.to_csv(output_anomalies, index=False)
                    logger.info(f"Added stock information to AIDA anomalies")
            except Exception as e:
                logger.error(f"Error adding stock information to AIDA anomalies: {e}")
            
            logger.info(f"AIDA execution successful. Results saved to {aida_output_dir}")
            return True, execution_time, {"scores": output_scores, "anomalies": output_anomalies, "time": time_file}
        else:
            logger.error("AIDA execution failed: Output files not found")
            return False, -1, {}
        
    except subprocess.CalledProcessError as e:
        logger.error(f"AIDA execution failed: {e}")
        return False, -1, {}
    except Exception as e:
        logger.error(f"AIDA execution error: {e}")
        return False, -1, {}


def run_isolation_forest_on_matrix(feature_array, metadata_list, output_dir, window_size, overlap):
    """
    Run Isolation Forest algorithm on matrix data.
    
    Args:
        feature_array (numpy.ndarray): Array of feature vectors
        metadata_list (list): List of metadata dictionaries
        output_dir (Path): Directory to save results
        window_size (int): Size of the subsequence window
        overlap (bool): Whether subsequences are overlapping
        
    Returns:
        tuple: (success, execution_time, output_files)
            success: Boolean indicating if execution was successful
            execution_time: Execution time in seconds
            output_files: Dictionary with paths to output files
    """
    # Create subdirectory name based on parameters
    subdir_name = f"w{window_size}_{'overlap' if overlap else 'nonoverlap'}"
    logger.info(f"Running Isolation Forest algorithm on matrix data ({subdir_name})...")
    
    # Ensure output directory exists
    iforest_output_dir = output_dir / "iforest" / subdir_name
    ensure_directory_exists(iforest_output_dir)
    
    try:
        # Start timing
        start_time = time.time()
        
        # Initialize and run Isolation Forest
        iforest = IForest(
            n_estimators=100,
            max_samples=min(256, feature_array.shape[0]),
            contamination=0.05
        )
        
        # Fit and predict
        scores, labels = iforest.fit_predict(feature_array)
        
        # End timing
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Save execution time
        time_file = iforest_output_dir / "iforest_execution_time.txt"
        with open(time_file, 'w') as f:
            f.write(f"{execution_time:.6f}")
            
        logger.info(f"Isolation Forest execution time: {execution_time:.6f} seconds")
        
        # Save scores
        scores_file = iforest_output_dir / "iforest_scores.dat"
        with open(scores_file, 'w') as f:
            f.write(f"{len(scores)}\n")
            for score in scores:
                f.write(f"{score}\n")
        
        # Save anomalies
        anomalies_file = iforest_output_dir / "iforest_anomalies.csv"
        anomaly_indices = np.where(labels == -1)[0]
        
        # Create DataFrame with anomalies
        anomaly_data = []
        
        try:
            # Extract stock tickers from metadata
            all_tickers = []
            for metadata in metadata_list:
                if 'tickers' in metadata:
                    all_tickers.extend(metadata['tickers'])
            
            # Deduplicate tickers
            unique_tickers = list(dict.fromkeys(all_tickers))
            
            # Get number of stocks/tickers
            n_stocks = len(unique_tickers)
            
            # Process anomalies
            for idx in anomaly_indices:
                stock_idx = idx % n_stocks
                time_period = idx // n_stocks
                stock_ticker = unique_tickers[stock_idx] if stock_idx < n_stocks else f"Unknown-{stock_idx}"
                
                anomaly_data.append({
                    'index': idx,
                    'stock_idx': stock_idx,
                    'time_period': time_period,
                    'stock_ticker': stock_ticker,
                    'score': scores[idx]
                })
        except Exception as e:
            logger.error(f"Error processing stock information: {e}")
            # Fallback with minimal information
            for idx in anomaly_indices:
                anomaly_data.append({
                    'index': idx,
                    'score': scores[idx]
                })
        
        if anomaly_data:
            anomaly_df = pd.DataFrame(anomaly_data)
            anomaly_df.to_csv(anomalies_file, index=False)
            
            logger.info(f"Saved Isolation Forest scores to {scores_file}")
            logger.info(f"Saved Isolation Forest anomalies to {anomalies_file}")
            
            return True, execution_time, {"scores": scores_file, "anomalies": anomalies_file, "time": time_file}
        else:
            logger.warning("No anomalies detected by Isolation Forest")
            return False, execution_time, {}
            
    except Exception as e:
        execution_time = time.time() - start_time if 'start_time' in locals() else -1
        logger.error(f"Isolation Forest execution error: {e}")
        return False, execution_time, {}


def run_lof_on_matrix(feature_array, metadata_list, output_dir, window_size, overlap):
    """
    Run Local Outlier Factor algorithm on matrix data.
    
    Args:
        feature_array (numpy.ndarray): Array of feature vectors
        metadata_list (list): List of metadata dictionaries
        output_dir (Path): Directory to save results
        window_size (int): Size of the subsequence window
        overlap (bool): Whether subsequences are overlapping
        
    Returns:
        tuple: (success, execution_time, output_files)
            success: Boolean indicating if execution was successful
            execution_time: Execution time in seconds
            output_files: Dictionary with paths to output files
    """
    # Create subdirectory name based on parameters
    subdir_name = f"w{window_size}_{'overlap' if overlap else 'nonoverlap'}"
    logger.info(f"Running Local Outlier Factor algorithm on matrix data ({subdir_name})...")
    
    # Ensure output directory exists
    lof_output_dir = output_dir / "lof" / subdir_name
    ensure_directory_exists(lof_output_dir)
    
    try:
        # Start timing
        start_time = time.time()
        
        # Initialize and run LOF
        lof = LOF(
            n_neighbors=min(20, feature_array.shape[0] - 1),
            p=2,  # Euclidean distance
            contamination=0.05
        )
        
        # Fit and predict
        scores, labels = lof.fit_predict(feature_array)
        
        # End timing
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Save execution time
        time_file = lof_output_dir / "lof_execution_time.txt"
        with open(time_file, 'w') as f:
            f.write(f"{execution_time:.6f}")
            
        logger.info(f"LOF execution time: {execution_time:.6f} seconds")
        
        # Save scores
        scores_file = lof_output_dir / "lof_scores.dat"
        with open(scores_file, 'w') as f:
            f.write(f"{len(scores)}\n")
            for score in scores:
                f.write(f"{score}\n")
        
        # Save anomalies
        anomalies_file = lof_output_dir / "lof_anomalies.csv"
        anomaly_indices = np.where(labels == -1)[0]
        
        # Create DataFrame with anomalies
        anomaly_data = []
        
        try:
            # Extract stock tickers from metadata
            all_tickers = []
            for metadata in metadata_list:
                if 'tickers' in metadata:
                    all_tickers.extend(metadata['tickers'])
            
            # Deduplicate tickers
            unique_tickers = list(dict.fromkeys(all_tickers))
            
            # Get number of stocks/tickers
            n_stocks = len(unique_tickers)
            
            # Process anomalies
            for idx in anomaly_indices:
                stock_idx = idx % n_stocks
                time_period = idx // n_stocks
                stock_ticker = unique_tickers[stock_idx] if stock_idx < n_stocks else f"Unknown-{stock_idx}"
                
                anomaly_data.append({
                    'index': idx,
                    'stock_idx': stock_idx,
                    'time_period': time_period,
                    'stock_ticker': stock_ticker,
                    'score': scores[idx]
                })
        except Exception as e:
            logger.error(f"Error processing stock information: {e}")
            # Fallback with minimal information
            for idx in anomaly_indices:
                anomaly_data.append({
                    'index': idx,
                    'score': scores[idx]
                })
        
        if anomaly_data:
            anomaly_df = pd.DataFrame(anomaly_data)
            anomaly_df.to_csv(anomalies_file, index=False)
            
            logger.info(f"Saved LOF scores to {scores_file}")
            logger.info(f"Saved LOF anomalies to {anomalies_file}")
            
            return True, execution_time, {"scores": scores_file, "anomalies": anomalies_file, "time": time_file}
        else:
            logger.warning("No anomalies detected by LOF")
            return False, execution_time, {}
            
    except Exception as e:
        execution_time = time.time() - start_time if 'start_time' in locals() else -1
        logger.error(f"LOF execution error: {e}")
        return False, execution_time, {}


def main():
    """
    Main function to run matrix-based anomaly detection on S&P 500 and constituent data.
    """
    parser = argparse.ArgumentParser(description="Run matrix-based anomaly detection on S&P 500 and constituent data")
    parser.add_argument(
        "--multi-ts-dir", 
        type=str, 
        default=str(config.PROCESSED_DATA_DIR / "multi_ts"),
        help="Directory containing multi-TS data"
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
        default=str(config.DATA_DIR / "matrix_results"),
        help="Directory to save algorithm results"
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
    multi_ts_dir = Path(args.multi_ts_dir)
    output_dir = Path(args.output)
    
    # Create output directory
    ensure_directory_exists(output_dir)
    
    # Load multi-TS data
    data_matrices, metadata_list = load_multi_ts_data(
        multi_ts_dir, 
        window_size=args.window_size, 
        overlap=overlap
    )
    
    if data_matrices is None or not data_matrices:
        logger.error(f"Failed to load multi-TS data for window_size={args.window_size}, overlap={overlap}. Exiting.")
        return
    
    # Prepare matrix features
    feature_array = prepare_matrix_features(data_matrices)
    
    if feature_array is None:
        logger.error("Failed to prepare matrix features. Exiting.")
        return
    
    # Initialize results dictionary
    results = {}
    
    # Run algorithms
    if "aida" in algorithms:
        success, execution_time, output_files = run_aida_on_matrix(
            feature_array, metadata_list, output_dir, args.window_size, overlap
        )
        results["aida"] = {
            "success": success, 
            "execution_time": execution_time, 
            "files": output_files
        }
    
    if "iforest" in algorithms:
        success, execution_time, output_files = run_isolation_forest_on_matrix(
            feature_array, metadata_list, output_dir, args.window_size, overlap
        )
        results["iforest"] = {
            "success": success, 
            "execution_time": execution_time, 
            "files": output_files
        }
    
    if "lof" in algorithms:
        success, execution_time, output_files = run_lof_on_matrix(
            feature_array, metadata_list, output_dir, args.window_size, overlap
        )
        results["lof"] = {
            "success": success, 
            "execution_time": execution_time, 
            "files": output_files
        }
    
    # Print summary
    logger.info(f"\nMatrix Analysis Summary for window_size={args.window_size}, overlap={overlap}:")
    for algo, result in results.items():
        success = result.get("success", False)
        execution_time = result.get("execution_time", -1)
        logger.info(f"{algo.upper()}: {'Success' if success else 'Failed'}, Execution time: {execution_time:.6f} seconds")
    
    # Save execution times for comparison
    execution_times = {
        algo: result.get("execution_time", -1) 
        for algo, result in results.items()
    }
    
    execution_times_file = output_dir / f"w{args.window_size}_{'overlap' if overlap else 'nonoverlap'}" / "execution_times.json"
    ensure_directory_exists(execution_times_file.parent)
    with open(execution_times_file, 'w') as f:
        json.dump(execution_times, f, indent=2)
    
    logger.info(f"Matrix analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
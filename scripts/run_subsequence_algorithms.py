#!/usr/bin/env python
"""
Script to run anomaly detection algorithms on subsequences of S&P 500 data.
Handles both overlapping and non-overlapping subsequences with configurable window sizes.
"""
import os
import sys
import logging
import argparse
import subprocess
import time
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

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


def load_subsequence_dataset(subsequence_dir, ticker, window_size=3, overlap=True):
    """
    Load subsequence data for anomaly detection.
    
    Args:
        subsequence_dir (Path): Directory containing subsequence files
        window_size (int): Size of the subsequence window (3, 5, etc.)
        overlap (bool): Whether to use overlapping or non-overlapping subsequences
        
    Returns:
        tuple: (subsequence_data, subsequence_dates)
            subsequence_data: List of DataFrames containing subsequences
            subsequence_dates: List of date ranges for subsequences
    """
    try:
        logger.info(f"Ticker now procesing: {ticker} beginning of loading subsequence dataset)")
        # Determine prefix based on parameters
        overlap_str = "overlap" if overlap else "nonoverlap"
        prefix = f"{ticker}_len{window_size}_{overlap_str}"
        
        logger.info(f"Loading subsequence dataset with prefix: {prefix}")
        
        # Get list of subsequence files
        subseq_files = get_file_list(subsequence_dir, f"{prefix}*.csv")
        
        if not subseq_files:
            logger.error(f"No subsequence files found with prefix {prefix} in {subsequence_dir}")
            return None, None
        
        logger.info(f"Found {len(subseq_files)} subsequence files")
        
        # Load subsequence data
        subsequence_data = []
        subsequence_dates = []
        
        for file_path in sorted(subseq_files):
            subseq = load_subsequence(file_path)
            
            if subseq is not None and not subseq.empty:
                # Extract metadata
                start_date = subseq.attrs.get('start_date', None)
                end_date = subseq.attrs.get('end_date', None)
                
                if start_date and end_date:
                    subsequence_dates.append((start_date, end_date))
                else:
                    # If no metadata, use index
                    subsequence_dates.append((subseq.index[0], subseq.index[-1]))
                
                # Store subsequence
                subsequence_data.append(subseq)
            
        logger.info(f"Loaded {len(subsequence_data)} valid subsequences")
        return subsequence_data, subsequence_dates
        
    except Exception as e:
        logger.error(f"Error loading subsequence dataset: {e}")
        return None, None


def prepare_subsequence_features(subsequence_data):
    """
    Prepare subsequence features for anomaly detection.
    """
    try:
        feature_vectors = []
        feature_names = None
        descriptive_feature_names = None

        for i, subseq in enumerate(subsequence_data):
            if i == 0:
                feature_names = list(subseq.columns)
                window_size = subseq.shape[0]
                # Build descriptive feature names: e.g., Close_0, Volume_2
                descriptive_feature_names = [
                    f"{col}_{day}" for day in range(window_size) for col in feature_names
                ]
                logger.info(f"Descriptive feature names: {descriptive_feature_names}")
            feature_vector = subseq.values.flatten()
            feature_vectors.append(feature_vector)

        feature_array = np.array(feature_vectors)
        logger.info(f"Prepared feature array with shape {feature_array.shape}")
        return feature_array, descriptive_feature_names

    except Exception as e:
        logger.error(f"Error preparing subsequence features: {e}")
        return None, None


def run_aida(feature_array, subsequence_dates, output_dir, window_size, overlap, descriptive_feature_names=None):
    """
    Run AIDA algorithm on subsequence features.
    
    Args:
        feature_array (numpy.ndarray): Array of feature vectors
        subsequence_dates (list): List of date ranges for subsequences
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
    logger.info(f"Running AIDA algorithm on subsequences ({subdir_name})...")
    
    # Ensure output directory exists
    aida_output_dir = output_dir / "aida" / subdir_name
    ensure_directory_exists(aida_output_dir)
    
    # Create temporary input file for AIDA
    temp_input_file = aida_output_dir / "subsequence_features.csv"
    
    try:
        # Save feature array to CSV with descriptive feature names if provided
        with open(temp_input_file, 'w') as f:
            if descriptive_feature_names is not None:
                f.write(','.join(descriptive_feature_names) + '\n')
            else:
                f.write(','.join([f'feature_{i}' for i in range(feature_array.shape[1])]) + '\n')
            for i in range(feature_array.shape[0]):
                f.write(','.join([str(val) for val in feature_array[i]]) + '\n')

        logger.info(f"Saved feature array to {temp_input_file}")
        
        # Paths to AIDA executable
        aida_cpp_dir = config.ROOT_DIR / "AIDA" / "C++"
        aida_executable = aida_cpp_dir / "build" / "aida_subsequence_detection"
        
        # Check if executable exists, if not attempt to compile
        if not aida_executable.exists():
            logger.info("AIDA executable not found. Attempting to compile...")
            src_models_cpp_dir = config.ROOT_DIR / "src" / "models" / "cpp"
            os.makedirs(src_models_cpp_dir, exist_ok=True)
            analysis_source_file = src_models_cpp_dir / "aida_subsequence_detection.cpp"
            
            # Basic C++ code for AIDA subsequence detection
            with open(analysis_source_file, "w") as f:
                f.write("""/* AIDA Anomaly Detection for Subsequences */
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
    cout << "Data loaded: " << n << " subsequences, " << nFnum << " features per subsequence" << endl;
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
        fanom << "index,subsequence_idx,score" << endl;
        for (int i = 0; i < n; ++i) {
            if (scoresAIDA[i] > threshold) {
                fanom << i << "," << i << "," << scoresAIDA[i] << endl;
                anomaly_count++;
            }
        }
        fanom.close();
        cout << "AIDA Subsequence Analysis Complete:" << endl;
        cout << "Total subsequences: " << n << endl;
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
                anomalies_df = pd.read_csv(output_anomalies)
                if 'subsequence_idx' in anomalies_df.columns and len(subsequence_dates) > 0:
                    start_dates = []
                    end_dates = []
                    for idx in anomalies_df['subsequence_idx']:
                        if 0 <= idx < len(subsequence_dates):
                            start_dates.append(subsequence_dates[idx][0])
                            end_dates.append(subsequence_dates[idx][1])
                        else:
                            start_dates.append(None)
                            end_dates.append(None)
                    anomalies_df['start_date'] = start_dates
                    anomalies_df['end_date'] = end_dates
                    anomalies_df.to_csv(output_anomalies, index=False)
                    logger.info(f"Added date information to AIDA anomalies")
            except Exception as e:
                logger.error(f"Error adding date information to AIDA anomalies: {e}")
            
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


def run_isolation_forest(feature_array, subsequence_dates, output_dir, window_size, overlap, descriptive_feature_names=None):
    """
    Run Isolation Forest algorithm on subsequence features.
    """
    # Create subdirectory name based on parameters
    subdir_name = f"w{window_size}_{'overlap' if overlap else 'nonoverlap'}"
    logger.info(f"Running Isolation Forest algorithm on subsequences ({subdir_name})...")
    
    # Ensure output directory exists
    iforest_output_dir = output_dir / "iforest" / subdir_name
    ensure_directory_exists(iforest_output_dir)
    
    # Save feature data for later analysis
    feature_file = iforest_output_dir / "iforest_features.csv"
    
    try:
        # Save feature array to CSV with descriptive feature names if provided
        if descriptive_feature_names is not None:
            feature_df = pd.DataFrame(feature_array, columns=descriptive_feature_names)
            feature_df.to_csv(feature_file, index=False)
        else:
            feature_df = pd.DataFrame(feature_array, columns=[f'feature_{i}' for i in range(feature_array.shape[1])])
            feature_df.to_csv(feature_file, index=False)
            
        logger.info(f"Saved feature array to {feature_file}")

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
        logger.info(f"Running Isolation Forest for ticker: {getattr(feature_array, 'ticker', 'unknown')} | Feature array shape: {feature_array.shape} | First row hash: {hash(tuple(feature_array[0])) if feature_array.size > 0 else 'empty'}")
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
        for idx in anomaly_indices:
            if 0 <= idx < len(subsequence_dates):
                anomaly_data.append({
                    'index': idx,
                    'subsequence_idx': idx,
                    'score': scores[idx],
                    'start_date': subsequence_dates[idx][0],
                    'end_date': subsequence_dates[idx][1]
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


def run_lof(feature_array, subsequence_dates, output_dir, window_size, overlap, descriptive_feature_names=None):
    """
    Run Local Outlier Factor algorithm on subsequence features.
    """
    # Create subdirectory name based on parameters
    subdir_name = f"w{window_size}_{'overlap' if overlap else 'nonoverlap'}"
    logger.info(f"Running Local Outlier Factor algorithm on subsequences ({subdir_name})...")
    
    # Ensure output directory exists
    lof_output_dir = output_dir / "lof" / subdir_name
    ensure_directory_exists(lof_output_dir)
    
    # Save feature data for later analysis
    feature_file = lof_output_dir / "lof_features.csv"
    
    try:
        # Save feature array to CSV with descriptive feature names if provided
        if descriptive_feature_names is not None:
            feature_df = pd.DataFrame(feature_array, columns=descriptive_feature_names)
            feature_df.to_csv(feature_file, index=False)
        else:
            feature_df = pd.DataFrame(feature_array, columns=[f'feature_{i}' for i in range(feature_array.shape[1])])
            feature_df.to_csv(feature_file, index=False)
            
        logger.info(f"Saved feature array to {feature_file}")
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
        for idx in anomaly_indices:
            if 0 <= idx < len(subsequence_dates):
                anomaly_data.append({
                    'index': idx,
                    'subsequence_idx': idx,
                    'score': scores[idx],
                    'start_date': subsequence_dates[idx][0],
                    'end_date': subsequence_dates[idx][1]
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
    Main function to run anomaly detection algorithms on subsequences.
    """
    parser = argparse.ArgumentParser(description="Run anomaly detection algorithms on S&P 500 subsequences")
    parser.add_argument(
        "--subsequence-dir", 
        type=str, 
        default=str(config.PROCESSED_DATA_DIR / "subsequences"),
        help="Directory containing subsequence data"
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
        help="Use overlapping subsequences (default: False)"
    )
    parser.add_argument(
        "--no-overlap", 
        action="store_true",
        help="Use non-overlapping subsequences"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=str(config.DATA_DIR / "analysis_results" / "subsequence_results"),
        help="Directory to save algorithm results"
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["aida", "iforest", "lof", "all"],
        default=["all"],
        help="Algorithms to run"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="sp500",
        help="Ticker symbol for which to run the analysis (default: sp500)"
    )

    args = parser.parse_args()
    logger.info(f"Ticker now procesing: {args.ticker} beginnig of run_subsequence_algorithms.py")
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
    subsequence_dir = Path(args.subsequence_dir)
    output_dir = Path(args.output)

    # Always append ticker to output_dir if not already present
    if output_dir.name != args.ticker:
        output_dir = output_dir / args.ticker

    # Create a subdirectory for results based on parameters
    config_dir = f"w{args.window_size}_{'overlap' if overlap else 'nonoverlap'}"
    results_dir = output_dir / config_dir
    ensure_directory_exists(results_dir)
    
    # Load subsequence dataset
    subsequence_data, subsequence_dates = load_subsequence_dataset(
        subsequence_dir,
        ticker=args.ticker, 
        window_size=args.window_size, 
        overlap=overlap
    )
    
    if subsequence_data is None or not subsequence_data:
        logger.error(f"Failed to load subsequence dataset for window_size={args.window_size}, overlap={overlap}. Exiting.")
        return
    
    # Prepare subsequence features (get both array and descriptive names)
    feature_array, descriptive_feature_names = prepare_subsequence_features(subsequence_data)

    if feature_array is None:
        logger.error("Failed to prepare subsequence features. Exiting.")
        return

    # Initialize results dictionary
    results = {}

    # Run algorithms
    if "aida" in algorithms:
        success, execution_time, output_files = run_aida(
            feature_array, subsequence_dates, output_dir, args.window_size, overlap, descriptive_feature_names
        )
        results["aida"] = {
            "success": success, 
            "execution_time": execution_time, 
            "files": output_files
        }
    
    if "iforest" in algorithms:
        success, execution_time, output_files = run_isolation_forest(
            feature_array, subsequence_dates, output_dir, args.window_size, overlap, descriptive_feature_names
        )
        results["iforest"] = {
            "success": success, 
            "execution_time": execution_time, 
            "files": output_files
        }
    
    if "lof" in algorithms:
        success, execution_time, output_files = run_lof(
            feature_array, subsequence_dates, output_dir, args.window_size, overlap, descriptive_feature_names
        )
        results["lof"] = {
            "success": success, 
            "execution_time": execution_time, 
            "files": output_files
        }
    
    # Print summary
    logger.info(f"\nAlgorithm execution summary for {config_dir}:")
    for algo, result in results.items():
        success = result.get("success", False)
        execution_time = result.get("execution_time", -1)
        logger.info(f"{algo.upper()}: {'Success' if success else 'Failed'}, Execution time: {execution_time:.6f} seconds")
    
    # Save execution times for comparison
    execution_times = {
        algo: result.get("execution_time", -1) 
        for algo, result in results.items()
    }
    
    execution_times_file = results_dir / "execution_times.json"
    with open(execution_times_file, 'w') as f:
        json.dump(execution_times, f, indent=2)
    
    logger.info(f"All results saved to {results_dir}")
    logger.info(f"Execution times saved to {execution_times_file}")


if __name__ == "__main__":
    main()
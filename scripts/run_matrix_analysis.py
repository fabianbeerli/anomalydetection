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
import sys
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


def detect_anomalies_aida_intrawindow(multi_ts_data, metadata_list, multi_ts_dir, window_size=5, overlap=True, contamination=0.05):
    """
    Detect anomalies in multi-TS data using AIDA matrix detection.
    Each matrix is saved as a separate CSV and analyzed individually.
    Records anomalies at the (window, stock) level.
    Also saves per-anomaly CSVs with all features for that stock in that subsequence.
    """
    try:
        logger.info(f"Running AIDA on {len(multi_ts_data)} multi-TS matrices (matrix mode)")

        temp_dir = Path(multi_ts_dir) / "temp_aida"
        temp_dir.mkdir(exist_ok=True, parents=True)

        # Adjust these paths as needed for your setup
        root_dir = Path(multi_ts_dir).parent.parent.parent
        aida_cpp_dir = root_dir / "AIDA" / "C++"
        aida_executable = aida_cpp_dir / "build" / "aida_matrix_detection"

        scores = []
        anomaly_records = []
        start_time = time.time()

        for window_idx, matrix in enumerate(multi_ts_data):
            # Save each matrix as a CSV file (flatten to 2D if needed)
            matrix_file = temp_dir / f"matrix_{window_idx}.csv"

            metadata = metadata_list[window_idx]
            feature_names = metadata.get('feature_names')
            period_tickers = metadata.get('tickers', [f"stock_{i}" for i in range(matrix.shape[0])])
            save_matrix_as_csv(matrix, matrix_file, feature_names=feature_names)

            # Call AIDA matrix detection
            cmd = [str(aida_executable), str(matrix_file)]
            subprocess.run(cmd, check=True)

            # Read results for this matrix
            scores_file = Path(str(matrix_file) + "_AIDA_scores.dat")
            anomalies_file = Path(str(matrix_file) + "_AIDA_anomalies.csv")

            # Read scores for all stocks in this window
            stock_scores = []
            if scores_file.exists():
                with open(scores_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:
                        stock_scores = [float(line.strip()) for line in lines[1:]]
            scores.append(np.mean(stock_scores) if stock_scores else 0.0)

            # Collect anomaly records from the C++ output
            if anomalies_file.exists():
                anomalies_df = pd.read_csv(anomalies_file)
                if not anomalies_df.empty and {'index', 'stock_idx', 'score'}.issubset(anomalies_df.columns):
                    for _, row in anomalies_df.iterrows():
                        stock_idx = int(row['stock_idx'])
                        # Check for off-by-one: print(period_tickers, stock_idx)
                        ticker = period_tickers[stock_idx] if stock_idx < len(period_tickers) else f"stock_{stock_idx}"
                        anomaly_record = {
                            'window_idx': window_idx,
                            'stock_idx': stock_idx,
                            'ticker': ticker,
                            'score': row['score'],
                            'start_date': metadata.get('start_date', 'Unknown'),
                            'end_date': metadata.get('end_date', 'Unknown'),
                        }
                        anomaly_records.append(anomaly_record)

                        # Save per-anomaly CSV with all features for this stock in this subsequence
                        subseq_dir = Path(multi_ts_dir) / f"multi_ts_w{window_size}_{'overlap' if overlap else 'nonoverlap'}" / f"Subsequence{window_idx}" / ticker
                        subseq_dir.mkdir(parents=True, exist_ok=True)
                        csv_path = subseq_dir / f"multi_ts_{ticker}_anomaly_subsequence{window_idx}.csv"

                        # Extract features for this stock in this subsequence
                        stock_features = matrix[stock_idx].reshape(-1)
                        if feature_names is not None:
                            feature_header = [f"{feat}_{day}" for day in range(matrix.shape[1]) for feat in feature_names]
                        else:
                            feature_header = [f"feature_{i}" for i in range(stock_features.shape[0])]
                        features_dict = dict(zip(feature_header, stock_features))

                        output_dict = {
                            'window_idx': window_idx,
                            'stock_idx': stock_idx,
                            'ticker': ticker,
                            'score': row['score'],
                            'start_date': metadata.get('start_date', 'Unknown'),
                            'end_date': metadata.get('end_date', 'Unknown'),
                            **features_dict
                        }
                        pd.DataFrame([output_dict]).to_csv(csv_path, index=False)

        end_time = time.time()
        execution_time = end_time - start_time

        logger.info(f"AIDA execution time: {execution_time:.2f} seconds")

        anomaly_indices = [(rec['window_idx'], rec['stock_idx']) for rec in anomaly_records]
        return np.array(scores), anomaly_indices, anomaly_records, execution_time

    except Exception as e:
        logger.error(f"Error detecting anomalies with AIDA: {e}")
        return None, None, None, -1
    
# def detect_anomalies_aida(multi_ts_data, metadata_list, multi_ts_dir, window_size=5, overlap=True, contamination=0.05):
#     """
#     Detect anomalies in multi-TS data using AIDA matrix detection.
#     Now treats each time window (with all stocks) as a unit and compares windows to windows.
#     
#     Args:
#         multi_ts_data (numpy.ndarray): Combined matrix with each row representing a window
#         metadata_list (list): List of metadata dictionaries for each window
#         multi_ts_dir (Path): Directory to save results
#         window_size (int): Window size used
#         overlap (bool): Whether windows overlap
#         contamination (float): Expected proportion of anomalies
#         
#     Returns:
#         tuple: (scores, anomaly_indices, anomaly_records, execution_time)
#     """
#     try:
#         logger.info(f"Running AIDA on multi-TS data with {len(multi_ts_data)} windows")
#         
#         # Determine matrix dimensions from the data
#         n_windows, flat_dim = multi_ts_data.shape
#         
#         # For AIDA Matrix Detection C++
#         aida_cpp_dir = Path(multi_ts_dir).parent.parent.parent / "AIDA" / "C++"
#         aida_executable = aida_cpp_dir / "build" / "aida_matrix_detection"
#         
#         # Check if executable exists, if not create it
#         if not aida_executable.exists():
#             logger.info(f"AIDA matrix detection executable not found. Creating...")
#             
#             # Create models/cpp directory if it doesn't exist
#             models_cpp_dir = Path(multi_ts_dir).parent.parent.parent / "src" / "models" / "cpp"
#             os.makedirs(models_cpp_dir, exist_ok=True)
#             
#             # Create the matrix detection source file directly
#             matrix_detection_file = models_cpp_dir / "aida_matrix_detection.cpp"
#             
#             with open(matrix_detection_file, 'w') as f:
#                 f.write("""/* AIDA Matrix Detection for Multi-TS Analysis */
# #include <iostream>
# #include <fstream>
# #include <sstream>
# #include <vector>
# #include <cmath>
# #include <string>
# using namespace std;
# 
# // Set these to your actual values!
# const int n_rows = 30; // e.g., number of stocks
# const int n_cols = 15; // e.g., window_size * n_features
# 
# double frobenius_distance(const vector<vector<double>>& A, const vector<vector<double>>& B) {
#     double sum = 0.0;
#     for (size_t i = 0; i < A.size(); ++i)
#         for (size_t j = 0; j < A[0].size(); ++j)
#             sum += (A[i][j] - B[i][j]) * (A[i][j] - B[i][j]);
#     return sqrt(sum);
# }
# 
# int main(int argc, char** argv) {
#     if (argc < 2) {
#         cerr << "Usage: " << argv[0] << " <input_csv_file>" << endl;
#         return 1;
#     }
#     string input_file = argv[1];
#     ifstream file(input_file);
#     if (!file.is_open()) {
#         cerr << "Error: Could not open input file " << input_file << endl;
#         return 1;
#     }
#     vector<vector<vector<double>>> matrices;
#     string line;
#     while (getline(file, line)) {
#         stringstream ss(line);
#         string cell;
#         vector<double> flat_row;
#         while (getline(ss, cell, ',')) {
#             flat_row.push_back(stod(cell));
#         }
#         // Each row is a stock, with window_size * n_features columns
#         vector<vector<double>> mat(n_rows, vector<double>(n_cols));
#         for (int i = 0; i < n_rows; ++i)
#             for (int j = 0; j < n_cols; ++j)
#                 mat[i][j] = flat_row[i * n_cols + j];
#         matrices.push_back(mat);
#     }
#     file.close();
# 
#     // Compute anomaly scores (average Frobenius distance to all others)
#     vector<double> scores(matrices.size(), 0.0);
#     for (size_t i = 0; i < matrices.size(); ++i) {
#         double sum_dist = 0.0;
#         for (size_t j = 0; j < matrices.size(); ++j) {
#             if (i == j) continue;
#             sum_dist += frobenius_distance(matrices[i], matrices[j]);
#         }
#         scores[i] = sum_dist / (matrices.size() - 1);
#     }
# 
#     // Output scores
#     ofstream fres(string(input_file) + "_AIDA_scores.dat");
#     fres << matrices.size() << endl;
#     for (size_t i = 0; i < scores.size(); ++i)
#         fres << scores[i] << endl;
#     fres.close();
# 
#     // Threshold and output anomalies
#     double mean = 0.0, stddev = 0.0;
#     for (double s : scores) mean += s;
#     mean /= scores.size();
#     for (double s : scores) stddev += (s - mean) * (s - mean);
#     stddev = sqrt(stddev / scores.size());
#     double threshold = mean + 2 * stddev;
# 
#     ofstream fanom(string(input_file) + "_AIDA_anomalies.csv");
#     fanom << "index,score" << endl;
#     for (size_t i = 0; i < scores.size(); ++i) {
#         if (scores[i] > threshold)
#             fanom << i << "," << scores[i] << endl;
#     }
#     fanom.close();
# 
#     cout << "Done. Scores and anomalies written." << endl;
#     return 0;
# }""")
#             
#             # Compile the C++ code
#             if sys.platform == 'darwin':  # macOS
#                 try:
#                     # Check if Homebrew and OpenMP are available
#                     subprocess.run(["brew", "--version"], check=True, capture_output=True)
#                     
#                     try:
#                         subprocess.run(["brew", "list", "libomp"], check=True, capture_output=True)
#                     except subprocess.CalledProcessError:
#                         logger.info("Installing OpenMP via Homebrew...")
#                         subprocess.run(["brew", "install", "libomp"], check=True)
#                     
#                     libomp_prefix = subprocess.run(
#                         ["brew", "--prefix", "libomp"], 
#                         check=True, 
#                         capture_output=True, 
#                         text=True
#                     ).stdout.strip()
#                     
#                     # Compile with OpenMP support
#                     compile_cmd = [
#                         "g++", "-std=c++11", "-O3", "-Xpreprocessor", "-fopenmp",
#                         f"-I{libomp_prefix}/include",
#                         f"-L{libomp_prefix}/lib",
#                         "-lomp",
#                         str(matrix_detection_file),
#                         "-o", str(aida_executable)
#                     ]
#                 except (subprocess.CalledProcessError, FileNotFoundError):
#                     # Fallback to basic compilation if Homebrew not available
#                     compile_cmd = [
#                         "g++", "-std=c++11", "-O3",
#                         str(matrix_detection_file),
#                         "-o", str(aida_executable)
#                     ]
#             else:  # Linux/Windows
#                 compile_cmd = [
#                     "g++", "-std=c++11", "-O3",
#                     str(matrix_detection_file),
#                     "-o", str(aida_executable)
#                 ]
#             
#             # Create the executable
#             os.makedirs(os.path.dirname(aida_executable), exist_ok=True)
#             subprocess.run(compile_cmd, check=True)
#             logger.info(f"Successfully created AIDA matrix detection executable")
#         
#         # Run AIDA matrix detection
#         matrix_file = multi_ts_dir / "multi_ts_features.csv"
#         
#         start_time = time.time()
#         env = os.environ.copy()
#         
#         # Handle macOS OpenMP library path
#         if sys.platform == 'darwin':
#             try:
#                 result = subprocess.run(["brew", "--prefix", "libomp"], 
#                                       check=True, 
#                                       capture_output=True, 
#                                       text=True)
#                 libomp_prefix = result.stdout.strip()
#                 env["DYLD_LIBRARY_PATH"] = f"{libomp_prefix}/lib"
#             except:
#                 pass
#         
#         cmd = [str(aida_executable), str(matrix_file)]
#         subprocess.run(cmd, env=env, check=True)
#         
#         end_time = time.time()
#         execution_time = end_time - start_time
# 
#         logger.info(f"AIDA execution time: {execution_time:.2f} seconds")
# 
#         # Read results
#         scores_file = Path(str(matrix_file) + "_AIDA_scores.dat")
#         anomalies_file = Path(str(matrix_file) + "_AIDA_anomalies.csv")
#         
#         if not scores_file.exists() or not anomalies_file.exists():
#             logger.error("AIDA failed to produce output files")
#             return None, None, None, -1
#         
#         # Read scores
#         scores = []
#         with open(scores_file, 'r') as f:
#             lines = f.readlines()
#             if len(lines) > 1:
#                 for i in range(1, len(lines)):
#                     scores.append(float(lines[i].strip()))
#         
#         # Read anomalies
#         anomaly_df = pd.read_csv(anomalies_file)
#         anomaly_indices = anomaly_df['index'].tolist()
#         
#         # Create anomaly records with metadata
#         anomaly_records = []
#         for idx, row in anomaly_df.iterrows():
#             window_idx = int(row['index'])
#             if 0 <= window_idx < len(metadata_list):
#                 metadata = metadata_list[window_idx]
#                 anomaly_record = {
#                     'window_idx': window_idx,
#                     'time_period_idx': window_idx,  # For compatibility with other functions
#                     'score': row['score'],
#                     'start_date': metadata.get('start_date', ''),
#                     'end_date': metadata.get('end_date', '')
#                 }
#                 # Add top stocks if possible (determined by some heuristic)
#                 if 'tickers' in metadata:
#                     anomaly_record['top_tickers'] = ', '.join(metadata['tickers'][:3])
#                 
#                 anomaly_records.append(anomaly_record)
#         
#         return np.array(scores), anomaly_indices, anomaly_records, execution_time
# 
#     except Exception as e:
#         logger.error(f"Error detecting anomalies with AIDA matrix detection: {e}")
#         return None, None, None, -1


def detect_anomalies_aida_windowwise(multi_ts_data, metadata_list, multi_ts_dir, window_size=3, overlap=True, contamination=0.05):
    """
    Detect anomalies in multi-TS data using AIDA matrix detection.
    Treats each time window (with all stocks) as a unit and compares windows to windows.
    
    Args:
        multi_ts_data (numpy.ndarray): Combined matrix with each row representing a window
        metadata_list (list): List of metadata dictionaries for each window
        multi_ts_dir (Path): Directory to save results
        window_size (int): Window size used
        overlap (bool): Whether windows overlap
        contamination (float): Expected proportion of anomalies
        
    Returns:
        tuple: (scores, anomaly_indices, anomaly_records, execution_time)
    """
    try:
        logger.info(f"Running AIDA on multi-TS data with {len(multi_ts_data)} windows")
        
        # Determine matrix dimensions from the data
        n_windows, flat_dim = multi_ts_data.shape
        
        # For AIDA Matrix Detection C++
        aida_cpp_dir = Path(multi_ts_dir).parent.parent.parent.parent.parent / "AIDA" / "C++"
        aida_executable = aida_cpp_dir / "build" / "aida_matrix_detection"
        
        logger.info(f"AIDA executable path: {aida_executable}")
        
        # Check if executable exists
        if not aida_executable.exists():
            logger.error(f"AIDA matrix detection executable not found at {aida_executable}")
            return None, None, None, -1
        
        # Run AIDA matrix detection
        matrix_file = multi_ts_dir / "multi_ts_features.csv"
        
        start_time = time.time()
        env = os.environ.copy()
        
        # Handle macOS OpenMP library path
        if sys.platform == 'darwin':
            try:
                result = subprocess.run(["brew", "--prefix", "libomp"], 
                                     check=True, 
                                     capture_output=True, 
                                     text=True)
                libomp_prefix = result.stdout.strip()
                env["DYLD_LIBRARY_PATH"] = f"{libomp_prefix}/lib"
            except:
                pass
        
        cmd = [str(aida_executable), str(matrix_file)]
        subprocess.run(cmd, env=env, check=True)
        
        end_time = time.time()
        execution_time = end_time - start_time

        logger.info(f"AIDA execution time: {execution_time:.2f} seconds")

        # Read results
        scores_file = Path(str(matrix_file) + "_AIDA_scores.dat")
        anomalies_file = Path(str(matrix_file) + "_AIDA_anomalies.csv")
        
        if not scores_file.exists() or not anomalies_file.exists():
            logger.error("AIDA failed to produce output files")
            return None, None, None, -1
        
        # Read scores
        scores = []
        with open(scores_file, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                for i in range(1, len(lines)):
                    scores.append(float(lines[i].strip()))
        
        # Read anomalies
        anomaly_df = pd.read_csv(anomalies_file)
        anomaly_indices = anomaly_df['index'].tolist()
        
        # Create anomaly records with metadata
        anomaly_records = []
        for idx, row in anomaly_df.iterrows():
            window_idx = int(row['index'])
            if 0 <= window_idx < len(metadata_list):
                metadata = metadata_list[window_idx]
                anomaly_record = {
                    'window_idx': window_idx,
                    'time_period_idx': window_idx,  # For compatibility with other functions
                    'score': row['score'],
                    'start_date': metadata.get('start_date', ''),
                    'end_date': metadata.get('end_date', '')
                }
                # Add top stocks if possible (determined by some heuristic)
                if 'tickers' in metadata:
                    anomaly_record['top_tickers'] = ', '.join(metadata['tickers'][:3])
                
                anomaly_records.append(anomaly_record)
        
        return np.array(scores), anomaly_indices, anomaly_records, execution_time

    except Exception as e:
        logger.error(f"Error detecting anomalies with AIDA matrix detection: {e}")
        return None, None, None, -1
    
def save_full_multi_ts_matrix(multi_ts_data, metadata_list, output_csv):
    """
    Save the full multi-TS feature matrix as a CSV with real feature names as header.
    Each row is a flattened window for a single stock, as in subsequence_features.csv.
    Adds a time_period_idx and ticker column for TIX analysis.
    """
    if not multi_ts_data or not metadata_list:
        logger.error("No multi-TS data or metadata to save.")
        return

    feature_names = metadata_list[0].get('feature_names')
    window_size = multi_ts_data[0].shape[1]

    header = [f"{feat}_{day}" for day in range(window_size) for feat in feature_names]
    rows = []
    time_period_indices = []
    tickers = []

    for period_idx, (matrix, metadata) in enumerate(zip(multi_ts_data, metadata_list)):
        # matrix shape: (n_stocks, window_size, n_features)
        n_stocks = matrix.shape[0]
        period_tickers = metadata.get('tickers', [f"stock_{i}" for i in range(n_stocks)])
        for stock_idx in range(n_stocks):
            row = matrix[stock_idx].reshape(-1)
            rows.append(row)
            time_period_indices.append(period_idx)
            tickers.append(period_tickers[stock_idx])

    df = pd.DataFrame(rows, columns=header)
    df['ticker'] = tickers
    df['time_period_idx'] = time_period_indices
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved full multi-TS feature matrix to {output_csv}")

def save_matrix_as_csv(matrix, csv_path, feature_names=None):
    """
    Save a 3D matrix (n_stocks, window_size, n_features) as a 2D CSV with descriptive headers.
    """
    n_stocks, window_size, n_features = matrix.shape
    matrix_2d = matrix.reshape(n_stocks, window_size * n_features)
    
    # Build descriptive feature names if provided
    if feature_names is not None:
        header = [f"{feat}_{day}" for day in range(window_size) for feat in feature_names]
        logger.info(f"Saving matrix to {csv_path} with header: {header}")
    else:
        header = [f"feature_{i}" for i in range(window_size * n_features)]
        logger.info(f"Saving matrix to {csv_path} with generic header: {header}")
    
    df = pd.DataFrame(matrix_2d, columns=header)
    df.to_csv(csv_path, index=False)

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
        # Create output directory with subfolder based on window level
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
                    'window_idx': period.get('window_idx'),
                    'stock_idx': period.get('stock_idx'),
                    'ticker': period.get('ticker'),
                    'score': period.get('score'),
                    'start_date': period.get('start_date'),
                    'end_date': period.get('end_date')
                })
            
            # Save as CSV
            pd.DataFrame(anomaly_data).to_csv(anomalies_file, index=False)
        else:
            # Create empty file
            with open(anomalies_file, 'w') as f:
                f.write("index,window_idx,stock_idx,ticker,score,start_date,end_date\n")
        
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

def run_multi_ts_analysis_windowwise(multi_ts_dir, output_dir, window_size=3, overlap=True, algorithms=None):
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
        algorithms = ['aida', 'iforest', 'lof'] 
    
    # Create output directory
    overlap_str = "overlap" if overlap else "nonoverlap"
    config_dir = f"multi_ts_w{window_size}_{overlap_str}"
    results_dir = Path(output_dir) / config_dir / "windowwise"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Load multi-TS data
    prefix = f"multi_ts_len{window_size}_{'overlap' if overlap else 'nonoverlap'}"
    multi_ts_files = list(Path(multi_ts_dir).glob(f"{prefix}*.npz"))

    if not multi_ts_files:
        logger.error(f"Failed to load multi-TS data. Exiting.")
        return {'status': 'error', 'message': 'Failed to load multi-TS data'}
    
    logger.info(f"Found {len(multi_ts_files)} multi-TS files")
    
    # Step 1: Load all multi-TS matrices and their metadata
    all_matrices = []
    all_metadata = []
    
    for file_path in sorted(multi_ts_files):
        try:
            # Load NPZ file
            npz_data = np.load(file_path, allow_pickle=True)
            matrix = npz_data['matrix']
            metadata = json.loads(npz_data['metadata'].item())
            
            # Add to lists
            all_matrices.append(matrix)
            all_metadata.append(metadata)
            
        except Exception as e:
            logger.error(f"Error loading multi-TS file {file_path}: {e}")
    
    if not all_matrices:
        logger.error(f"No valid matrices loaded. Exiting.")
        return {'status': 'error', 'message': 'No valid matrices loaded'}
    
    # Step 2: Prepare combined matrix for window-to-window comparison
    # Each row represents one time window with all stocks flattened
    combined_matrix = []
    
    for matrix in all_matrices:
        # Flatten the 3D matrix (n_stocks, window_size, n_features) into a 1D array
        flattened = matrix.flatten()
        combined_matrix.append(flattened)
    
    combined_matrix = np.array(combined_matrix)
    logger.info(f"Prepared combined matrix with shape {combined_matrix.shape}")
    
    # Step 3: Save the combined matrix to a file for AIDA C++
    matrix_file = results_dir / "multi_ts_features.csv"
    np.savetxt(matrix_file, combined_matrix, delimiter=',')
    logger.info(f"Saved combined matrix to {matrix_file}")
    
    # Save metadata for future reference
    metadata_file = results_dir / "multi_ts_metadata.json"
    window_metadata = {
        'window_count': len(all_matrices),
        'matrix_shape': all_matrices[0].shape,
        'windows': all_metadata
    }
    with open(metadata_file, 'w') as f:
        json.dump(window_metadata, f, indent=2)
    
    # Initialize results
    results = {'status': 'success', 'algorithms': {}}
    
    # Run algorithms
    for algo in algorithms:
        algo_dir = results_dir / algo
        algo_dir.mkdir(exist_ok=True, parents=True)
        
        if algo == 'aida':
            scores, indices, periods, time_taken = detect_anomalies_aida_windowwise(
                combined_matrix, all_metadata, results_dir, window_size=window_size, overlap=overlap
            )
            
            if scores is not None:
                files = save_multi_ts_results('aida', scores, periods, time_taken, results_dir)
                results['algorithms']['aida'] = {
                    'success': True,
                    'anomaly_count': len(indices) if indices is not None else 0,
                    'execution_time': time_taken,
                    'files': files,
                    'windowlevel': 'windowwise'
                }
        elif algo == 'iforest':
            from sklearn.ensemble import IsolationForest
            logger.info("Running Isolation Forest for multi-TS windowwise anomaly detection")
            model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
            model.fit(combined_matrix)
            scores = -model.decision_function(combined_matrix)
            # Use mean+2std as threshold for anomaly
            threshold = scores.mean() + 2 * scores.std()
            anomaly_indices = np.where(scores > threshold)[0]
            anomaly_records = []
            for idx in anomaly_indices:
                metadata = all_metadata[idx]
                anomaly_records.append({
                    'window_idx': idx,
                    'time_period_idx': idx,
                    'score': scores[idx],
                    'start_date': metadata.get('start_date', ''),
                    'end_date': metadata.get('end_date', '')
                })
            files = save_multi_ts_results('iforest', scores, anomaly_records, None, results_dir)
            results['algorithms']['iforest'] = {
                'success': True,
                'anomaly_count': len(anomaly_indices),
                'execution_time': None,
                'files': files,
                'windowlevel': 'windowwise'
            }
        elif algo == 'lof':
            from sklearn.neighbors import LocalOutlierFactor
            logger.info("Running LOF for multi-TS windowwise anomaly detection")
            # LOF does not have a fit_predict for out-of-sample, so use fit_predict
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
            lof_labels = lof.fit_predict(combined_matrix)
            scores = -lof.negative_outlier_factor_
            threshold = scores.mean() + 2 * scores.std()
            anomaly_indices = np.where(scores > threshold)[0]
            anomaly_records = []
            for idx in anomaly_indices:
                metadata = all_metadata[idx]
                anomaly_records.append({
                    'window_idx': idx,
                    'time_period_idx': idx,
                    'score': scores[idx],
                    'start_date': metadata.get('start_date', ''),
                    'end_date': metadata.get('end_date', '')
                })
            files = save_multi_ts_results('lof', scores, anomaly_records, None, results_dir)
            results['algorithms']['lof'] = {
                'success': True,
                'anomaly_count': len(anomaly_indices),
                'execution_time': None,
                'files': files,
                'windowlevel': 'windowwise'
            }
    
    # Create a summary of results
    summary_file = results_dir  / "multi_ts_analysis_summary.json"
    
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

def run_multi_ts_analysis_intrawindow(multi_ts_dir, output_dir, window_size=3, overlap=True, algorithms=None):
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
        algorithms = ['aida', 'iforest', 'lof'] 
    
    # Create output directory
    overlap_str = "overlap" if overlap else "nonoverlap"
    config_dir = f"multi_ts_w{window_size}_{overlap_str}"
    results_dir = Path(output_dir) / config_dir / "intrawindow"
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Load multi-TS data
    prefix = f"multi_ts_len{window_size}_{'overlap' if overlap else 'nonoverlap'}"
    multi_ts_data, metadata_list = load_multi_ts_data(multi_ts_dir, prefix)

    if multi_ts_data is None or not multi_ts_data:
        logger.error(f"Failed to load multi-TS data. Exiting.")
        return {'status': 'error', 'message': 'Failed to load multi-TS data'}

    # Save the full multi-TS feature matrix as a CSV for downstream analysis (e.g., TIX)
    full_matrix_csv = results_dir / "multi_ts_features.csv"
    save_full_multi_ts_matrix(multi_ts_data, metadata_list, full_matrix_csv)

    if multi_ts_data is None:
        logger.error(f"Failed to load multi-TS data. Exiting.")
        return {'status': 'error', 'message': 'Failed to load multi-TS data'}

    
    # Initialize results
    results = {'status': 'success', 'algorithms': {}}
    
    
    if 'aida' in algorithms:
        scores, indices, periods, time_taken = detect_anomalies_aida_intrawindow(
            multi_ts_data, metadata_list, multi_ts_dir, window_size=window_size, overlap=overlap
        )
        if scores is not None:
            files = save_multi_ts_results('aida', scores, periods, time_taken, results_dir)
            results['algorithms']['aida'] = {
                'success': True,
                'anomaly_count': len(indices) if indices is not None else 0,
                'execution_time': time_taken,
                'files': files,
                'windowlevel': 'intrawindow'
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
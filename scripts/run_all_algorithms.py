#!/usr/bin/env python
"""
Script to run all anomaly detection algorithms (AIDA, Isolation Forest, LOF) on S&P 500 data.
"""
import os
import sys
import logging
import argparse
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.isolation_forest import IForest, TemporalIsolationForest
from src.models.lof import LOF, TemporalLOF
from src import config
from src.data.preparation import load_ticker_data
from src.utils.helpers import ensure_directory_exists

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_aida(input_file, output_dir):
    """
    Run AIDA algorithm via the compiled C++ executable.
    
    Args:
        input_file (Path): Path to the input CSV file
        output_dir (Path): Directory to save the output
        
    Returns:
        tuple: (success, output_files)
            success: Boolean indicating if the execution was successful
            output_files: Dictionary with paths to output files
    """
    logger.info("Running AIDA algorithm...")
    
    # Paths to AIDA executable
    aida_cpp_dir = config.ROOT_DIR / "AIDA" / "C++"
    aida_executable = aida_cpp_dir / "build" / "aida_sp500_anomaly_detection"
    
    # Check if executable exists, otherwise try to compile
    if not aida_executable.exists():
        logger.info("AIDA executable not found. Attempting to compile...")
        try:
            os.makedirs(aida_cpp_dir / "build", exist_ok=True)
            
            # Check if we need to create the analysis source file
            src_models_cpp_dir = config.ROOT_DIR / "src" / "models" / "cpp"
            os.makedirs(src_models_cpp_dir, exist_ok=True)
            
            analysis_source_file = src_models_cpp_dir / "aida_sp500_anomaly_detection.cpp"
            
            # If the source file doesn't exist, we need to create it first
            if not analysis_source_file.exists():
                logger.info("Creating AIDA S&P 500 analysis source file...")
                
                # Basic source code for AIDA S&P 500 analysis
                aida_source_code = """/* AIDA Anomaly Detection for S&P 500 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

#include "aida_class.h"

using namespace std;

int main(int argc, char** argv) {
    // Check if input file is provided
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_csv_file>" << endl;
        return 1;
    }

    // Input CSV file path
    string input_file = argv[1];
    
    // Output file paths
    string output_scores_file = input_file + "_AIDA_scores.dat";
    string output_anomalies_file = input_file + "_AIDA_anomalies.dat";

    // Read the CSV file
    ifstream file(input_file);
    if (!file.is_open()) {
        cerr << "Error: Could not open input file " << input_file << endl;
        return 1;
    }

    // Parse CSV header
    string header_line;
    getline(file, header_line);
    
    // Vectors to store data
    vector<vector<double>> numerical_data;
    string line;
    
    // Read numerical data
    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<double> row;
        
        // Skip the first column (likely a date/timestamp)
        getline(ss, cell, ',');
        
        while (getline(ss, cell, ',')) {
            try {
                // Convert string to double
                row.push_back(stod(cell));
            } catch (const std::invalid_argument& e) {
                // Skip non-numeric cells or handle as needed
                row.push_back(0.0);
            }
        }
        
        if (!row.empty()) {
            numerical_data.push_back(row);
        }
    }
    file.close();

    // Prepare data for AIDA
    int n = numerical_data.size();
    int nFnum = numerical_data[0].size();
    int nFnom = 1;  // Nominal features (set to 1 with all zeros)

    cout << "Data loaded: " << n << " rows, " << nFnum << " numerical features" << endl;

    // Allocate memory for numerical and nominal features
    double* Xnum = new double[n * nFnum];
    int* Xnom = new int[n * nFnom];

    // Fill numerical features
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < nFnum; ++j) {
            Xnum[j + i * nFnum] = numerical_data[i][j];
        }
        // Fill nominal features with zeros
        Xnom[i] = 0;
    }

    // AIDA Parameters
    int N = 100;  // Number of subsamples
    string aggregate_type = "aom";
    string score_function = "variance";
    double alpha_min = 1.0;
    double alpha_max = 1.0;
    string distance_metric = "manhattan";

    // Anomaly detection parameters
    int subsample_min = 50;
    int subsample_max = min(512, n);  // Limit to dataset size
    int dmin = min(nFnum, max(2, nFnum / 2));  // At least 2 features
    int dmax = nFnum;

    // Allocate memory for scores
    double* scoresAIDA = new double[n];

    try {
        cout << "Training AIDA..." << endl;
        
        // Train AIDA
        AIDA aida(N, aggregate_type, score_function, alpha_min, alpha_max, distance_metric);
        aida.fit(n, nFnum, Xnum, nFnom, Xnom, subsample_min, subsample_max, dmin, dmax, nFnom, nFnom);
        
        cout << "Computing anomaly scores..." << endl;
        
        // Compute anomaly scores
        aida.score_samples(n, scoresAIDA, Xnum, Xnom);

        // Write scores to file
        ofstream fres(output_scores_file);
        fres << n << endl;
        for (int i = 0; i < n; ++i) {
            fres << scoresAIDA[i] << endl;
        }
        fres.close();

        // Detect anomalies (simple threshold-based approach)
        double mean_score = 0.0;
        double std_score = 0.0;
        
        // Compute mean and standard deviation
        for (int i = 0; i < n; ++i) {
            mean_score += scoresAIDA[i];
        }
        mean_score /= n;
        
        for (int i = 0; i < n; ++i) {
            std_score += (scoresAIDA[i] - mean_score) * (scoresAIDA[i] - mean_score);
        }
        std_score = sqrt(std_score / n);

        // Threshold: 2 standard deviations
        double threshold = mean_score + 2 * std_score;

        // Write anomalies to file
        ofstream fanom(output_anomalies_file);
        int anomaly_count = 0;
        
        fanom << "index,score" << endl;
        for (int i = 0; i < n; ++i) {
            if (scoresAIDA[i] > threshold) {
                fanom << i << "," << scoresAIDA[i] << endl;
                anomaly_count++;
            }
        }
        fanom.close();

        cout << "AIDA Anomaly Detection Complete:" << endl;
        cout << "Total samples: " << n << endl;
        cout << "Anomalies detected: " << anomaly_count << endl;
        cout << "Scores saved to: " << output_scores_file << endl;
        cout << "Anomalies saved to: " << output_anomalies_file << endl;
    }
    catch (const std::exception& e) {
        cerr << "Error during AIDA processing: " << e.what() << endl;
        
        // Clean up
        delete[] Xnum;
        delete[] Xnom;
        delete[] scoresAIDA;
        
        return 1;
    }

    // Clean up
    delete[] Xnum;
    delete[] Xnom;
    delete[] scoresAIDA;

    return 0;
}
"""
                
                # Write the source code to file
                with open(analysis_source_file, "w") as f:
                    f.write(aida_source_code)
                    
                logger.info(f"Created source file at {analysis_source_file}")
                
            # Platform-specific compilation
            if sys.platform == 'darwin':  # macOS
                # For macOS, we need to install OpenMP first
                try:
                    # Check if OpenMP is installed via Homebrew
                    subprocess.run(["brew", "--version"], check=True, capture_output=True)
                    
                    try:
                        # Check if libomp is installed
                        result = subprocess.run(["brew", "list", "libomp"], check=True, capture_output=True)
                        logger.info("OpenMP found via Homebrew")
                    except subprocess.CalledProcessError:
                        # Install OpenMP
                        logger.info("Installing OpenMP via Homebrew...")
                        subprocess.run(["brew", "install", "libomp"], check=True)
                    
                    # Get OpenMP paths
                    libomp_prefix = subprocess.run(["brew", "--prefix", "libomp"], 
                                                check=True, 
                                                capture_output=True, 
                                                text=True).stdout.strip()
                    
                    # macOS compilation with OpenMP support
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
                    logger.warning("Homebrew not found, using basic macOS compilation (OpenMP may not work)")
                    
                    # Basic macOS compilation without OpenMP
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
                # Windows compilation
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
                # Linux compilation
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
        except subprocess.CalledProcessError as e:
            logger.error(f"AIDA compilation failed: {e}")
            return False, {}
        except Exception as e:
            logger.error(f"AIDA compilation error: {e}")
            return False, {}
    
    # Run AIDA executable
    try:
        # Output files will be created with the input filename as prefix
        scores_file = str(input_file) + "_AIDA_scores.dat"
        anomalies_file = str(input_file) + "_AIDA_anomalies.dat"
        
        # Make sure output directory exists
        ensure_directory_exists(output_dir)
        
        # Check if executable exists after compilation attempts
        if not aida_executable.exists():
            logger.error(f"AIDA executable not found at {aida_executable}")
            return False, {}
        
        # Execute AIDA
        logger.info(f"Running AIDA on {input_file}")
        
        # Set environment variables for macOS OpenMP support
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
        
        cmd = [str(aida_executable), str(input_file)]
        result = subprocess.run(cmd, env=env, check=True)
        
        # Copy output files to the output directory
        if Path(scores_file).exists() and Path(anomalies_file).exists():
            # Copy to output directory with standardized names
            import shutil
            output_scores = output_dir / "aida_scores.dat"
            output_anomalies = output_dir / "aida_anomalies.csv"
            
            shutil.copy2(scores_file, output_scores)
            shutil.copy2(anomalies_file, output_anomalies)
            
            logger.info(f"AIDA execution successful. Results saved to {output_dir}")
            return True, {"scores": output_scores, "anomalies": output_anomalies}
        else:
            logger.error("AIDA execution failed: Output files not found")
            return False, {}
            
    except subprocess.CalledProcessError as e:
        logger.error(f"AIDA execution failed: {e}")
        return False, {}
    except Exception as e:
        logger.error(f"AIDA execution error: {e}")
        return False, {}


def run_isolation_forest(data, output_dir):
    """
    Run Isolation Forest algorithm.
    
    Args:
        data (pandas.DataFrame): Input data
        output_dir (Path): Directory to save the output
        
    Returns:
        tuple: (success, output_files)
            success: Boolean indicating if the execution was successful
            output_files: Dictionary with paths to output files
    """
    logger.info("Running Isolation Forest algorithm...")
    
    try:
        # Create output directory
        ensure_directory_exists(output_dir)
        
        # Initialize and run Isolation Forest
        iforest = IForest(
            n_estimators=100,
            max_samples=256,
            contamination=0.05
        )
        
        # Fit and predict
        scores, labels = iforest.fit_predict(data)
        
        # Save results
        scores_file, anomalies_file = iforest.save_results(
            scores,
            labels,
            data.index,
            output_dir,
            prefix="iforest"
        )
        
        # Calculate and save feature importance
        feature_importance = iforest.get_feature_importance(data)
        if feature_importance is not None:
            importance_file = output_dir / "iforest_feature_importance.csv"
            feature_importance.to_csv(importance_file)
            logger.info(f"Saved feature importance to {importance_file}")
            
            # Visualize feature importance
            plt.figure(figsize=(10, 6))
            feature_importance.sort_values().plot(kind='barh')
            plt.title('Isolation Forest Feature Importance')
            plt.tight_layout()
            plt.savefig(output_dir / "iforest_feature_importance.png")
            plt.close()
        
        logger.info(f"Isolation Forest execution successful. Results saved to {output_dir}")
        output_files = {"scores": scores_file, "anomalies": anomalies_file}
        if feature_importance is not None:
            output_files["importance"] = importance_file
            
        return True, output_files
        
    except Exception as e:
        logger.error(f"Isolation Forest execution error: {e}")
        return False, {}


def run_lof(data, output_dir):
    """
    Run Local Outlier Factor algorithm.
    
    Args:
        data (pandas.DataFrame): Input data
        output_dir (Path): Directory to save the output
        
    Returns:
        tuple: (success, output_files)
            success: Boolean indicating if the execution was successful
            output_files: Dictionary with paths to output files
    """
    logger.info("Running Local Outlier Factor algorithm...")
    
    try:
        # Create output directory
        ensure_directory_exists(output_dir)
        
        # Initialize and run LOF
        lof = LOF(
            n_neighbors=20,
            p=2,  # Euclidean distance
            contamination=0.05
        )
        
        # Fit and predict
        scores, labels = lof.fit_predict(data)
        
        # Save results
        scores_file, anomalies_file = lof.save_results(
            scores,
            labels,
            data.index,
            output_dir,
            prefix="lof"
        )
        
        logger.info(f"LOF execution successful. Results saved to {output_dir}")
        return True, {"scores": scores_file, "anomalies": anomalies_file}
        
    except Exception as e:
        logger.error(f"LOF execution error: {e}")
        return False, {}


def run_temporal_algorithms(data, output_dir, window_size=5, step=1):
    """
    Run temporal versions of algorithms (sliding window approach).
    
    Args:
        data (pandas.DataFrame): Input data
        output_dir (Path): Directory to save the output
        window_size (int): Size of sliding window
        step (int): Step size between consecutive windows
        
    Returns:
        dict: Dictionary with results for each algorithm
    """
    logger.info(f"Running temporal algorithms with window_size={window_size}, step={step}...")
    results = {}
    
    try:
        # Create output directory
        temporal_dir = output_dir / f"temporal_w{window_size}_s{step}"
        ensure_directory_exists(temporal_dir)
        
        # Run Temporal Isolation Forest
        try:
            logger.info("Running Temporal Isolation Forest...")
            tiforest = TemporalIsolationForest(
                n_estimators=100,
                max_samples=min(256, window_size * data.shape[1]),
                contamination=0.05,
                window_size=window_size,
                step=step
            )
            
            window_scores, anomaly_windows = tiforest.fit_predict_temporal(data)
            
            scores_file, anomalies_file = tiforest.save_temporal_results(
                window_scores,
                anomaly_windows,
                data,
                temporal_dir,
                prefix="temporal_iforest"
            )
            
            results["tiforest"] = {
                "success": True,
                "scores": scores_file,
                "anomalies": anomalies_file
            }
            
            logger.info("Temporal Isolation Forest completed successfully")
            
        except Exception as e:
            logger.error(f"Temporal Isolation Forest error: {e}")
            results["tiforest"] = {"success": False}
        
        # Run Temporal LOF
        try:
            logger.info("Running Temporal LOF...")
            tlof = TemporalLOF(
                n_neighbors=min(20, window_size - 1),
                p=2,  # Euclidean distance
                contamination=0.05,
                window_size=window_size,
                step=step
            )
            
            window_scores, anomaly_windows = tlof.fit_predict_temporal(data)
            
            scores_file, anomalies_file = tlof.save_temporal_results(
                window_scores,
                anomaly_windows,
                data,
                temporal_dir,
                prefix="temporal_lof"
            )
            
            results["tlof"] = {
                "success": True,
                "scores": scores_file,
                "anomalies": anomalies_file
            }
            
            logger.info("Temporal LOF completed successfully")
            
        except Exception as e:
            logger.error(f"Temporal LOF error: {e}")
            results["tlof"] = {"success": False}
        
        return results
        
    except Exception as e:
        logger.error(f"Error running temporal algorithms: {e}")
        return {"success": False}


def main():
    """
    Main function to run all anomaly detection algorithms.
    """
    parser = argparse.ArgumentParser(description="Run anomaly detection algorithms on S&P 500 data")
    parser.add_argument(
        "--data", 
        type=str, 
        default=str(config.PROCESSED_DATA_DIR / "index_GSPC_processed.csv"),
        help="Path to the processed S&P 500 CSV file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=str(config.DATA_DIR / "algorithm_results"),
        help="Directory to save algorithm results"
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["aida", "iforest", "lof", "temporal"],
        default=["aida", "iforest", "lof", "temporal"],
        help="Algorithms to run"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Window size for temporal algorithms"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Step size for temporal algorithms"
    )
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    input_file = Path(args.data)
    output_dir = Path(args.output)
    
    # Load data
    logger.info(f"Loading data from {input_file}")
    df = load_ticker_data(input_file)
    
    if df is None or df.empty:
        logger.error(f"Failed to load data from {input_file}")
        return
    
    logger.info(f"Loaded data with shape {df.shape}")
    
    # Create output directory
    ensure_directory_exists(output_dir)
    
    # Initialize results dictionary
    results = {}
    
    # Run algorithms
    if "aida" in args.algorithms:
        aida_output_dir = output_dir / "aida"
        success, output_files = run_aida(input_file, aida_output_dir)
        results["aida"] = {"success": success, "files": output_files}
    
    if "iforest" in args.algorithms:
        iforest_output_dir = output_dir / "iforest"
        success, output_files = run_isolation_forest(df, iforest_output_dir)
        results["iforest"] = {"success": success, "files": output_files}
    
    if "lof" in args.algorithms:
        lof_output_dir = output_dir / "lof"
        success, output_files = run_lof(df, lof_output_dir)
        results["lof"] = {"success": success, "files": output_files}
    
    if "temporal" in args.algorithms:
        temporal_output_dir = output_dir / "temporal"
        temporal_results = run_temporal_algorithms(
            df, 
            temporal_output_dir,
            window_size=args.window_size,
            step=args.step
        )
        results["temporal"] = temporal_results
    
    # Print summary
    logger.info("\nAlgorithm execution summary:")
    for algo, result in results.items():
        if algo == "temporal":
            for temporal_algo, temporal_result in result.items():
                success = temporal_result.get("success", False)
                logger.info(f"Temporal {temporal_algo.upper()}: {'Success' if success else 'Failed'}")
        else:
            success = result.get("success", False)
            logger.info(f"{algo.upper()}: {'Success' if success else 'Failed'}")
    
    logger.info(f"All results saved to {output_dir}")


if __name__ == "__main__":
    main()
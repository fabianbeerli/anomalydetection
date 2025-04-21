"""
AIDA helper functions for constituent anomaly detection.
Provides standardized interfaces to integrate AIDA with the constituent analysis pipeline.
"""
import os
import logging
import subprocess
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

from src import config
from src.utils.helpers import ensure_directory_exists

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_aida_on_constituent(ticker, feature_data, output_dir):
    """
    Run AIDA algorithm on constituent stock data.
    
    Args:
        ticker (str): Ticker symbol
        feature_data (pandas.DataFrame): Feature-engineered data for the ticker
        output_dir (Path): Directory to save results
        
    Returns:
        dict: Dictionary with results information
    """
    # Ensure output directory exists
    constituent_dir = output_dir / "aida" / ticker
    ensure_directory_exists(constituent_dir)
    
    # Create temporary input file for AIDA
    with tempfile.NamedTemporaryFile(suffix='.csv', mode='w+', delete=False) as temp_file:
        temp_input_file = Path(temp_file.name)
        
        # Write feature data to CSV
        feature_data.to_csv(temp_file.name, index=True)
        logger.info(f"Saved feature data for {ticker} to temporary file: {temp_input_file}")
    
    try:
        # Paths to AIDA executable and C++ directory
        aida_cpp_dir = config.ROOT_DIR / "AIDA" / "C++"
        aida_executable = aida_cpp_dir / "build" / "aida_constituent"
        
        # Check if executable exists, if not attempt to build
        if not aida_executable.exists():
            logger.info(f"AIDA constituent executable not found. Building...")
            build_aida_constituent_executable(aida_cpp_dir, aida_executable)
        
        # Run AIDA
        logger.info(f"Running AIDA on {ticker}...")
        
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
                logger.info(f"Set DYLD_LIBRARY_PATH to {libomp_prefix}/lib for OpenMP support")
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("Could not set OpenMP library path")
        
        cmd = [str(aida_executable), str(temp_input_file), ticker]
        result = subprocess.run(cmd, env=env, check=True)
        
        # Output files
        scores_file = Path(f"{temp_input_file}_AIDA_scores.dat")
        anomalies_file = Path(f"{temp_input_file}_AIDA_anomalies.csv")
        time_file = Path(f"{temp_input_file}_AIDA_time.txt")
        
        # Check if output files exist
        if scores_file.exists() and anomalies_file.exists():
            # Copy files to output directory
            output_scores = constituent_dir / "aida_scores.dat"
            output_anomalies = constituent_dir / "aida_anomalies.csv"
            
            import shutil
            shutil.copy2(scores_file, output_scores)
            shutil.copy2(anomalies_file, output_anomalies)
            
            # Read execution time
            execution_time = -1
            if time_file.exists():
                with open(time_file, 'r') as f:
                    execution_time = float(f.read().strip())
            
            # Save execution time
            with open(constituent_dir / "aida_execution_time.txt", 'w') as f:
                f.write(f"{execution_time}")
            
            # Get anomaly count
            try:
                anomalies_df = pd.read_csv(output_anomalies)
                anomaly_count = len(anomalies_df)
            except:
                anomaly_count = 0
            
            logger.info(f"AIDA completed for {ticker}. Found {anomaly_count} anomalies.")
            logger.info(f"Execution time: {execution_time:.2f} seconds")
            
            # Cleanup temporary files
            for file in [temp_input_file, scores_file, anomalies_file, time_file]:
                if file.exists():
                    file.unlink()
            
            return {
                "success": True,
                "ticker": ticker,
                "execution_time": execution_time,
                "anomaly_count": anomaly_count,
                "output_dir": constituent_dir
            }
        else:
            logger.error(f"AIDA execution failed for {ticker}: Output files not found")
            return {"success": False, "ticker": ticker}
    
    except subprocess.CalledProcessError as e:
        logger.error(f"AIDA execution failed for {ticker}: {e}")
        return {"success": False, "ticker": ticker, "error": str(e)}
    except Exception as e:
        logger.error(f"Error running AIDA on {ticker}: {e}")
        return {"success": False, "ticker": ticker, "error": str(e)}
    finally:
        # Clean up temporary files
        if temp_input_file.exists():
            temp_input_file.unlink()


def build_aida_constituent_executable(aida_cpp_dir, output_executable):
    """
    Build the AIDA constituent executable.
    
    Args:
        aida_cpp_dir (Path): Path to AIDA C++ directory
        output_executable (Path): Path to output executable
        
    Returns:
        bool: True if build was successful, False otherwise
    """
    try:
        # Create models/cpp directory if it doesn't exist
        models_cpp_dir = config.ROOT_DIR / "src" / "models" / "cpp"
        ensure_directory_exists(models_cpp_dir)
        
        # Create the AIDA constituent source file
        source_file = models_cpp_dir / "aida_constituent.cpp"
        
        with open(source_file, 'w') as f:
            f.write("""/* AIDA Anomaly Detection for Constituent Stocks */
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
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input_csv_file> <ticker>" << endl;
        return 1;
    }
    
    string input_file = argv[1];
    string ticker = argv[2];
    string output_scores_file = input_file + "_AIDA_scores.dat";
    string output_anomalies_file = input_file + "_AIDA_anomalies.csv";
    
    // Read CSV file
    ifstream file(input_file);
    if (!file.is_open()) {
        cerr << "Error: Could not open input file " << input_file << endl;
        return 1;
    }
    
    // Parse header
    string header_line;
    getline(file, header_line);
    stringstream header_stream(header_line);
    string field_name;
    vector<string> field_names;
    
    // Skip first column (date/index)
    getline(header_stream, field_name, ',');
    
    int nFnum = 0;
    while (getline(header_stream, field_name, ',')) {
        field_names.push_back(field_name);
        nFnum++;
    }
    
    // Read numerical data and dates
    vector<vector<double>> numerical_data;
    vector<string> dates;
    string line;
    
    while (getline(file, line)) {
        stringstream ss(line);
        string date;
        getline(ss, date, ',');  // Read date
        dates.push_back(date);
        
        vector<double> row;
        string cell;
        while (getline(ss, cell, ',')) {
            try {
                row.push_back(stod(cell));
            } catch (const std::invalid_argument& e) {
                row.push_back(0.0);  // Handle missing/invalid values
            }
        }
        
        if (!row.empty()) {
            numerical_data.push_back(row);
        }
    }
    file.close();
    
    int n = numerical_data.size();
    int nFnom = 1;  // Single nominal feature (all zeros)
    
    cout << "Data loaded for " << ticker << ": " << n << " samples, " << nFnum << " features" << endl;
    
    // Prepare data for AIDA
    double* Xnum = new double[n * nFnum];
    int* Xnom = new int[n * nFnom];
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < nFnum; ++j) {
            if (j < numerical_data[i].size()) {
                Xnum[j + i * nFnum] = numerical_data[i][j];
            } else {
                Xnum[j + i * nFnum] = 0.0;  // Handle missing columns
            }
        }
        Xnom[i] = 0;  // Single nominal feature (all zeros)
    }
    
    // AIDA Parameters
    int N = 100;  // Number of random subsamples
    string aggregate_type = "aom";
    string score_function = "variance";
    double alpha_min = 1.0;
    double alpha_max = 1.0;
    string distance_metric = "euclidean";
    int subsample_min = 50;
    int subsample_max = min(512, n);
    int dmin = max(2, nFnum / 2);
    int dmax = nFnum;
    
    double* scoresAIDA = new double[n];
    
    try {
        cout << "Training AIDA for " << ticker << "..." << endl;
        
        // Start timing
        auto start_time = high_resolution_clock::now();
        
        // Initialize and run AIDA
        AIDA aida(N, aggregate_type, score_function, alpha_min, alpha_max, distance_metric);
        aida.fit(n, nFnum, Xnum, nFnom, Xnom, subsample_min, subsample_max, dmin, dmax, nFnom, nFnom);
        aida.score_samples(n, scoresAIDA, Xnum, Xnom);
        
        // Stop timing
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time);
        
        // Write scores
        ofstream fres(output_scores_file);
        fres << n << endl;
        for (int i = 0; i < n; ++i) {
            fres << scoresAIDA[i] << endl;
        }
        fres.close();
        
        // Calculate mean and std for anomaly threshold
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
        
        // Write anomalies
        ofstream fanom(output_anomalies_file);
        int anomaly_count = 0;
        
        fanom << "index,date,score,start_date,end_date" << endl;
        for (int i = 0; i < n; ++i) {
            if (scoresAIDA[i] > threshold) {
                fanom << i << "," << dates[i] << "," << scoresAIDA[i] << "," 
                      << dates[i] << "," << dates[i] << endl;
                anomaly_count++;
            }
        }
        fanom.close();
        
        // Save execution time
        ofstream ftime(input_file + "_AIDA_time.txt");
        ftime << duration.count() / 1000.0 << endl;
        ftime.close();
        
        // Report results
        cout << "AIDA analysis complete for " << ticker << ":" << endl;
        cout << "Total samples: " << n << endl;
        cout << "Anomalies detected: " << anomaly_count << endl;
        cout << "Execution time: " << duration.count() / 1000.0 << " seconds" << endl;
        
        // Clean up
        delete[] Xnum;
        delete[] Xnom;
        delete[] scoresAIDA;
        
        return 0;
    } catch (const std::exception& e) {
        cerr << "Error during AIDA processing: " << e.what() << endl;
        
        // Clean up
        delete[] Xnum;
        delete[] Xnom;
        delete[] scoresAIDA;
        
        return 1;
    }
}
""")
        
        logger.info(f"Created AIDA constituent source file: {source_file}")
        
        # Compile the executable
        ensure_directory_exists(output_executable.parent)
        
        # Platform-specific compilation
        if sys.platform == 'darwin':  # macOS
            try:
                # Check if Homebrew is available
                subprocess.run(["brew", "--version"], check=True, capture_output=True)
                
                # Check if OpenMP is installed
                try:
                    result = subprocess.run(["brew", "list", "libomp"], check=True, capture_output=True)
                    logger.info("OpenMP found via Homebrew")
                except subprocess.CalledProcessError:
                    logger.info("Installing OpenMP via Homebrew...")
                    subprocess.run(["brew", "install", "libomp"], check=True)
                
                # Get OpenMP path
                libomp_prefix = subprocess.run(
                    ["brew", "--prefix", "libomp"], 
                    check=True, 
                    capture_output=True, 
                    text=True
                ).stdout.strip()
                
                # Compile with OpenMP support
                compile_cmd = [
                    "g++", "-std=c++11", "-O3", "-Xpreprocessor", "-fopenmp",
                    f"-I{aida_cpp_dir/'include'}",
                    f"-I{libomp_prefix}/include",
                    f"-L{libomp_prefix}/lib",
                    "-lomp",
                    str(source_file),
                    str(aida_cpp_dir/"src"/"aida_class.cpp"),
                    str(aida_cpp_dir/"src"/"distance_metrics.cpp"),
                    str(aida_cpp_dir/"src"/"isolation_formulas.cpp"),
                    str(aida_cpp_dir/"src"/"aggregation_functions.cpp"),
                    str(aida_cpp_dir/"src"/"rng_class.cpp"),
                    "-o", str(output_executable)
                ]
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("Homebrew not found, using basic macOS compilation")
                compile_cmd = [
                    "g++", "-std=c++11", "-O3",
                    f"-I{aida_cpp_dir/'include'}",
                    str(source_file),
                    str(aida_cpp_dir/"src"/"aida_class.cpp"),
                    str(aida_cpp_dir/"src"/"distance_metrics.cpp"),
                    str(aida_cpp_dir/"src"/"isolation_formulas.cpp"),
                    str(aida_cpp_dir/"src"/"aggregation_functions.cpp"),
                    str(aida_cpp_dir/"src"/"rng_class.cpp"),
                    "-o", str(output_executable)
                ]
        elif sys.platform.startswith('win'):  # Windows
            compile_cmd = [
                "g++", "-std=c++11", "-O3", "-fopenmp",
                f"-I{aida_cpp_dir/'include'}",
                str(source_file),
                str(aida_cpp_dir/"src"/"aida_class.cpp"),
                str(aida_cpp_dir/"src"/"distance_metrics.cpp"),
                str(aida_cpp_dir/"src"/"isolation_formulas.cpp"),
                str(aida_cpp_dir/"src"/"aggregation_functions.cpp"),
                str(aida_cpp_dir/"src"/"rng_class.cpp"),
                "-o", str(output_executable)
            ]
        else:  # Linux
            compile_cmd = [
                "g++", "-std=c++11", "-O3", "-fopenmp",
                f"-I{aida_cpp_dir/'include'}",
                str(source_file),
                str(aida_cpp_dir/"src"/"aida_class.cpp"),
                str(aida_cpp_dir/"src"/"distance_metrics.cpp"),
                str(aida_cpp_dir/"src"/"isolation_formulas.cpp"),
                str(aida_cpp_dir/"src"/"aggregation_functions.cpp"),
                str(aida_cpp_dir/"src"/"rng_class.cpp"),
                "-o", str(output_executable)
            ]
        
        # Execute compilation command
        logger.info(f"Compiling AIDA constituent executable with: {' '.join(compile_cmd)}")
        result = subprocess.run(compile_cmd, check=True)
        
        logger.info(f"Successfully built AIDA constituent executable: {output_executable}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to build AIDA constituent executable: {e}")
        return False
    except Exception as e:
        logger.error(f"Error building AIDA constituent executable: {e}")
        return False


class AIDAConstituentAnalyzer:
    """
    Class to analyze constituent stocks using AIDA algorithm.
    Provides a consistent interface for constituent analysis.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the analyzer.
        
        Args:
            output_dir (Path or str, optional): Directory to save results
        """
        self.output_dir = Path(output_dir) if output_dir else Path(config.DATA_DIR) / "constituent_results"
        ensure_directory_exists(self.output_dir)
    
    def analyze_ticker(self, ticker, feature_data):
        """
        Analyze a single ticker using AIDA.
        
        Args:
            ticker (str): Ticker symbol
            feature_data (pandas.DataFrame): Feature-engineered data for the ticker
            
        Returns:
            dict: Results of the analysis
        """
        return run_aida_on_constituent(ticker, feature_data, self.output_dir)
    
    def analyze_multiple_tickers(self, ticker_data_dict):
        """
        Analyze multiple tickers using AIDA.
        
        Args:
            ticker_data_dict (dict): Dictionary mapping tickers to feature data
            
        Returns:
            dict: Results for each ticker
        """
        results = {}
        
        for ticker, data in ticker_data_dict.items():
            logger.info(f"Analyzing {ticker} with AIDA...")
            result = self.analyze_ticker(ticker, data)
            results[ticker] = result
        
        # Save summary
        self._save_summary(results)
        
        return results
    
    def _save_summary(self, results):
        """
        Save a summary of the analysis results.
        
        Args:
            results (dict): Dictionary of analysis results by ticker
        """
        summary_data = []
        
        for ticker, result in results.items():
            summary_data.append({
                'ticker': ticker,
                'success': result.get('success', False),
                'execution_time': result.get('execution_time', -1),
                'anomaly_count': result.get('anomaly_count', 0),
                'error': result.get('error', '')
            })
        
        if summary_data:
            # Create summary DataFrame
            summary_df = pd.DataFrame(summary_data)
            
            # Save to CSV
            summary_file = self.output_dir / "aida" / "analysis_summary.csv"
            ensure_directory_exists(summary_file.parent)
            summary_df.to_csv(summary_file, index=False)
            
            # Save execution time statistics
            successful_results = [r for r in summary_data if r['success']]
            
            if successful_results:
                execution_times = [r['execution_time'] for r in successful_results]
                anomaly_counts = [r['anomaly_count'] for r in successful_results]
                
                stats = {
                    'total_tickers': len(results),
                    'successful_tickers': len(successful_results),
                    'failed_tickers': len(results) - len(successful_results),
                    'total_anomalies': sum(anomaly_counts),
                    'mean_execution_time': np.mean(execution_times),
                    'median_execution_time': np.median(execution_times),
                    'min_execution_time': np.min(execution_times),
                    'max_execution_time': np.max(execution_times),
                    'total_execution_time': sum(execution_times)
                }
                
                # Save statistics to JSON
                stats_file = self.output_dir / "aida" / "execution_statistics.json"
                with open(stats_file, 'w') as f:
                    json.dump(stats, f, indent=2)
                
                logger.info(f"Saved AIDA analysis summary to {summary_file}")
                logger.info(f"Saved AIDA execution statistics to {stats_file}")
                
                logger.info(f"AIDA Analysis Summary:")
                logger.info(f"  Total tickers analyzed: {stats['total_tickers']}")
                logger.info(f"  Successfully analyzed: {stats['successful_tickers']}")
                logger.info(f"  Failed: {stats['failed_tickers']}")
                logger.info(f"  Total anomalies detected: {stats['total_anomalies']}")
                logger.info(f"  Average execution time: {stats['mean_execution_time']:.2f} seconds")
                logger.info(f"  Total execution time: {stats['total_execution_time']:.2f} seconds")
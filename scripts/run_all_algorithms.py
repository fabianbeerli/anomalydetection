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
            # Determine platform for appropriate compilation command
            if sys.platform.startswith('win'):
                # Windows compilation
                compile_cmd = [
                    "g++", "-std=c++11", "-O3", "-fopenmp",
                    f"-I{aida_cpp_dir/'include'}",
                    str(aida_cpp_dir/"src"/"aida_sp500_anomaly_detection.cpp"),
                    str(aida_cpp_dir/"src"/"aida_class.cpp"),
                    str(aida_cpp_dir/"src"/"distance_metrics.cpp"),
                    str(aida_cpp_dir/"src"/"isolation_formulas.cpp"),
                    str(aida_cpp_dir/"src"/"aggregation_functions.cpp"),
                    str(aida_cpp_dir/"src"/"rng_class.cpp"),
                    "-o", str(aida_executable)
                ]
            else:
                # Unix compilation
                compile_cmd = [
                    "g++", "-std=c++11", "-O3", "-fopenmp",
                    f"-I{aida_cpp_dir/'include'}",
                    str(aida_cpp_dir/"src"/"aida_sp500_anomaly_detection.cpp"),
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
        
        # Execute AIDA
        logger.info(f"Running AIDA on {input_file}")
        cmd = [str(aida_executable), str(input_file)]
        result = subprocess.run(cmd, check=True)
        
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
            logger.info(f"{algo.upper()}: {result}")
        else:
            success = result.get("success", False)
            logger.info(f"{algo.upper()}: {'Success' if success else 'Failed'}")
    
    logger.info(f"All results saved to {output_dir}")


if __name__ == "__main__":
    main()
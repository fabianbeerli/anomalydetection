"""
TIX (Tempered Isolation-based eXplanation) helper functions.
Provides interfaces to explain anomalies detected by AIDA using the TIX algorithm.
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
import matplotlib.pyplot as plt
import seaborn as sns

from src import config
from src.utils.helpers import ensure_directory_exists

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_tix_executable(aida_cpp_dir, output_executable):
    """
    Build the TIX executable for feature importance analysis.
    """
    try:
        models_cpp_dir = config.ROOT_DIR / "src" / "models" / "cpp"
        ensure_directory_exists(models_cpp_dir)
        source_file = models_cpp_dir / "tix_analysis.cpp"
        # ... (rest of the function unchanged)
        # [Omitted for brevity, as this part is not changed]
        # ...
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to build TIX executable: {e}")
        return False
    except Exception as e:
        logger.error(f"Error building TIX executable: {e}")
        return False


def run_tix_analysis_for_single_anomaly(data_file, anomaly_index, output_dir):
    """
    Run TIX analysis for a single anomaly in a time series.
    """
    try:
        ensure_directory_exists(output_dir)
        aida_cpp_dir = config.ROOT_DIR / "AIDA" / "C++"
        tix_executable = aida_cpp_dir / "build" / "tix_analysis"
        if not tix_executable.exists():
            logger.info(f"TIX executable not found. Building...")
            build_tix_executable(aida_cpp_dir, tix_executable)
        output_file = output_dir / f"tix_results_point_{anomaly_index}.csv"
        logger.info(f"Running TIX analysis for anomaly index {anomaly_index}...")
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
        cmd = [str(tix_executable), str(data_file), str(anomaly_index), str(output_file)]
        result = subprocess.run(cmd, env=env, check=True)
        if output_file.exists():
            importance_df = pd.read_csv(output_file)
            features = importance_df['feature_name'].tolist()
            scores = importance_df['importance_score'].tolist()
            importance_dict = {feature: score for feature, score in zip(features, scores)}
            importance_df = importance_df.sort_values('importance_score', ascending=False)
            top_features = importance_df.head(min(5, len(importance_df)))
            summary = {
                'total_features': len(features),
                'top_features': top_features['feature_name'].tolist(),
                'top_scores': top_features['importance_score'].tolist(),
                'results_file': str(output_file)
            }
            logger.info(f"TIX analysis completed for anomaly index {anomaly_index}")
            logger.info(f"Top features: {', '.join(summary['top_features'])}")
            return {
                'feature_importance': importance_dict,
                'summary': summary
            }
        else:
            logger.error(f"TIX output file not found: {output_file}")
            return None
    except subprocess.CalledProcessError as e:
        logger.error(f"TIX execution failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Error running TIX analysis: {e}")
        return None


def run_tix_analysis_for_subsequence_anomalies(subsequence_anomalies, subsequence_data_dir, output_dir, algorithm='aida'):
    """
    Run TIX analysis for detected subsequence anomalies.
    """
    try:
        tix_output_dir = output_dir / "tix_analysis"
        ensure_directory_exists(tix_output_dir)
        if subsequence_anomalies.empty:
            logger.warning(f"No {algorithm} anomalies to analyze with TIX")
            return {}
        feature_file = list(Path(subsequence_data_dir).glob("subsequence_features.csv"))
        if not feature_file:
            logger.error(f"Could not find subsequence_features.csv in {subsequence_data_dir}")
            return {}
        feature_file = feature_file[0]
        tix_results = {}
        for idx, anomaly in subsequence_anomalies.iterrows():
            if 'subsequence_idx' not in anomaly:
                logger.warning(f"Missing 'subsequence_idx' in anomaly: {anomaly}")
                continue
            subsequence_idx = int(anomaly['subsequence_idx'])
            row_number = subsequence_idx
            anomaly_output_dir = tix_output_dir / f"{algorithm}_anomaly_{subsequence_idx}"
            ensure_directory_exists(anomaly_output_dir)
            result = run_tix_analysis_for_single_anomaly(
                data_file=feature_file,
                anomaly_index=row_number,
                output_dir=anomaly_output_dir
            )
            if result:
                tix_results[subsequence_idx] = result
                if 'feature_importance' in result:
                    visualize_feature_importance(
                        result['feature_importance'],
                        anomaly_output_dir / f"feature_importance_{subsequence_idx}.png",
                        title=f"{algorithm.upper()} Anomaly {subsequence_idx} - Feature Importance"
                    )
                metadata = {
                    'algorithm': algorithm,
                    'subsequence_idx': subsequence_idx,
                    'anomaly_score': float(anomaly.get('score', float('nan'))),
                    'start_date': str(anomaly.get('start_date', '')),
                    'end_date': str(anomaly.get('end_date', ''))
                }
                with open(anomaly_output_dir / "anomaly_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
        if tix_results:
            summary = {
                'algorithm': algorithm,
                'total_anomalies_analyzed': len(tix_results),
                'anomalies': list(tix_results.keys())
            }
            with open(tix_output_dir / f"{algorithm}_tix_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
        logger.info(f"TIX analysis completed for {len(tix_results)} {algorithm} anomalies")
        return tix_results
    except Exception as e:
        logger.error(f"Error running TIX analysis for subsequence anomalies: {e}")
        return {}


def run_tix_analysis_for_multi_ts_anomalies(multi_ts_anomalies, multi_ts_dir, output_dir):
    """
    Run TIX analysis for detected multi-TS matrix anomalies.
    """
    try:
        tix_output_dir = output_dir / "tix_analysis"
        ensure_directory_exists(tix_output_dir)
        if multi_ts_anomalies.empty:
            logger.warning("No multi-TS anomalies to analyze with TIX")
            return {}
        tix_results = {}
        for idx, anomaly in multi_ts_anomalies.iterrows():
            if 'time_period_idx' not in anomaly:
                logger.warning(f"Missing 'time_period_idx' in anomaly: {anomaly}")
                continue
            time_period_idx = int(anomaly['time_period_idx'])
            multi_ts_files = list(Path(multi_ts_dir).glob(f"*_{time_period_idx}.npz"))
            if not multi_ts_files:
                logger.warning(f"No multi-TS file found for index {time_period_idx}")
                continue
            multi_ts_file = multi_ts_files[0]
            anomaly_output_dir = tix_output_dir / f"multi_ts_anomaly_{time_period_idx}"
            ensure_directory_exists(anomaly_output_dir)
            try:
                npz_data = np.load(multi_ts_file, allow_pickle=True)
                matrix = npz_data['matrix']
                metadata = json.loads(npz_data['metadata'].item())
                tickers = metadata.get('tickers', [f'Stock_{i}' for i in range(matrix.shape[0])])
                features = metadata.get('features', [f'Feature_{i}' for i in range(matrix.shape[2])])
                stock_importance = {}
                for i, ticker in enumerate(tickers):
                    if i >= matrix.shape[0]:
                        continue
                    stock_data = matrix[i, :, :]
                    stock_vector = stock_data.flatten()
                    with tempfile.NamedTemporaryFile(suffix='.csv', mode='w+', delete=False) as temp_file:
                        temp_file_path = Path(temp_file.name)
                        header = ','.join([f'feature_{j}' for j in range(len(stock_vector))])
                        temp_file.write(f"index,{header}\n")
                        temp_file.write(f"{time_period_idx},{','.join([str(val) for val in stock_vector])}\n")
                    stock_output_dir = anomaly_output_dir / ticker
                    ensure_directory_exists(stock_output_dir)
                    result = run_tix_analysis_for_single_anomaly(
                        data_file=temp_file_path,
                        anomaly_index=0,
                        output_dir=stock_output_dir
                    )
                    temp_file_path.unlink()
                    if result:
                        stock_importance[ticker] = result
                if stock_importance:
                    visualize_multi_ts_importance(
                        stock_importance,
                        anomaly_output_dir / f"multi_ts_importance_{time_period_idx}.png",
                        title=f"Multi-TS Anomaly {time_period_idx} - Feature Importance"
                    )
                    tix_results[time_period_idx] = {
                        'stock_importance': stock_importance,
                        'metadata': metadata
                    }
                    anomaly_metadata = {
                        'time_period_idx': time_period_idx,
                        'anomaly_score': float(anomaly.get('score', np.nan)),
                        'start_date': str(anomaly.get('start_date', '')),
                        'end_date': str(anomaly.get('end_date', '')),
                        'stocks_analyzed': len(stock_importance)
                    }
                    with open(anomaly_output_dir / "anomaly_metadata.json", 'w') as f:
                        json.dump(anomaly_metadata, f, indent=2)
            except Exception as e:
                logger.error(f"Error processing multi-TS file {multi_ts_file}: {e}")
                continue
        if tix_results:
            summary = {
                'total_anomalies_analyzed': len(tix_results),
                'anomalies': list(tix_results.keys())
            }
            with open(tix_output_dir / "multi_ts_tix_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
        logger.info(f"TIX analysis completed for {len(tix_results)} multi-TS anomalies")
        return tix_results
    except Exception as e:
        logger.error(f"Error running TIX analysis for multi-TS anomalies: {e}")
        return {}


def visualize_feature_importance(feature_importance, output_file, title="Feature Importance"):
    try:
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, scores = zip(*sorted_features)
        plt.figure(figsize=(10, 6))
        plt.barh(features, scores)
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        logger.info(f"Feature importance visualization saved to {output_file}")
    except Exception as e:
        logger.error(f"Error visualizing feature importance: {e}")


def visualize_multi_ts_importance(stock_importance, output_file, title="Multi-TS Feature Importance"):
    try:
        all_features = set()
        for stock, result in stock_importance.items():
            if 'feature_importance' in result:
                all_features.update(result['feature_importance'].keys())
        top_features = sorted(all_features, key=lambda f: sum(
            result.get('feature_importance', {}).get(f, 0) 
            for result in stock_importance.values()
        ), reverse=True)[:min(10, len(all_features))]
        stocks = list(stock_importance.keys())
        data = np.zeros((len(stocks), len(top_features)))
        for i, stock in enumerate(stocks):
            for j, feature in enumerate(top_features):
                data[i, j] = stock_importance[stock].get('feature_importance', {}).get(feature, 0)
        plt.figure(figsize=(12, 8))
        sns.heatmap(data, annot=True, fmt=".2f", xticklabels=top_features, yticklabels=stocks, cmap="YlGnBu")
        plt.title(title)
        plt.xlabel('Features')
        plt.ylabel('Stocks')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        plt.figure(figsize=(10, 6))
        avg_importance = data.mean(axis=0)
        plt.barh(top_features, avg_importance)
        plt.xlabel('Average Importance Score')
        plt.ylabel('Feature')
        plt.title(f"{title} - Average Across Stocks")
        plt.tight_layout()
        plt.savefig(str(output_file).replace('.png', '_avg.png'))
        plt.close()
        logger.info(f"Multi-TS feature importance visualization saved to {output_file}")
    except Exception as e:
        logger.error(f"Error visualizing multi-TS feature importance: {e}")


def analyze_distance_profiles(anomaly_data, feature_columns, output_dir, title="Distance Profile Plot"):
    try:
        ensure_directory_exists(output_dir)
        feature_data = anomaly_data[feature_columns]
        distance_lists = []
        for n_dim in range(1, len(feature_columns) + 1):
            features_subset = feature_columns[:n_dim]
            distances = np.sum(np.abs(feature_data.values - feature_data.iloc[0].values), axis=1)
            sorted_distances = np.sort(distances)
            normalized_distances = sorted_distances / sorted_distances.max() if sorted_distances.max() > 0 else sorted_distances
            distance_lists.append(normalized_distances)
        plt.figure(figsize=(10, 6))
        plt.boxplot(distance_lists, vert=False, widths=0.7)
        plt.xlabel('Distance Profile (normalized)')
        plt.ylabel('Number of Features')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.yticks(range(1, len(feature_columns) + 1))
        output_file = output_dir / "distance_profile_plot.png"
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        logger.info(f"Distance profile plot saved to {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error creating distance profile plot: {e}")
        return None


class TIXAnalyzer:
    """
    Class to coordinate TIX analysis for different types of anomalies.
    """
    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir) if output_dir else Path(config.DATA_DIR) / "tix_results"
        ensure_directory_exists(self.output_dir)

    def analyze_subsequence_anomalies(self, algorithms=['aida'], window_sizes=[3], overlap_types=['overlap']):
        results = {}
        for algorithm in algorithms:
            for window_size in window_sizes:
                for overlap_type in overlap_types:
                    logger.info(f"Running TIX analysis for {algorithm} subsequence anomalies (w{window_size}, {overlap_type})")
                    subsequence_dir = (
                        Path(config.DATA_DIR)
                        / "analysis_results"
                        / "subsequence_results"
                        / algorithm
                        / f"w{window_size}_{overlap_type}"
                    )
                    subsequence_results_dir = config.DATA_DIR / "analysis_results" / "subsequence_results"
                    anomalies_file = subsequence_results_dir / algorithm / f"w{window_size}_{overlap_type}" / f"{algorithm}_anomalies.csv"
                    if not anomalies_file.exists():
                        logger.warning(f"No anomalies file found at {anomalies_file}")
                        continue
                    anomalies_df = pd.read_csv(anomalies_file)
                    if anomalies_df.empty:
                        logger.warning(f"No anomalies found for {algorithm} (w{window_size}, {overlap_type})")
                        continue
                    tix_output_dir = self.output_dir / "subsequence" / algorithm / f"w{window_size}_{overlap_type}"
                    tix_results = run_tix_analysis_for_subsequence_anomalies(
                        subsequence_anomalies=anomalies_df,
                        subsequence_data_dir=subsequence_dir,
                        output_dir=tix_output_dir,
                        algorithm=algorithm
                    )
                    config_key = f"{algorithm}_w{window_size}_{overlap_type}"
                    results[config_key] = tix_results
        return results

    def analyze_constituent_anomalies(self, algorithms=['aida'], window_sizes=[3], overlap_types=['overlap']):
        """
        Analyze constituent anomalies using TIX for the new folder structure.
        For each anomaly and ticker, match the anomaly to the correct row in the full processed feature file,
        and run TIX on that row.
        """
        results = {}
        for algorithm in algorithms:
            for window_size in window_sizes:
                for overlap_type in overlap_types:
                    constituent_base_dir = (
                        Path(config.DATA_DIR)
                        / "analysis_results"
                        / "constituent_results"
                        / algorithm
                        / f"w{window_size}_{overlap_type}"
                    )
                    if not constituent_base_dir.exists():
                        logger.warning(f"Constituent results directory not found: {constituent_base_dir}")
                        continue
                    for anomaly_dir in constituent_base_dir.glob("anomaly_*"):
                        if not anomaly_dir.is_dir():
                            continue
                        anomaly_index = int(anomaly_dir.name.replace("anomaly_", ""))
                        logger.info(f"Running TIX analysis for constituent anomalies in {anomaly_dir}")
                        for anomaly_file in anomaly_dir.glob("*_anomaly_*.csv"):
                            ticker = anomaly_file.name.split("_anomaly_")[0]
                            constituent_feature_file = (
                                Path(config.PROCESSED_DATA_DIR)
                                / "constituents"
                                / f"{ticker}_processed.csv"
                            )
                            if not constituent_feature_file.exists():
                                logger.warning(f"No processed feature file found for {ticker}")
                                continue

                            # Load the anomaly info to get the correct date/index
                            anomaly_df = pd.read_csv(anomaly_file)
                            if anomaly_df.empty:
                                logger.warning(f"No anomaly info in {anomaly_file}")
                                continue

                            # Try to match by date or index
                            feature_df = pd.read_csv(constituent_feature_file)
                            row_number = None
                            if 'start_date' in anomaly_df.columns and 'date' in feature_df.columns:
                                # Match by date
                                anomaly_date = pd.to_datetime(anomaly_df.iloc[0]['start_date'])
                                feature_df['date'] = pd.to_datetime(feature_df['date'])
                                match = feature_df[feature_df['date'] == anomaly_date]
                                if not match.empty:
                                    row_number = match.index[0]
                            elif 'index' in anomaly_df.columns and 'index' in feature_df.columns:
                                # Match by index
                                anomaly_idx = int(anomaly_df.iloc[0]['index'])
                                match = feature_df[feature_df['index'] == anomaly_idx]
                                if not match.empty:
                                    row_number = match.index[0]
                            else:
                                # Fallback: use anomaly_index as row number
                                row_number = anomaly_index

                            if row_number is None:
                                logger.warning(f"Could not match anomaly to row in features for {ticker} anomaly {anomaly_index}")
                                continue

                            tix_output_dir = (
                                self.output_dir
                                / "constituent"
                                / algorithm
                                / f"w{window_size}_{overlap_type}"
                                / f"anomaly_{anomaly_index}"
                                / ticker
                            )
                            ensure_directory_exists(tix_output_dir)
                            result = run_tix_analysis_for_single_anomaly(
                                data_file=constituent_feature_file,
                                anomaly_index=row_number,
                                output_dir=tix_output_dir
                            )
                            if result:
                                results.setdefault(algorithm, {}).setdefault(f"w{window_size}_{overlap_type}", {}).setdefault(anomaly_index, {})[ticker] = result
                                if 'feature_importance' in result:
                                    visualize_feature_importance(
                                        result['feature_importance'],
                                        tix_output_dir / f"feature_importance_{ticker}_anomaly_{anomaly_index}.png",
                                        title=f"{ticker} Anomaly {anomaly_index} - Feature Importance"
                                    )
                                metadata = {
                                    'ticker': ticker,
                                    'anomaly_index': anomaly_index,
                                    'algorithm': algorithm,
                                    'window_size': window_size,
                                    'mode': overlap_type
                                }
                                with open(tix_output_dir / "anomaly_metadata.json", 'w') as f:
                                    json.dump(metadata, f, indent=2)
        return results

    def analyze_multi_ts_anomalies(self, window_sizes=[3], overlap_types=['overlap']):
        results = {}
        algorithm = 'aida'
        for window_size in window_sizes:
            for overlap_type in overlap_types:
                overlap_str = "overlap" if overlap_type == "overlap" else "nonoverlap"
                logger.info(f"Running TIX analysis for multi-TS anomalies (w{window_size}, {overlap_str})")
                multi_ts_dir = config.PROCESSED_DATA_DIR / "multi_ts"
                multi_ts_results_dir = config.DATA_DIR / "analysis_results" / "multi_ts_results" / f"multi_ts_w{window_size}_{overlap_str}" / algorithm
                anomalies_file = multi_ts_results_dir / f"{algorithm}_multi_ts_anomalies.csv"
                if not multi_ts_dir.exists():
                    logger.warning(f"No multi-TS directory found at {multi_ts_dir}")
                    continue
                if not anomalies_file.exists():
                    logger.warning(f"No anomalies file found at {anomalies_file}")
                    continue
                anomalies_df = pd.read_csv(anomalies_file)
                if anomalies_df.empty:
                    logger.warning(f"No multi-TS anomalies found for w{window_size}_{overlap_str}")
                    continue
                tix_output_dir = self.output_dir / "multi_ts" / f"w{window_size}_{overlap_str}"
                tix_results = run_tix_analysis_for_multi_ts_anomalies(
                    multi_ts_anomalies=anomalies_df,
                    multi_ts_dir=multi_ts_dir,
                    output_dir=tix_output_dir
                )
                config_key = f"w{window_size}_{overlap_str}"
                results[config_key] = tix_results
        return results
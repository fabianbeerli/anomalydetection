#!/usr/bin/env python
"""
Script to visualize feature importance results for LOF and Isolation Forest, similar to TIX-AIDA.
Creates a heatmap for each (ticker, algorithm, window, overlap) in the correct folder structure.
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

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.utils.helpers import ensure_directory_exists

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_feature_importance_summary(feature_importance_dir, ticker, algorithm, window_size, overlap_type):
    summary_file = Path(feature_importance_dir) / "subsequence_feature_importance_summary.json"
    if not summary_file.exists():
        logger.error(f"Feature importance summary file not found: {summary_file}")
        return None
    try:
        with open(summary_file, "r") as f:
            summary = json.load(f)
        key = f"{ticker}_{algorithm}_w{window_size}_{overlap_type}"
        return summary.get(key, None)
    except Exception as e:
        logger.error(f"Error loading feature importance summary: {e}")
        return None

def visualize_feature_importance_heatmap(feature_importance_dir, ticker, algorithm, window_size, overlap_type, output_dir):
    """
    Create a heatmap for all anomalies for a given (ticker, algorithm, window, overlap).
    Also saves a summary JSON with the same basename as the PNG.
    """
    summary = load_feature_importance_summary(
        feature_importance_dir, ticker, algorithm, window_size, overlap_type
    )
    if not summary:
        logger.warning(f"No feature importance summary found for {ticker} with {algorithm}")
        return False
    try:
        anomaly_indices = []
        feature_importances = []
        for idx, result in summary.items():
            if not isinstance(result, dict) or "feature_importance" not in result:
                continue
            anomaly_indices.append(int(idx))
            feature_importances.append(result["feature_importance"])
        if not anomaly_indices:
            logger.warning(f"No anomalies with feature importance results for {ticker} with {algorithm}")
            return False
        all_features = sorted({f for fi in feature_importances for f in fi})
        importance_matrix = np.zeros((len(anomaly_indices), len(all_features)))
        for i, fi in enumerate(feature_importances):
            for j, f in enumerate(all_features):
                importance_matrix[i, j] = fi.get(f, 0)
        # Output folder structure: output_dir/ticker/algorithm/w{window_size}_{overlap_type}/
        output_folder = Path(output_dir) / ticker / algorithm / f"w{window_size}_{overlap_type}"
        ensure_directory_exists(output_folder)
        output_file = output_folder / f"{algorithm}_feature_importance_heatmap_w{window_size}_{overlap_type}.png"

        # --- Save summary JSON ---
        summary_json = {
            "anomalies": [
                {
                    "anomaly_index": anomaly_indices[i],
                    "feature_importances": {
                        feature: float(importance_matrix[i][j])
                        for j, feature in enumerate(all_features)
                    }
                }
                for i in range(len(anomaly_indices))
            ]
        }
        summary_file = output_file.with_suffix('.json')
        with open(summary_file, "w") as f:
            json.dump(summary_json, f, indent=2)
        logger.info(f"Summary JSON saved to {summary_file}")

        # Plot heatmap
        plt.figure(figsize=(min(20, 0.5*len(all_features)), max(8, 0.3*len(anomaly_indices))))
        sns.heatmap(
            importance_matrix,
            annot=False,
            xticklabels=all_features,
            yticklabels=[f"Anomaly #{idx}" for idx in anomaly_indices],
            cmap="YlGnBu"
        )
        plt.title(f"{ticker} {algorithm.upper()} Feature Importance Heatmap (w{window_size}, {overlap_type})")
        plt.xlabel("Features")
        plt.ylabel("Anomalies")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Feature importance heatmap saved to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error creating feature importance heatmap: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Visualize feature importance results for LOF and Isolation Forest")
    parser.add_argument(
        "--feature-importance-dir", 
        type=str, 
        default=str(config.DATA_DIR / "analysis_results" / "feature_importance_results"),
        help="Directory containing feature importance results"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=str(config.DATA_DIR / "analysis_results" / "feature_importance_visualization"),
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--algorithms", 
        type=str, 
        default="iforest,lof",
        help="Comma-separated list of algorithms to visualize"
    )
    parser.add_argument(
        "--window-sizes", 
        type=str,
        default="3",
        help="Comma-separated list of window sizes"
    )
    parser.add_argument(
        "--only-overlap", 
        action="store_true",
        help="Only visualize overlapping subsequences"
    )
    parser.add_argument(
        "--only-nonoverlap", 
        action="store_true",
        help="Only visualize non-overlapping subsequences"
    )
    parser.add_argument(
        "--visualize-all", 
        action="store_true",
        help="Visualize all available configurations"
    )
    parser.add_argument(
        "--ticker", 
        type=str, 
        default=None,
        help="Ticker to visualize (if not set, visualize all found)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    ensure_directory_exists(args.output_dir)
    algorithms = args.algorithms.split(',')
    window_sizes = [int(size) for size in args.window_sizes.split(',')]
    overlap_types = []
    if args.only_overlap:
        overlap_types = ['overlap']
    elif args.only_nonoverlap:
        overlap_types = ['nonoverlap']
    else:
        overlap_types = ['overlap', 'nonoverlap']
    feature_importance_dir = Path(args.feature_importance_dir)
    # If visualize-all is specified, find all available configurations
    if args.visualize_all:
        summary_file = feature_importance_dir / "subsequence_feature_importance_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, "r") as f:
                    summary = json.load(f)
                configs = []
                for key in summary.keys():
                    parts = key.split('_')
                    if len(parts) >= 4:
                        ticker = parts[0]
                        algorithm = parts[1]
                        window_part = parts[2]
                        overlap_part = parts[3]
                        if window_part.startswith('w') and window_part[1:].isdigit():
                            window_size = int(window_part[1:])
                            configs.append((ticker, algorithm, window_size, overlap_part))
                if configs:
                    tickers = list(set(c[0] for c in configs))
                    algorithms = list(set(c[1] for c in configs))
                    window_sizes = list(set(c[2] for c in configs))
                    overlap_types = list(set(c[3] for c in configs))
                    logger.info(f"Found {len(configs)} configurations to visualize")
                    logger.info(f"Tickers: {tickers}")
                    logger.info(f"Algorithms: {algorithms}")
                    logger.info(f"Window sizes: {window_sizes}")
                    logger.info(f"Overlap types: {overlap_types}")
                    for ticker in tickers:
                        for algorithm in algorithms:
                            for window_size in window_sizes:
                                for overlap_type in overlap_types:
                                    if (ticker, algorithm, window_size, overlap_type) in configs:
                                        logger.info(f"Visualizing {ticker} with {algorithm} (w{window_size}, {overlap_type})")
                                        visualize_feature_importance_heatmap(
                                            feature_importance_dir=feature_importance_dir,
                                            ticker=ticker,
                                            algorithm=algorithm,
                                            window_size=window_size,
                                            overlap_type=overlap_type,
                                            output_dir=Path(args.output_dir)
                                        )
                    logger.info("All visualizations completed")
                    return
            except Exception as e:
                logger.error(f"Error loading feature importance summary: {e}")
    # Create visualizations based on specified parameters
    tickers = []
    if args.ticker:
        tickers = [args.ticker]
    else:
        # Try to infer tickers from the summary file
        summary_file = feature_importance_dir / "subsequence_feature_importance_summary.json"
        if summary_file.exists():
            with open(summary_file, "r") as f:
                summary = json.load(f)
            tickers = list(set(key.split('_')[0] for key in summary.keys()))
    for ticker in tickers:
        for algorithm in algorithms:
            for window_size in window_sizes:
                for overlap_type in overlap_types:
                    logger.info(f"Visualizing {ticker} with {algorithm} (w{window_size}, {overlap_type})")
                    visualize_feature_importance_heatmap(
                        feature_importance_dir=feature_importance_dir,
                        ticker=ticker,
                        algorithm=algorithm,
                        window_size=window_size,
                        overlap_type=overlap_type,
                        output_dir=Path(args.output_dir)
                    )
    logger.info("All visualizations completed")

if __name__ == "__main__":
    main()
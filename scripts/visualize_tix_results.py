#!/usr/bin/env python
"""
Script to visualize TIX analysis results.
Creates comprehensive visualizations that combine anomaly detection and feature importance.
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
from matplotlib.dates import DateFormatter
import matplotlib.gridspec as gridspec

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

def load_tix_summary(tix_results_dir, ticker, algorithm, window_size, overlap_type):
    summary_file = Path(tix_results_dir) / "subsequence_tix_summary.json"
    if not summary_file.exists():
        logger.error(f"TIX summary file not found: {summary_file}")
        return None
    with open(summary_file, "r") as f:
        summary = json.load(f)
    key = f"{ticker}_{algorithm}_w{window_size}_{overlap_type}"
    return summary.get(key, None)

def load_price_data(data_file):
    try:
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        if df is None or df.empty:
            logger.error(f"Failed to load data from {data_file}")
            return None
        if 'Close' not in df.columns and 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        return df
    except Exception as e:
        logger.error(f"Error loading price data: {e}")
        return None

def visualize_multi_ts_tix_per_anomaly_from_csv(multi_ts_dir, title_prefix="Multi-TS TIX Feature Importance"):
    """
    For each anomaly in multi-TS TIX results (CSV), create a single-row heatmap of feature importances.
    """
    multi_ts_dir = Path(multi_ts_dir)
    for anomaly_dir in multi_ts_dir.glob("anomaly_*_*"):
        tix_csv = anomaly_dir / "tix_results_point_0.csv"
        if not tix_csv.exists():
            logger.warning(f"TIX CSV not found: {tix_csv}")
            continue
        # Load feature importances
        df = pd.read_csv(tix_csv)
        feature_names = df['feature_name'].tolist()
        importance_scores = df['importance_score'].tolist()
        ticker = anomaly_dir.name.split("_")[-1]
        window_idx = anomaly_dir.name.split("_")[1]
        # Prepare data for single anomaly
        data = np.array(importance_scores).reshape(1, -1)
        output_file = anomaly_dir / f"feature_importance_heatmap_{ticker}_w{window_idx}_0.png"
        plt.figure(figsize=(max(8, 0.3*len(feature_names)), 2))
        sns.heatmap(
            data,
            annot=False,
            xticklabels=feature_names,
            yticklabels=[f"{ticker}_w{window_idx}"],
            cmap="YlGnBu"
        )
        plt.title(f"{title_prefix} ({ticker} w{window_idx})")
        plt.xlabel("Features")
        plt.ylabel("Anomaly")
        plt.tight_layout()
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"Per-anomaly heatmap saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Visualize TIX analysis results")
    parser.add_argument(
        "--sp500-data", 
        type=str, 
        default=str(config.PROCESSED_DATA_DIR / "index_GSPC_processed.csv"),
        help="Path to S&P 500 processed data"
    )
    parser.add_argument(
        "--visualize-multits",
        action="store_true",
        help="Visualize multi-TS TIX results"
    )
    parser.add_argument(
        "--visualize-subsequence", 
        action="store_true",
        help="Visualize subsequence TIX results"
    )
    parser.add_argument(
        "--visualize-all", 
        action="store_true",
        help="Visualize all TIX results"
    )
    parser.add_argument(
        "--window-size", 
        type=int, 
        default=3,
        help="Window size for subsequence visualization"
    )
    parser.add_argument(
        "--overlap-type", 
        choices=["overlap", "nonoverlap"],
        default="overlap",
        help="Overlap type for subsequence visualization"
    )
    parser.add_argument(
        "--algorithm", 
        type=str, 
        default="aida",
        help="Algorithm for subsequence visualization"
    )
    parser.add_argument(
        "--tix-results-dir", 
        type=str, 
        default=str(config.DATA_DIR / "analysis_results" / "tix_results"),
        help="Directory containing TIX results"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=str(config.DATA_DIR / "analysis_results" / "tix_visualizations"),
        help="Directory to save visualizations"
    )
    args = parser.parse_args()
    ensure_directory_exists(args.output_dir)
    visualizations_to_create = []
    if args.visualize_all:
        visualizations_to_create = ["subsequence"]
    else:
        if args.visualize_subsequence:
            visualizations_to_create.append("subsequence")
    if not visualizations_to_create and not (args.visualize_all or args.visualize_multits):
        logger.warning("No visualizations selected. Use --visualize-all or specific visualization flags.")
        return
    
    if "subsequence" in visualizations_to_create:
        logger.info("= Creating Subsequence TIX Visualizations =")
        subsequence_results_dir = Path(config.DATA_DIR) / "analysis_results" / "subsequence_results"
        tix_results_dir = Path(args.tix_results_dir)
        subsequence_output_dir = Path(args.output_dir) / "subsequence"
        ensure_directory_exists(subsequence_output_dir)
        for ticker_dir in (tix_results_dir / "subsequence").iterdir():
            if not ticker_dir.is_dir():
                continue
            ticker = ticker_dir.name
            logger.info(f"Visualizing TIX results for ticker: {ticker}")

            # Load TIX summary
            tix_summary = load_tix_summary(tix_results_dir, ticker, args.algorithm, args.window_size, args.overlap_type)
            if not tix_summary:
                logger.warning(f"No TIX summary for {ticker}")
                continue

            # Build anomalies DataFrame and feature importance matrix
            anomaly_indices = []
            feature_importances = []
            for idx, result in tix_summary.items():
                if not isinstance(result, dict) or "feature_importance" not in result:
                    continue
                anomaly_indices.append(int(idx))
                feature_importances.append(result["feature_importance"])
            if not anomaly_indices:
                logger.warning(f"No anomalies with TIX results for {ticker}")
                continue

            # Build DataFrame for heatmap
            all_features = sorted({f for fi in feature_importances for f in fi})
            importance_matrix = np.zeros((len(anomaly_indices), len(all_features)))
            for i, fi in enumerate(feature_importances):
                for j, f in enumerate(all_features):
                    importance_matrix[i, j] = fi.get(f, 0)

            # Plot only the heatmap
            plt.figure(figsize=(min(20, 0.5*len(anomaly_indices)), max(8, 0.3*len(all_features))))
            sns.heatmap(
                importance_matrix,
                annot=False,
                xticklabels=all_features,
                yticklabels=[f"Anomaly #{idx}" for idx in anomaly_indices],
                cmap="YlGnBu"
            )
            plt.title(f"{ticker} {args.algorithm.upper()} Feature Importance Heatmap (w{args.window_size}, {args.overlap_type})")
            plt.xlabel("Features")
            plt.ylabel("Anomalies")
            output_file = subsequence_output_dir / ticker / f"{args.algorithm}_tix_heatmap_w{args.window_size}_{args.overlap_type}.png"
            ensure_directory_exists(output_file.parent)
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Visualization saved to {output_file}")
    # Multi-TS TIX visualization block
    if args.visualize_all or args.visualize_multits:
        for multi_ts_key in ["multi_ts_w3_overlap", "multi_ts_w3_nonoverlap"]:
            multi_ts_dir = Path(args.tix_results_dir) / multi_ts_key
            if not multi_ts_dir.exists():
                logger.warning(f"Multi-TS TIX directory not found: {multi_ts_dir}")
                continue
            visualize_multi_ts_tix_per_anomaly_from_csv(
                multi_ts_dir=multi_ts_dir,
                title_prefix=f"Multi-TS TIX Feature Importance"
            )
    logger.info(f"All visualizations completed and saved to {args.output_dir}")


if __name__ == "__main__":
    main()
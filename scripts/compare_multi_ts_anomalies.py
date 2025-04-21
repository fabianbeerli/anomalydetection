#!/usr/bin/env python
"""
Compare and visualize anomaly detection results for multi-TS analysis.
Supports selecting specific window sizes and overlap settings.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ALGORITHMS = ['aida', 'iforest', 'lof']

def load_multi_ts_anomaly_results(results_dir, window_size, overlap_type):
    """
    Load multi-TS anomaly results for different algorithms.
    """
    anomaly_results = {}
    overlap_str = "overlap" if overlap_type == "overlap" else "nonoverlap"
    config_name = f"multi_ts_w{window_size}_{overlap_str}"

    for algo in ALGORITHMS:
        algo_dir = Path(results_dir) / config_name / algo
        anomalies_file = algo_dir / f"{algo}_multi_ts_anomalies.csv"
        if anomalies_file.exists():
            anomalies_df = pd.read_csv(anomalies_file)
            # Convert date columns to datetime
            for date_col in ['start_date', 'end_date']:
                if date_col in anomalies_df.columns:
                    anomalies_df[date_col] = pd.to_datetime(anomalies_df[date_col])
            anomaly_results[algo] = anomalies_df
            logger.info(f"Loaded {len(anomalies_df)} anomalies for {algo}")
        else:
            logger.warning(f"No anomalies file found for {algo} at {anomalies_file}")
    return anomaly_results

def visualize_multi_ts_anomalies(anomaly_results, window_size, overlap_type, output_dir):
    """
    Visualize multi-TS anomaly detection results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overlap_str = "overlap" if overlap_type == "overlap" else "nonoverlap"

    for algo, anomalies in anomaly_results.items():
        if anomalies.empty:
            logger.warning(f"No anomalies to plot for {algo}")
            continue

        plt.figure(figsize=(14, 6))
        plt.title(f"Multi-TS Anomalies ({algo.upper()}, w{window_size}, {overlap_str})")
        plt.plot(anomalies['score'], marker='o', linestyle='-', label='Anomaly Score')
        for idx, row in anomalies.iterrows():
            plt.text(idx, row['score'], str(row.get('top_tickers', '')), fontsize=7, rotation=45)
        plt.xlabel("Anomaly Index")
        plt.ylabel("Anomaly Score")
        plt.legend()
        plt.tight_layout()
        plot_file = output_dir / f"{algo}_multi_ts_anomalies_w{window_size}_{overlap_str}.png"
        plt.savefig(plot_file)
        plt.close()
        logger.info(f"Saved plot to {plot_file}")

def main():
    parser = argparse.ArgumentParser(description="Compare multi-TS anomaly detection results")
    parser.add_argument(
        "--results-base", 
        type=str, 
        default="data/analysis_results/multi_ts_results",
        help="Base directory containing all multi-TS results"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/analysis_results/multi_ts_analysis",
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=3,
        help="Window size to analyze"
    )
    parser.add_argument(
        "--overlap-type",
        choices=["overlap", "nonoverlap"],
        default="overlap",
        help="Overlap type to analyze"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_base)
    output_dir = Path(args.output)
    window_size = args.window_size
    overlap_type = args.overlap_type

    anomaly_results = load_multi_ts_anomaly_results(results_dir, window_size, overlap_type)
    visualize_multi_ts_anomalies(anomaly_results, window_size, overlap_type, output_dir)

if __name__ == "__main__":
    main()
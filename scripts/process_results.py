#!/usr/bin/env python
"""
Script to process and visualize results from all anomaly detection algorithms.
"""
import os
import sys
import logging
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2, venn3

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.data.preparation import load_ticker_data
from src.utils.helpers import ensure_directory_exists

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_algorithm_scores(results_dir, algorithm):
    """
    Load anomaly scores for a specific algorithm.
    
    Args:
        results_dir (Path): Directory containing algorithm results
        algorithm (str): Algorithm name (aida, iforest, lof)
        
    Returns:
        pandas.Series: Series of anomaly scores or None if loading fails
    """
    try:
        algo_dir = results_dir / algorithm
        
        if algorithm == "aida":
            scores_file = algo_dir / "aida_scores.dat"
            # AIDA scores are in a specific format with count in the first line
            with open(scores_file, 'r') as f:
                count = int(f.readline().strip())
                scores = [float(f.readline().strip()) for _ in range(count)]
            return pd.Series(scores, name=algorithm.upper())
            
        else:  # iforest or lof
            scores_file = algo_dir / f"{algorithm}_scores.dat"
            # First line is count, rest are scores
            with open(scores_file, 'r') as f:
                count = int(f.readline().strip())
                scores = [float(f.readline().strip()) for _ in range(count)]
            return pd.Series(scores, name=algorithm.upper())
            
    except Exception as e:
        logger.error(f"Error loading {algorithm} scores: {e}")
        return None


def load_algorithm_anomalies(results_dir, algorithm):
    """
    Load detected anomalies for a specific algorithm.
    
    Args:
        results_dir (Path): Directory containing algorithm results
        algorithm (str): Algorithm name (aida, iforest, lof)
        
    Returns:
        pandas.DataFrame: DataFrame of anomalies or None if loading fails
    """
    try:
        algo_dir = results_dir / algorithm
        anomalies_file = algo_dir / f"{algorithm}_anomalies.csv"
        
        if not anomalies_file.exists():
            # Try alternative filename for AIDA
            if algorithm == "aida":
                anomalies_file = algo_dir / "aida_anomalies.csv"
        
        # Load anomalies
        anomalies = pd.read_csv(anomalies_file)
        return anomalies
            
    except Exception as e:
        logger.error(f"Error loading {algorithm} anomalies: {e}")
        return None


def load_temporal_results(results_dir, window_size=5, step=1):
    """
    Load temporal algorithm results.
    
    Args:
        results_dir (Path): Directory containing algorithm results
        window_size (int): Window size used for temporal analysis
        step (int): Step size used for temporal analysis
        
    Returns:
        dict: Dictionary of results for each temporal algorithm
    """
    try:
        temporal_dir = results_dir / "temporal" / f"temporal_w{window_size}_s{step}"
        results = {}
        
        # Load Temporal Isolation Forest results
        tiforest_scores = temporal_dir / "temporal_iforest_window_scores.csv"
        tiforest_anomalies = temporal_dir / "temporal_iforest_window_anomalies.csv"
        
        if tiforest_scores.exists() and tiforest_anomalies.exists():
            results["tiforest"] = {
                "scores": pd.read_csv(tiforest_scores),
                "anomalies": pd.read_csv(tiforest_anomalies)
            }
        
        # Load Temporal LOF results
        tlof_scores = temporal_dir / "temporal_lof_window_scores.csv"
        tlof_anomalies = temporal_dir / "temporal_lof_window_anomalies.csv"
        
        if tlof_scores.exists() and tlof_anomalies.exists():
            results["tlof"] = {
                "scores": pd.read_csv(tlof_scores),
                "anomalies": pd.read_csv(tlof_anomalies)
            }
            
        return results
            
    except Exception as e:
        logger.error(f"Error loading temporal results: {e}")
        return {}


def create_score_comparison(scores_dict, output_dir):
    """
    Create comparison visualizations for anomaly scores from different algorithms.
    
    Args:
        scores_dict (dict): Dictionary mapping algorithm names to score Series
        output_dir (Path): Directory to save visualizations
    """
    try:
        # Ensure output directory exists
        ensure_directory_exists(output_dir)
        
        # Combine scores into a DataFrame
        scores_df = pd.DataFrame(scores_dict)
        
        # Calculate score statistics
        stats_df = scores_df.describe()
        stats_file = output_dir / "score_statistics.csv"
        stats_df.to_csv(stats_file)
        logger.info(f"Score statistics saved to {stats_file}")
        
        # Histogram of scores for each algorithm
        plt.figure(figsize=(12, 6))
        for algo, scores in scores_dict.items():
            plt.hist(scores, bins=50, alpha=0.5, label=algo)
        plt.title('Distribution of Anomaly Scores by Algorithm')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        hist_file = output_dir / "score_distributions.png"
        plt.savefig(hist_file)
        plt.close()
        
        # Box plot of scores
        plt.figure(figsize=(10, 6))
        scores_df.boxplot()
        plt.title('Boxplot of Anomaly Scores by Algorithm')
        plt.ylabel('Anomaly Score')
        plt.grid(False)
        plt.tight_layout()
        box_file = output_dir / "score_boxplots.png"
        plt.savefig(box_file)
        plt.close()
        
        # Score correlation matrix
        plt.figure(figsize=(8, 6))
        corr = scores_df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix of Anomaly Scores')
        plt.tight_layout()
        corr_file = output_dir / "score_correlations.png"
        plt.savefig(corr_file)
        plt.close()
        
        # Create time series plot if provided with time index
        # Would require original data with index
        
        return {
            "statistics": stats_file,
            "histogram": hist_file,
            "boxplot": box_file,
            "correlation": corr_file
        }
        
    except Exception as e:
        logger.error(f"Error creating score comparison: {e}")
        return {}


def analyze_anomaly_overlap(anomalies_dict, data_index, output_dir):
    """
    Analyze overlap between anomalies detected by different algorithms.
    
    Args:
        anomalies_dict (dict): Dictionary mapping algorithm names to anomaly DataFrames
        data_index (pandas.Index): Index of the original data (for time information)
        output_dir (Path): Directory to save analysis results
    """
    try:
        # Ensure output directory exists
        ensure_directory_exists(output_dir)
        
        # Extract indices of anomalies for each algorithm
        algo_anomalies = {}
        for algo, anomalies_df in anomalies_dict.items():
            if 'index' in anomalies_df.columns:
                algo_anomalies[algo] = set(anomalies_df['index'].astype(int))
            else:
                logger.warning(f"No 'index' column found in {algo} anomalies")
        
        if not algo_anomalies:
            logger.error("No valid anomaly indices found for any algorithm")
            return {}
        
        # Calculate overlap statistics
        overlap_stats = {}
        algos = list(algo_anomalies.keys())
        
        # Pairwise overlap
        for i, algo1 in enumerate(algos):
            for algo2 in algos[i+1:]:
                intersection = algo_anomalies[algo1].intersection(algo_anomalies[algo2])
                union = algo_anomalies[algo1].union(algo_anomalies[algo2])
                
                if union:  # Avoid division by zero
                    jaccard = len(intersection) / len(union)
                else:
                    jaccard = 0
                    
                overlap_stats[f"{algo1}_{algo2}_intersection"] = len(intersection)
                overlap_stats[f"{algo1}_{algo2}_union"] = len(union)
                overlap_stats[f"{algo1}_{algo2}_jaccard"] = jaccard
        
        # Common anomalies across all algorithms
        if len(algos) > 1:
            common_across_all = set.intersection(*[algo_anomalies[algo] for algo in algos])
            overlap_stats["common_across_all"] = len(common_across_all)
            overlap_stats["common_across_all_indices"] = list(common_across_all)
        
        # Save overlap statistics
        overlap_df = pd.DataFrame([overlap_stats])
        overlap_file = output_dir / "anomaly_overlap_statistics.csv"
        overlap_df.to_csv(overlap_file, index=False)
        
        # Create a Venn diagram for visualization (if matplotlib_venn is installed)
        if len(algos) == 2:
            plt.figure(figsize=(8, 6))
            venn2([algo_anomalies[algos[0]], algo_anomalies[algos[1]]], 
                 set_labels=algos)
            plt.title('Overlap of Anomalies Detected by Different Algorithms')
            plt.tight_layout()
            venn_file = output_dir / "anomaly_overlap_venn2.png"
            plt.savefig(venn_file)
            plt.close()
        
        elif len(algos) == 3:
            plt.figure(figsize=(8, 6))
            venn3([algo_anomalies[algos[0]], algo_anomalies[algos[1]], algo_anomalies[algos[2]]], 
                 set_labels=algos)
            plt.title('Overlap of Anomalies Detected by Different Algorithms')
            plt.tight_layout()
            venn_file = output_dir / "anomaly_overlap_venn3.png"
            plt.savefig(venn_file)
            plt.close()
            
        # Create a detailed table of all detected anomalies with which algorithm found them
        all_anomalies = set().union(*[anomalies for anomalies in algo_anomalies.values()])
        anomaly_table = []
        
        for idx in sorted(all_anomalies):
            # Extract timestamp if available
            if data_index is not None and 0 <= idx < len(data_index):
                timestamp = data_index[idx]
            else:
                timestamp = None
                
            # Create record for this anomaly
            record = {
                'index': idx,
                'timestamp': timestamp
            }
            
            # Add detection status for each algorithm
            for algo in algos:
                record[f'detected_by_{algo}'] = int(idx in algo_anomalies[algo])
                
                # Add score if available
                for anom_df in anomalies_dict.values():
                    if 'index' in anom_df.columns and 'score' in anom_df.columns:
                        matching_rows = anom_df[anom_df['index'] == idx]
                        if not matching_rows.empty:
                            record[f'{algo}_score'] = matching_rows.iloc[0]['score']
            
            anomaly_table.append(record)
        
        # Save the detailed table
        if anomaly_table:
            anomaly_df = pd.DataFrame(anomaly_table)
            detail_file = output_dir / "anomaly_detection_details.csv"
            anomaly_df.to_csv(detail_file, index=False)
            logger.info(f"Detailed anomaly information saved to {detail_file}")
        
        return {
            "statistics": overlap_file,
            "details": detail_file if anomaly_table else None
        }
        
    except Exception as e:
        logger.error(f"Error analyzing anomaly overlap: {e}")
        return {}


def analyze_temporal_results(temporal_results, output_dir):
    """
    Analyze and visualize temporal anomaly detection results.
    
    Args:
        temporal_results (dict): Dictionary of temporal algorithm results
        output_dir (Path): Directory to save analysis results
    """
    try:
        # Ensure output directory exists
        ensure_directory_exists(output_dir)
        
        # Process each algorithm's temporal results
        for algo, results in temporal_results.items():
            if 'scores' not in results or 'anomalies' not in results:
                logger.warning(f"Missing data in temporal results for {algo}")
                continue
                
            scores_df = results['scores']
            anomalies_df = results['anomalies']
            
            # Create timeline visualization of scores with anomalies highlighted
            if 'start_date' in scores_df.columns and 'score' in scores_df.columns:
                plt.figure(figsize=(15, 6))
                
                # Plot all scores
                plt.plot(pd.to_datetime(scores_df['start_date']), scores_df['score'], label='Scores', alpha=0.7)
                
                # Highlight anomalies
                if not anomalies_df.empty and 'start_date' in anomalies_df.columns and 'score' in anomalies_df.columns:
                    plt.scatter(pd.to_datetime(anomalies_df['start_date']), anomalies_df['score'], 
                               color='red', s=50, label='Anomalies')
                
                plt.title(f'Temporal {algo.upper()} Scores with Detected Anomalies')
                plt.xlabel('Date')
                plt.ylabel('Anomaly Score')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Rotate date labels for readability
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save the figure
                timeline_file = output_dir / f"{algo}_temporal_timeline.png"
                plt.savefig(timeline_file)
                plt.close()
                
                logger.info(f"Temporal timeline for {algo} saved to {timeline_file}")
            
            # Save statistics about the windows and anomalies
            stats = {
                'total_windows': len(scores_df),
                'anomalous_windows': len(anomalies_df),
                'anomaly_rate': len(anomalies_df) / len(scores_df) if len(scores_df) > 0 else 0,
                'mean_score': scores_df['score'].mean() if 'score' in scores_df.columns else None,
                'std_score': scores_df['score'].std() if 'score' in scores_df.columns else None,
                'min_score': scores_df['score'].min() if 'score' in scores_df.columns else None,
                'max_score': scores_df['score'].max() if 'score' in scores_df.columns else None
            }
            
            stats_df = pd.DataFrame([stats])
            stats_file = output_dir / f"{algo}_temporal_statistics.csv"
            stats_df.to_csv(stats_file, index=False)
            
            logger.info(f"Temporal statistics for {algo} saved to {stats_file}")
        
        # If we have results from multiple algorithms, compare them
        if len(temporal_results) > 1:
            # Compare anomaly detection rates
            algo_comparison = []
            for algo, results in temporal_results.items():
                if 'scores' in results and 'anomalies' in results:
                    anomaly_rate = len(results['anomalies']) / len(results['scores']) if len(results['scores']) > 0 else 0
                    algo_comparison.append({
                        'algorithm': algo,
                        'total_windows': len(results['scores']),
                        'anomalous_windows': len(results['anomalies']),
                        'anomaly_rate': anomaly_rate
                    })
            
            if algo_comparison:
                comparison_df = pd.DataFrame(algo_comparison)
                comparison_file = output_dir / "temporal_algorithm_comparison.csv"
                comparison_df.to_csv(comparison_file, index=False)
                logger.info(f"Temporal algorithm comparison saved to {comparison_file}")
                
                # Create bar chart of anomaly rates
                plt.figure(figsize=(10, 6))
                plt.bar(comparison_df['algorithm'], comparison_df['anomaly_rate'])
                plt.title('Anomaly Detection Rate by Temporal Algorithm')
                plt.xlabel('Algorithm')
                plt.ylabel('Anomaly Rate')
                plt.tight_layout()
                rate_chart_file = output_dir / "temporal_anomaly_rates.png"
                plt.savefig(rate_chart_file)
                plt.close()
                
            # Analyze overlap in detected anomalies
            if len(temporal_results) >= 2:
                algos = list(temporal_results.keys())
                window_overlap = {}
                
                for i, algo1 in enumerate(algos):
                    for j, algo2 in enumerate(algos[i+1:], i+1):
                        if 'anomalies' in temporal_results[algo1] and 'anomalies' in temporal_results[algo2]:
                            # Extract window start indices
                            if 'window_start' in temporal_results[algo1]['anomalies'].columns and 'window_start' in temporal_results[algo2]['anomalies'].columns:
                                windows1 = set(temporal_results[algo1]['anomalies']['window_start'])
                                windows2 = set(temporal_results[algo2]['anomalies']['window_start'])
                                
                                intersection = windows1.intersection(windows2)
                                union = windows1.union(windows2)
                                
                                window_overlap[f"{algo1}_{algo2}_common_windows"] = len(intersection)
                                window_overlap[f"{algo1}_{algo2}_total_windows"] = len(union)
                                window_overlap[f"{algo1}_{algo2}_jaccard"] = len(intersection) / len(union) if len(union) > 0 else 0
                
                if window_overlap:
                    overlap_df = pd.DataFrame([window_overlap])
                    overlap_file = output_dir / "temporal_window_overlap.csv"
                    overlap_df.to_csv(overlap_file, index=False)
                    logger.info(f"Temporal window overlap analysis saved to {overlap_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error analyzing temporal results: {e}")
        return False


def main():
    """
    Main function to process and visualize algorithm results.
    """
    parser = argparse.ArgumentParser(description="Process and visualize anomaly detection results")
    parser.add_argument(
        "--results", 
        type=str, 
        default=str(config.DATA_DIR / "algorithm_results"),
        help="Directory containing algorithm results"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=str(config.DATA_DIR / "analysis_results"),
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default=str(config.PROCESSED_DATA_DIR / "index_GSPC_processed.csv"),
        help="Path to the original processed data file (for time index)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Window size used in temporal analysis"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Step size used in temporal analysis"
    )
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    results_dir = Path(args.results)
    output_dir = Path(args.output)
    data_file = Path(args.data)
    
    # Ensure output directory exists
    ensure_directory_exists(output_dir)
    
    # Load original data for time index
    data_df = None
    if data_file.exists():
        try:
            data_df = load_ticker_data(data_file)
            logger.info(f"Loaded original data from {data_file} with shape {data_df.shape}")
        except Exception as e:
            logger.error(f"Error loading original data: {e}")
    
    # Load algorithm scores
    score_series = {}
    for algo in ["aida", "iforest", "lof"]:
        scores = load_algorithm_scores(results_dir, algo)
        if scores is not None:
            score_series[algo] = scores
    
    if score_series:
        logger.info(f"Loaded scores for {', '.join(score_series.keys())}")
        
        # Create score comparisons
        comparison_dir = output_dir / "score_comparison"
        comparison_results = create_score_comparison(score_series, comparison_dir)
        
        if comparison_results:
            logger.info(f"Score comparison results saved to {comparison_dir}")
    else:
        logger.warning("No algorithm scores found")
    
    # Load algorithm anomalies
    anomalies_dict = {}
    for algo in ["aida", "iforest", "lof"]:
        anomalies = load_algorithm_anomalies(results_dir, algo)
        if anomalies is not None:
            anomalies_dict[algo] = anomalies
    
    if anomalies_dict:
        logger.info(f"Loaded anomalies for {', '.join(anomalies_dict.keys())}")
        
        # Analyze anomaly overlap
        overlap_dir = output_dir / "anomaly_overlap"
        overlap_results = analyze_anomaly_overlap(anomalies_dict, data_df.index if data_df is not None else None, overlap_dir)
        
        if overlap_results:
            logger.info(f"Anomaly overlap analysis saved to {overlap_dir}")
    else:
        logger.warning("No algorithm anomalies found")
    
    # Load and analyze temporal results
    temporal_results = load_temporal_results(results_dir, args.window_size, args.step)
    
    if temporal_results:
        logger.info(f"Loaded temporal results for {', '.join(temporal_results.keys())}")
        
        # Analyze temporal results
        temporal_dir = output_dir / f"temporal_w{args.window_size}_s{args.step}"
        temporal_success = analyze_temporal_results(temporal_results, temporal_dir)
        
        if temporal_success:
            logger.info(f"Temporal analysis results saved to {temporal_dir}")
    else:
        logger.warning("No temporal results found")
    
    logger.info(f"All analysis results saved to {output_dir}")


if __name__ == "__main__":
    main()
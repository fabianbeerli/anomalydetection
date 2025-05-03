#!/usr/bin/env python
"""
Script to run feature importance analysis for Isolation Forest and LOF algorithms.
"""
import os
import sys
import logging
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.models.isolation_forest import IForest
from src.models.lof import LOF
from src.utils.helpers import ensure_directory_exists, load_subsequence

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """
    Class to analyze feature importance for Isolation Forest and LOF algorithms.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the analyzer.
        
        Args:
            output_dir (Path or str, optional): Directory to save results
        """
        self.output_dir = Path(output_dir) if output_dir else Path(config.DATA_DIR) / "feature_importance_results"
        ensure_directory_exists(self.output_dir)
    
    def analyze_iforest_feature_importance(self, feature_data, anomaly_indices, algorithm_params=None):
        """
        Analyze feature importance for Isolation Forest.
        
        Args:
            feature_data (pandas.DataFrame): Feature-engineered data
            anomaly_indices (list): Indices of anomalies to explain
            algorithm_params (dict): Optional parameters for the algorithm
            
        Returns:
            dict: Feature importance results for each anomaly
        """
        if algorithm_params is None:
            algorithm_params = {}
        
        # Create Isolation Forest model
        model = IForest(
            n_estimators=algorithm_params.get('n_estimators', 100),
            max_samples=algorithm_params.get('max_samples', 256),
            contamination=algorithm_params.get('contamination', 0.05)
        )
        
        # Fit the model
        model.model.fit(feature_data)
        
        # Get feature importance for each anomaly
        feature_importance_results = {}
        
        for idx in anomaly_indices:
            if idx >= len(feature_data):
                logger.warning(f"Anomaly index {idx} out of range, skipping")
                continue
            
            try:
                # Get the anomaly data point
                anomaly_data = feature_data.iloc[[idx]]
                
                # Calculate feature importance for this anomaly
                feature_importance = self._get_iforest_feature_importance(model, anomaly_data)
                
                if feature_importance is not None:
                    # Store results
                    feature_importance_results[idx] = {
                        'feature_importance': feature_importance.to_dict(),
                        'summary': {
                            'total_features': len(feature_importance),
                            'top_features': feature_importance.nlargest(5).index.tolist(),
                            'top_scores': feature_importance.nlargest(5).tolist()
                        }
                    }
            except Exception as e:
                logger.error(f"Error calculating feature importance for anomaly {idx}: {e}")
        
        return feature_importance_results
    
    def _get_iforest_feature_importance(self, model, anomaly_data):
        """
        Calculate feature importance for Isolation Forest.
        
        This method uses path depth analysis to determine feature importance.
        Features that appear earlier (closer to the root) in the trees
        are more important for isolating the anomaly.
        
        Args:
            model (IForest): Isolation Forest model
            anomaly_data (pandas.DataFrame): Data point to explain
            
        Returns:
            pandas.Series: Feature importance scores
        """
        try:
            if not isinstance(anomaly_data, pd.DataFrame):
                logger.error("Feature importance calculation requires DataFrame with column names")
                return None
                
            # Initialize feature importance values
            n_features = anomaly_data.shape[1]
            feature_names = anomaly_data.columns
            feature_importance = np.zeros(n_features)
            
            # For each tree in the forest
            for tree in model.model.estimators_:
                # Find the path that the anomaly takes through the tree
                node_indicator = tree.decision_path(anomaly_data)
                leaf_id = tree.apply(anomaly_data)
                
                # Get the path (list of node indices)
                node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
                
                # For each node in the path, get the feature used for splitting
                for depth, node_id in enumerate(node_index):
                    # Skip leaf nodes
                    if node_id != leaf_id[0]:
                        # Get the feature used at this node
                        feature = tree.tree_.feature[node_id]
                        
                        # Skip leaf nodes (marked with -1 or -2)
                        if feature >= 0:
                            # Features used closer to the root have higher importance
                            # Use inverse of depth as importance weight
                            importance_weight = 1.0 / (depth + 1.0)
                            feature_importance[feature] += importance_weight
            
            # Normalize feature importance
            if np.sum(feature_importance) > 0:
                feature_importance = feature_importance / np.sum(feature_importance)
                
            # Create Series with feature names
            return pd.Series(feature_importance, index=feature_names)
            
        except Exception as e:
            logger.error(f"Error in feature importance calculation: {e}")
            return None
    
    def analyze_lof_feature_importance(self, feature_data, anomaly_indices, algorithm_params=None):
        """
        Analyze feature importance for Local Outlier Factor.
        
        Args:
            feature_data (pandas.DataFrame): Feature-engineered data
            anomaly_indices (list): Indices of anomalies to explain
            algorithm_params (dict): Optional parameters for the algorithm
            
        Returns:
            dict: Feature importance results for each anomaly
        """
        if algorithm_params is None:
            algorithm_params = {}
        
        # Create LOF model
        model = LOF(
            n_neighbors=algorithm_params.get('n_neighbors', 20),
            p=algorithm_params.get('p', 2),
            contamination=algorithm_params.get('contamination', 0.05)
        )
        
        # Fit the model
        model.model.fit(feature_data)
        
        # Get feature importance for each anomaly
        feature_importance_results = {}
        
        for idx in anomaly_indices:
            if idx >= len(feature_data):
                logger.warning(f"Anomaly index {idx} out of range, skipping")
                continue
            
            try:
                # Get the anomaly data point
                anomaly_data = feature_data.iloc[[idx]]
                
                # Calculate feature importance for this anomaly
                feature_importance = self._get_lof_feature_importance(model, feature_data, anomaly_data)
                
                if feature_importance is not None:
                    # Store results
                    feature_importance_results[idx] = {
                        'feature_importance': feature_importance.to_dict(),
                        'summary': {
                            'total_features': len(feature_importance),
                            'top_features': feature_importance.nlargest(5).index.tolist(),
                            'top_scores': feature_importance.nlargest(5).tolist()
                        }
                    }
            except Exception as e:
                logger.error(f"Error calculating feature importance for anomaly {idx}: {e}")
        
        return feature_importance_results
    
    def _get_lof_feature_importance(self, model, full_data, anomaly_data):
        """
        Calculate feature importance for Local Outlier Factor.
        
        This method uses a leave-one-out approach to determine feature importance.
        For each feature, it calculates how much the LOF score changes when the feature
        is removed from the calculation.
        
        Args:
            model (LOF): LOF model
            full_data (pandas.DataFrame): Full dataset
            anomaly_data (pandas.DataFrame): Data point to explain
            
        Returns:
            pandas.Series: Feature importance scores
        """
        try:
            if not isinstance(anomaly_data, pd.DataFrame):
                logger.error("Feature importance calculation requires DataFrame with column names")
                return None
                
            # Calculate original LOF score
            original_score = -model.model._decision_function(anomaly_data)[0]
            
            # Initialize feature importance values
            feature_names = anomaly_data.columns
            feature_importance = {}
            
            # For each feature, calculate the change in LOF score when the feature is removed
            for feature in feature_names:
                # Create a copy of the data with the feature removed
                reduced_data = full_data.drop(columns=[feature])
                reduced_anomaly = anomaly_data.drop(columns=[feature])
                
                # Create a new LOF model with the same parameters
                reduced_model = LOF(
                    n_neighbors=model.n_neighbors,
                    p=model.p,
                    contamination=model.contamination
                )
                
                # Fit the model
                reduced_model.model.fit(reduced_data)
                
                # Calculate the LOF score without this feature
                reduced_score = -reduced_model.model._decision_function(reduced_anomaly)[0]
                
                # Calculate the change in score
                score_change = abs(original_score - reduced_score)
                
                # Store the feature importance
                feature_importance[feature] = score_change
            
            # Convert to Series
            importance_series = pd.Series(feature_importance)
            
            # Normalize
            if importance_series.sum() > 0:
                importance_series = importance_series / importance_series.sum()
            
            return importance_series
            
        except Exception as e:
            logger.error(f"Error in LOF feature importance calculation: {e}")
            return None
    
    def visualize_feature_importance(self, feature_importance, output_file, title="Feature Importance"):
        """
        Visualize feature importance for an anomaly.
        
        Args:
            feature_importance (dict): Dictionary mapping features to importance scores
            output_file (Path): Path to save the visualization
            title (str): Title for the visualization
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert to Series if dict
            if isinstance(feature_importance, dict):
                feature_importance = pd.Series(feature_importance)
            
            # Sort features by importance
            sorted_features = feature_importance.sort_values(ascending=False)
            
            # Create visualization
            plt.figure(figsize=(10, 6))
            plt.barh(sorted_features.index, sorted_features.values)
            plt.xlabel('Importance Score')
            plt.ylabel('Feature')
            plt.title(title)
            plt.tight_layout()
            
            # Save figure
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature importance visualization saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error visualizing feature importance: {e}")
            return False
    
    def analyze_subsequence_anomalies(self, ticker, algorithm, window_size, overlap_type):
        """
        Analyze feature importance for subsequence anomalies.
        
        Args:
            ticker (str): Ticker symbol
            algorithm (str): Algorithm name ('iforest' or 'lof')
            window_size (int): Window size
            overlap_type (str): Overlap type ('overlap' or 'nonoverlap')
            
        Returns:
            dict: Feature importance results
        """
        # Base directory for subsequence results
        subsequence_results_dir = Path(config.DATA_DIR) / "analysis_results" / "subsequence_results"
        
        # Path to anomalies file
        anomalies_file = subsequence_results_dir / ticker / algorithm / f"w{window_size}_{overlap_type}" / f"{algorithm}_anomalies.csv"
        
        if not anomalies_file.exists():
            logger.warning(f"No anomalies file found at {anomalies_file}")
            return {}
        
        # Path to feature file
        if algorithm == 'iforest':
            feature_file = subsequence_results_dir / ticker / algorithm / f"w{window_size}_{overlap_type}" / "iforest_features.csv"
        elif algorithm == 'lof':
            feature_file = subsequence_results_dir / ticker / algorithm / f"w{window_size}_{overlap_type}" / "lof_feature.csv"

        if not feature_file.exists():
            logger.warning(f"No feature file found at {feature_file}")
            return {}
        
        # Load anomalies and features
        try:
            anomalies_df = pd.read_csv(anomalies_file)
            feature_df = pd.read_csv(feature_file)
            
            if anomalies_df.empty:
                logger.warning(f"No anomalies found for {ticker} with {algorithm}")
                return {}
            
            # Get anomaly indices
            if 'subsequence_idx' in anomalies_df.columns:
                anomaly_indices = anomalies_df['subsequence_idx'].tolist()
            elif 'index' in anomalies_df.columns:
                anomaly_indices = anomalies_df['index'].tolist()
            else:
                logger.warning("No valid index column found in anomalies file")
                return {}
            
            # Analyze feature importance
            output_dir = self.output_dir / "subsequence" / ticker / algorithm / f"w{window_size}_{overlap_type}"
            ensure_directory_exists(output_dir)
            
            if algorithm == 'iforest':
                feature_importance_results = self.analyze_iforest_feature_importance(feature_df, anomaly_indices)
            elif algorithm == 'lof':
                feature_importance_results = self.analyze_lof_feature_importance(feature_df, anomaly_indices)
            else:
                logger.warning(f"Unsupported algorithm: {algorithm}")
                return {}
            
            # Visualize feature importance for each anomaly
            for idx, result in feature_importance_results.items():
                if 'feature_importance' in result:
                    anomaly_output_dir = output_dir / f"{algorithm}_anomaly_{idx}"
                    ensure_directory_exists(anomaly_output_dir)
                    
                    # Create visualization
                    self.visualize_feature_importance(
                        result['feature_importance'],
                        anomaly_output_dir / f"feature_importance_{idx}.png",
                        title=f"{ticker} {algorithm.upper()} Anomaly {idx} - Feature Importance"
                    )
                    
                    # Save metadata
                    anomaly_row = anomalies_df[anomalies_df['subsequence_idx'] == idx].iloc[0] if 'subsequence_idx' in anomalies_df.columns else \
                                 anomalies_df[anomalies_df['index'] == idx].iloc[0]
                    
                    metadata = {
                        'ticker': ticker,
                        'algorithm': algorithm,
                        'subsequence_idx': int(idx),
                        'anomaly_score': float(anomaly_row.get('score', float('nan'))),
                        'start_date': str(anomaly_row.get('start_date', '')),
                        'end_date': str(anomaly_row.get('end_date', ''))
                    }
                    
                    with open(anomaly_output_dir / "anomaly_metadata.json", 'w') as f:
                        json.dump(metadata, f, indent=2)
            
            # Save summary
            if feature_importance_results:
                summary = {
                    'algorithm': algorithm,
                    'total_anomalies_analyzed': len(feature_importance_results),
                    'anomalies': list(feature_importance_results.keys())
                }
                
                with open(output_dir / f"{algorithm}_feature_importance_summary.json", 'w') as f:
                    json.dump(summary, f, indent=2)
            
            logger.info(f"Feature importance analysis completed for {ticker} with {algorithm}")
            return feature_importance_results
            
        except Exception as e:
            logger.error(f"Error analyzing subsequence anomalies for {ticker} with {algorithm}: {e}")
            return {}


def main():
    """
    Main function to run feature importance analysis.
    """
    parser = argparse.ArgumentParser(description="Run feature importance analysis for LOF and Isolation Forest")
    
    # General arguments
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=str(config.DATA_DIR / "analysis_results" / "feature_importance_results"),
        help="Directory for feature importance results"
    )
    
    # Sequence analysis arguments
    parser.add_argument(
        "--ticker", 
        type=str, 
        default="sp500",
        help="Ticker to analyze"
    )
    parser.add_argument(
        "--algorithms", 
        type=str, 
        default="iforest,lof",
        help="Comma-separated list of algorithms to analyze"
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
        help="Only analyze overlapping subsequences"
    )
    parser.add_argument(
        "--only-nonoverlap", 
        action="store_true",
        help="Only analyze non-overlapping subsequences"
    )
    
    # Debug option
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    # Create analyzer
    analyzer = FeatureImportanceAnalyzer(output_dir=args.output_dir)
    
    # Parse arguments
    algorithms = args.algorithms.split(',')
    window_sizes = [int(size) for size in args.window_sizes.split(',')]
    
    overlap_types = []
    if args.only_overlap:
        overlap_types = ['overlap']
    elif args.only_nonoverlap:
        overlap_types = ['nonoverlap']
    else:
        overlap_types = ['overlap', 'nonoverlap']
    
    # Run analysis for each configuration
    results = {}
    
    for algorithm in algorithms:
        for window_size in window_sizes:
            for overlap_type in overlap_types:
                logger.info(f"Analyzing {args.ticker} with {algorithm} (w{window_size}, {overlap_type})")
                
                result = analyzer.analyze_subsequence_anomalies(
                    ticker=args.ticker,
                    algorithm=algorithm,
                    window_size=window_size,
                    overlap_type=overlap_type
                )
                
                results[f"{args.ticker}_{algorithm}_w{window_size}_{overlap_type}"] = result
    
    # Save all results
    all_results_file = Path(args.output_dir) / "subsequence_feature_importance_summary.json"
    
    # Make results JSON serializable
    json_results = {}
    for key, value in results.items():
        json_results[key] = {}
        for idx, result in value.items():
            json_results[key][idx] = {
                'summary': result.get('summary', {}),
                'feature_importance': {k: float(v) for k, v in result.get('feature_importance', {}).items()}
            }
    
    with open(all_results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"All results saved to {all_results_file}")


if __name__ == "__main__":
    main()
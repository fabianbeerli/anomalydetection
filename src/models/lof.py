"""
Local Outlier Factor (LOF) implementation for anomaly detection.
"""
import os
import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from pathlib import Path

from src import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LOF:
    """
    Local Outlier Factor implementation for anomaly detection.
    """

    def __init__(self, n_neighbors=20, p=2, contamination=0.05):
        """
        Initialize the LOF model.
        
        Args:
            n_neighbors (int): Number of neighbors to use for k-neighbors queries.
            p (int): Parameter for Minkowski metric (1 for Manhattan, 2 for Euclidean).
            contamination (float): Expected proportion of anomalies in the dataset.
        """
        self.n_neighbors = n_neighbors
        self.p = p
        self.contamination = contamination
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            p=p,
            contamination=contamination,
            n_jobs=-1,  # Use all available processors
            novelty=False  # Set to False for unsupervised learning
        )
        
    def fit_predict(self, data):
        """
        Fit the LOF model and predict anomalies.
        
        Args:
            data (pandas.DataFrame or numpy.ndarray): Input data.
            
        Returns:
            tuple: (anomaly_scores, anomaly_labels)
                anomaly_scores: Higher values indicate more anomalous points
                anomaly_labels: -1 for anomalies, 1 for normal points
        """
        # Ensure input is a numpy array
        if isinstance(data, pd.DataFrame):
            X = data.values
        else:
            X = data
            
        # Fit and predict
        try:
            logger.info(f"Running LOF with n_neighbors={self.n_neighbors}, p={self.p}")
            labels = self.model.fit_predict(X)
            
            # Get anomaly scores (-1 * negative_outlier_factor_)
            # Convert to positive values where higher = more anomalous
            scores = -self.model.negative_outlier_factor_
            
            # Normalize scores similar to AIDA
            scores = (scores - np.mean(scores)) / np.std(scores)
            
            # Use mean + 2*std threshold instead of contamination
            threshold = np.mean(scores) + 2 * np.std(scores)
            labels = np.ones(len(scores))
            labels[scores > threshold] = -1  # -1 for anomalies, 1 for normal points
            
            logger.info(f"LOF completed. Found {np.sum(labels == -1)} anomalies.")
            
            return scores, labels
        except Exception as e:
            logger.error(f"Error during LOF execution: {e}")
            return None, None
        
    def save_results(self, scores, labels, data_index, output_dir, prefix="lof"):
        """
        Save the LOF results to files.
        
        Args:
            scores (numpy.ndarray): Anomaly scores.
            labels (numpy.ndarray): Anomaly labels (-1 for anomalies, 1 for normal).
            data_index (pandas.Index): Index of the original data.
            output_dir (str or Path): Directory to save results.
            prefix (str): Prefix for output files.
            
        Returns:
            tuple: (scores_file, anomalies_file) Paths to saved files
        """
        if scores is None or labels is None:
            logger.error("No results to save.")
            return None, None
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scores
        scores_file = output_dir / f"{prefix}_scores.dat"
        with open(scores_file, 'w') as f:
            f.write(f"{len(scores)}\n")
            for score in scores:
                f.write(f"{score}\n")
                
        # Save anomalies
        anomalies_file = output_dir / f"{prefix}_anomalies.csv"
        anomaly_indices = np.where(labels == -1)[0]
        
        # Create DataFrame with anomalies
        if hasattr(data_index, 'values'):
            # For time index, use the actual timestamp
            anomaly_df = pd.DataFrame({
                'index': anomaly_indices,
                'date': [data_index.values[i] for i in anomaly_indices],
                'score': scores[anomaly_indices]
            })
        else:
            anomaly_df = pd.DataFrame({
                'index': anomaly_indices,
                'score': scores[anomaly_indices]
            })
            
        anomaly_df.to_csv(anomalies_file, index=False)
        
        logger.info(f"Saved LOF scores to {scores_file}")
        logger.info(f"Saved LOF anomalies to {anomalies_file}")
        
        return scores_file, anomalies_file


class TemporalLOF(LOF):
    """
    Extension of LOF for temporal data.
    Applies LOF to sliding windows of time series data.
    """
    
    def __init__(self, n_neighbors=20, p=2, contamination=0.05, window_size=5, step=1):
        """
        Initialize the Temporal LOF model.
        
        Args:
            n_neighbors (int): Number of neighbors to use.
            p (int): Parameter for Minkowski metric.
            contamination (float): Expected proportion of anomalies.
            window_size (int): Size of each sliding window.
            step (int): Step size between consecutive windows.
        """
        super().__init__(n_neighbors, p, contamination)
        self.window_size = window_size
        self.step = step
        
    def fit_predict_temporal(self, time_series):
        """
        Apply LOF to sliding windows of time series data.
        
        Args:
            time_series (pandas.DataFrame): Time series data.
            
        Returns:
            tuple: (window_scores, anomaly_windows)
                window_scores: Dictionary mapping window indices to anomaly scores
                anomaly_windows: Dictionary mapping window indices to anomaly status
        """
        n_samples = len(time_series)
        window_scores = {}
        anomaly_windows = {}
        
        # Apply LOF to each window
        for i in range(0, n_samples - self.window_size + 1, self.step):
            window = time_series.iloc[i:i+self.window_size]
            window_data = window.values.reshape(1, -1)  # Reshape to 1 x (window_size * features)
            
            try:
                # Create a new model for each window (or reuse with partial_fit if large dataset)
                window_model = LocalOutlierFactor(
                    n_neighbors=min(self.n_neighbors, len(window_data) - 1),
                    p=self.p,
                    contamination=self.contamination,
                    n_jobs=-1
                )
                
                # Compute scores
                labels = window_model.fit_predict(window_data)
                scores = -window_model.negative_outlier_factor_
                
                window_scores[i] = scores[0]
                anomaly_windows[i] = (labels[0] == -1)
                
            except Exception as e:
                logger.error(f"Error processing window {i}: {e}")
                window_scores[i] = 0
                anomaly_windows[i] = False
                
        return window_scores, anomaly_windows
    
    def get_lof_feature_importance(model, X, point_idx=None):
        n_features = X.shape[1]
        feature_importances = np.zeros(n_features)
        
        # Get original LOF scores
        original_scores = -model.negative_outlier_factor_
        
        # If specific point is provided, focus on that point
        if point_idx is not None:
            points_to_analyze = [point_idx]
        else:
            # Focus on anomalies
            points_to_analyze = np.where(original_scores > np.percentile(original_scores, 95))[0]
        
        # For each feature, calculate how score changes when feature is removed
        for feature_idx in range(n_features):
            # Create copy of data without this feature
            X_modified = np.delete(X, feature_idx, axis=1)
            
            # Create and fit a new LOF model
            new_model = LocalOutlierFactor(n_neighbors=model.n_neighbors)
            new_model.fit(X_modified)
            
            # Get new scores
            new_scores = -new_model.negative_outlier_factor_
            
            # Calculate importance based on score changes for points of interest
            for idx in points_to_analyze:
                feature_importances[feature_idx] += abs(original_scores[idx] - new_scores[idx])
        
        # Normalize
        if feature_importances.sum() > 0:
            feature_importances = feature_importances / feature_importances.sum()
        
        return feature_importances

    def save_temporal_results(self, window_scores, anomaly_windows, time_series, output_dir, prefix="temporal_lof"):
        """
        Save the Temporal LOF results to files.
        
        Args:
            window_scores (dict): Anomaly scores for each window.
            anomaly_windows (dict): Anomaly status for each window.
            time_series (pandas.DataFrame): Original time series data.
            output_dir (str or Path): Directory to save results.
            prefix (str): Prefix for output files.
            
        Returns:
            tuple: (scores_file, anomalies_file) Paths to saved files
        """
        if not window_scores or not anomaly_windows:
            logger.error("No temporal results to save.")
            return None, None
            
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert dictionaries to DataFrames
        windows_df = pd.DataFrame({
            'window_start': window_scores.keys(),
            'score': window_scores.values(),
            'is_anomaly': [anomaly_windows[k] for k in window_scores.keys()]
        })
        
        # Add start and end dates for each window
        windows_df['start_date'] = windows_df['window_start'].apply(
            lambda x: time_series.index[x] if x < len(time_series.index) else None
        )
        windows_df['end_date'] = windows_df['window_start'].apply(
            lambda x: time_series.index[min(x + self.window_size - 1, len(time_series.index) - 1)] 
            if x < len(time_series.index) else None
        )
        
        # Save all window scores
        scores_file = output_dir / f"{prefix}_window_scores.csv"
        windows_df.to_csv(scores_file, index=False)
        
        # Save anomaly windows
        anomalies_file = output_dir / f"{prefix}_window_anomalies.csv"
        anomaly_df = windows_df[windows_df['is_anomaly']]
        anomaly_df.to_csv(anomalies_file, index=False)
        
        logger.info(f"Saved Temporal LOF window scores to {scores_file}")
        logger.info(f"Saved Temporal LOF window anomalies to {anomalies_file}")
        
        return scores_file, anomalies_file
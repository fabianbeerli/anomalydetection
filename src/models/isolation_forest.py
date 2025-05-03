"""
Isolation Forest implementation for anomaly detection.
"""
import os
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from pathlib import Path

from src import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IForest:
    """
    Isolation Forest implementation for anomaly detection.
    """

    def __init__(self, n_estimators=100, max_samples=256, contamination=0.05, random_state=42):
        """
        Initialize the Isolation Forest model.
        
        Args:
            n_estimators (int): Number of trees in the forest.
            max_samples (int or float): Number of samples to draw to train each tree.
            contamination (float): Expected proportion of anomalies in the dataset.
            random_state (int): Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1  # Use all available processors
        )
        
    # For IForest in isolation_forest.py
    def fit_predict(self, data):
        # Ensure input is a numpy array
        if isinstance(data, pd.DataFrame):
            X = data.values
        else:
            X = data
            
        # Fit the model
        try:
            logger.info(f"Running Isolation Forest with n_estimators={self.n_estimators}, max_samples={self.max_samples}")
            self.model.fit(X)
            
            # Get anomaly scores (negative decision function values)
            # Convert to positive values where higher = more anomalous
            scores = -self.model.decision_function(X)
            
            # Normalize scores similar to AIDA
            scores = (scores - np.mean(scores)) / np.std(scores)
            
            # Use mean + 2*std threshold instead of contamination
            threshold = np.mean(scores) + 2 * np.std(scores)
            labels = np.ones(len(scores))
            labels[scores > threshold] = -1  # -1 for anomalies, 1 for normal points
            
            logger.info(f"Isolation Forest completed. Found {np.sum(labels == -1)} anomalies.")
            
            return scores, labels
        except Exception as e:
            logger.error(f"Error during Isolation Forest execution: {e}")
            return None, None
        
    def save_results(self, scores, labels, data_index, output_dir, prefix="iforest"):
        """
        Save the Isolation Forest results to files.
        
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
        
        logger.info(f"Saved Isolation Forest scores to {scores_file}")
        logger.info(f"Saved Isolation Forest anomalies to {anomalies_file}")
        
        return scores_file, anomalies_file
    
    def get_feature_importance(self, data):
        """
        Calculate feature importance scores for the Isolation Forest model.
        
        Higher scores indicate more important features for anomaly detection.
        
        Args:
            data (pandas.DataFrame): Input data with column names.
            
        Returns:
            pandas.Series: Feature importance scores, indexed by feature names.
        """
        if not isinstance(data, pd.DataFrame):
            logger.error("Feature importance calculation requires DataFrame with column names")
            return None
            
        if not hasattr(self.model, 'estimators_'):
            logger.error("Model must be fit before calculating feature importance")
            return None
            
        # Initialize feature importance values
        n_features = data.shape[1]
        feature_names = data.columns
        feature_importance = np.zeros(n_features)
        
        # Calculate average path length for each feature across all trees
        try:
            # For each tree in the forest
            for tree in self.model.estimators_:
                # Get the feature used at each node
                node_feature = tree.tree_.feature
                
                # Count feature usage
                for feature in node_feature:
                    # Skip leaf nodes (marked with -1 or -2)
                    if feature >= 0:
                        feature_importance[feature] += 1
                        
            # Normalize feature importance
            if np.sum(feature_importance) > 0:
                feature_importance = feature_importance / np.sum(feature_importance)
                
            # Create Series with feature names
            return pd.Series(feature_importance, index=feature_names)
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return None


class TemporalIsolationForest(IForest):
    """
    Extension of Isolation Forest for temporal data.
    Applies Isolation Forest to sliding windows of time series data.
    """
    
    def __init__(self, n_estimators=100, max_samples=256, contamination=0.05, 
                 random_state=42, window_size=5, step=1):
        """
        Initialize the Temporal Isolation Forest model.
        
        Args:
            n_estimators (int): Number of trees in the forest.
            max_samples (int or float): Number of samples to draw to train each tree.
            contamination (float): Expected proportion of anomalies in the dataset.
            random_state (int): Random seed for reproducibility.
            window_size (int): Size of each sliding window.
            step (int): Step size between consecutive windows.
        """
        super().__init__(n_estimators, max_samples, contamination, random_state)
        self.window_size = window_size
        self.step = step
        
    def fit_predict_temporal(self, time_series):
        """
        Apply Isolation Forest to sliding windows of time series data.
        
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
        
        # Apply Isolation Forest to each window
        for i in range(0, n_samples - self.window_size + 1, self.step):
            window = time_series.iloc[i:i+self.window_size]
            
            # Ensure we have enough data for meaningful analysis
            if len(window) < 3:
                logger.warning(f"Window {i} has too few samples ({len(window)}), skipping")
                continue
                
            window_data = window.values.reshape(1, -1)  # Reshape to 1 x (window_size * features)
            
            try:
                # For each window, use a separate model to avoid memory issues
                # Or use a single model with partial_fit if large dataset
                window_model = IsolationForest(
                    n_estimators=self.n_estimators,
                    max_samples=min(self.max_samples, len(window_data)),
                    contamination=self.contamination,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                
                # Compute scores
                # For single windows, we need a collection of windows for comparison
                # Create a surrounding context of windows
                context_start = max(0, i - 5 * self.step)
                context_end = min(n_samples, i + 5 * self.step + self.window_size)
                context_windows = []
                
                for j in range(context_start, context_end - self.window_size + 1, self.step):
                    context_windows.append(time_series.iloc[j:j+self.window_size].values.flatten())
                    
                context_data = np.array(context_windows)
                
                # Fit the model on context windows
                window_model.fit(context_data)
                
                # Get score for our target window
                score = -window_model.decision_function([window_data.flatten()])[0]
                
                # Set a threshold for detection
                threshold = np.percentile(
                    -window_model.decision_function(context_data), 
                    100 * (1 - self.contamination)
                )
                
                window_scores[i] = score
                anomaly_windows[i] = (score > threshold)
                
            except Exception as e:
                logger.error(f"Error processing window {i}: {e}")
                window_scores[i] = 0
                anomaly_windows[i] = False
                
        return window_scores, anomaly_windows
    
    def get_iforest_feature_importance(model, X):
        n_features = X.shape[1]
        feature_importances = np.zeros(n_features)
        
        # For each tree in the forest
        for tree in model.estimators_:
            # Get the feature used at each node
            node_feature = tree.tree_.feature
            
            # Count feature usage, weighing by depth
            for node_idx, feature_idx in enumerate(node_feature):
                # Skip leaf nodes (marked with -1 or -2)
                if feature_idx >= 0:
                    # Get depth of this node
                    depth = 0
                    parent = node_idx
                    while parent != 0:  # While not at root
                        parent = (parent - 1) // 2  # Find parent
                        depth += 1
                    
                    # More important features appear closer to root (lower depth)
                    feature_importances[feature_idx] += 1.0 / (depth + 1.0)
        
        # Normalize
        if feature_importances.sum() > 0:
            feature_importances = feature_importances / feature_importances.sum()
        
        return feature_importances

    def save_temporal_results(self, window_scores, anomaly_windows, time_series, output_dir, prefix="temporal_iforest"):
        """
        Save the Temporal Isolation Forest results to files.
        
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
        
        logger.info(f"Saved Temporal Isolation Forest window scores to {scores_file}")
        logger.info(f"Saved Temporal Isolation Forest window anomalies to {anomalies_file}")
        
        return scores_file, anomalies_file
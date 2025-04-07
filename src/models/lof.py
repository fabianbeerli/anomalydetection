"""
Local Outlier Factor (LOF) implementation for anomaly detection.

This module provides a wrapper around sklearn's LocalOutlierFactor
with additional functionality for time series and subsequence analysis.
"""
import numpy as np
import logging
from sklearn.neighbors import LocalOutlierFactor as SklearnLOF
from sklearn.utils.validation import check_array, check_is_fitted
from typing import List, Union, Optional, Tuple

logger = logging.getLogger(__name__)


class LOF:
    """
    Wrapper for sklearn's LocalOutlierFactor with additional functionality.
    
    Parameters
    ----------
    n_neighbors : int, default=20
        Number of neighbors to use by default for k-neighbors queries.
    
    metric : str, default='minkowski'
        Metric to use for distance computation.
    
    p : int, default=2
        Parameter for the Minkowski metric.
        When p = 1, this is equivalent to using manhattan_distance,
        and euclidean_distance for p = 2.
    
    contamination : float, default='auto'
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set.
    
    Attributes
    ----------
    model_ : LocalOutlierFactor
        The underlying sklearn LocalOutlierFactor model.
    
    scores_ : ndarray of shape (n_samples,)
        The anomaly scores for each sample. Higher values represent more
        abnormal samples.
    """
    
    def __init__(
        self,
        n_neighbors: int = 20,
        metric: str = 'minkowski',
        p: int = 2,
        contamination: Union[float, str] = 'auto'
    ):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p
        self.contamination = contamination
    
    def fit(self, X, y=None):
        """
        Fit the LOF model.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        
        y : Ignored
            Not used, present for API consistency by convention.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Initialize and fit the sklearn model
        self.model_ = SklearnLOF(
            n_neighbors=self.n_neighbors,
            algorithm='auto',
            metric=self.metric,
            p=self.p,
            contamination=self.contamination,
            novelty=False
        )
        
        self.model_.fit(X)
        
        # Store negative_outlier_factor_ as scores_
        # We negate it so higher values = more anomalous
        self.scores_ = -self.model_.negative_outlier_factor_
        
        # Compute feature importance via sensitivity analysis
        self._compute_feature_importances(X)
        
        return self
    
    def _compute_feature_importances(self, X):
        """
        Compute feature importances using sensitivity analysis.
        
        For LOF, we compute feature importance by measuring the change in
        anomaly scores when each feature is removed.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        """
        n_samples, n_features = X.shape
        
        # Initialize feature importances
        self.feature_importances_ = np.zeros(n_features)
        
        # If there's only one feature, its importance is 1
        if n_features == 1:
            self.feature_importances_[0] = 1.0
            return
        
        try:
            # Baseline scores
            baseline_scores = self.scores_
            
            # For each feature, measure the impact of removing it
            for i in range(n_features):
                # Create a copy of the data without this feature
                X_reduced = np.delete(X, i, axis=1)
                
                # Fit a new LOF model
                lof_reduced = SklearnLOF(
                    n_neighbors=self.n_neighbors,
                    algorithm='auto',
                    metric=self.metric,
                    p=self.p,
                    contamination=self.contamination,
                    novelty=False
                )
                lof_reduced.fit(X_reduced)
                
                # Compute new scores
                new_scores = -lof_reduced.negative_outlier_factor_
                
                # Compute the mean absolute difference in scores
                score_diff = np.mean(np.abs(baseline_scores - new_scores))
                
                # Store as feature importance
                self.feature_importances_[i] = score_diff
            
            # Normalize feature importances
            if np.sum(self.feature_importances_) > 0:
                self.feature_importances_ = self.feature_importances_ / np.sum(self.feature_importances_)
        except Exception as e:
            # In case of error, use uniform feature importances
            self.feature_importances_ = np.ones(n_features) / n_features
            print(f"Warning: Could not compute feature importances for LOF: {e}")
            print("Using uniform feature importances instead.")
    
    def decision_function(self, X):
        """
        Compute the anomaly scores for new samples.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            The higher, the more abnormal.
        """
        check_is_fitted(self, ["model_"])
        
        # Return negative of the outlier factors for consistency with other models
        # (higher values = more anomalous)
        if hasattr(self.model_, 'decision_function'):
            return -self.model_.decision_function(X)
        else:
            # LocalOutlierFactor with novelty=False does not have decision_function
            # Since we can't predict scores for new samples, return zeros as a placeholder
            return np.zeros(X.shape[0])
    
    def predict(self, X):
        """
        Predict if a particular sample is an outlier or not.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        
        Returns
        -------
        is_outlier : ndarray of shape (n_samples,)
            For each observation, tells whether or not it should be considered
            as an outlier according to the fitted model.
            -1 for outliers, 1 for inliers.
        """
        check_is_fitted(self, ["model_"])
        return self.model_.predict(X)
    

class TemporalLOF(LOF):
    """
    Temporal version of LOF for time series data.
    
    This extension is designed to work with time series subsequences.
    
    Parameters
    ----------
    subsequence_length : int, default=3
        Length of subsequences to analyze.
    
    step : int, default=1
        Step size between subsequences (1 for overlapping, subsequence_length for non-overlapping).
    
    n_neighbors : int, default=20
        Number of neighbors to use by default for k-neighbors queries.
    
    metric : str, default='minkowski'
        Metric to use for distance computation.
    
    p : int, default=2
        Parameter for the Minkowski metric.
    
    contamination : float, default='auto'
        The amount of contamination of the data set.
    """
    
    def __init__(
        self,
        subsequence_length: int = 3,
        step: int = 1,
        n_neighbors: int = 20,
        metric: str = 'minkowski',
        p: int = 2,
        contamination: Union[float, str] = 'auto'
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            metric=metric,
            p=p,
            contamination=contamination
        )
        self.subsequence_length = subsequence_length
        self.step = step
    
    def create_subsequences(self, X):
        """
        Create subsequences from time series data.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input time series data.
        
        Returns
        -------
        subsequences : ndarray of shape (n_subsequences, subsequence_length * n_features)
            The flattened subsequences.
        
        subsequence_indices : ndarray of shape (n_subsequences, 2)
            Start and end indices of each subsequence in the original data.
        """
        n_samples, n_features = X.shape
        n_subsequences = max(0, (n_samples - self.subsequence_length) // self.step + 1)
        
        subsequences = np.zeros((n_subsequences, self.subsequence_length * n_features))
        subsequence_indices = np.zeros((n_subsequences, 2), dtype=int)
        
        for i in range(n_subsequences):
            start_idx = i * self.step
            end_idx = start_idx + self.subsequence_length
            
            # Extract subsequence
            subseq = X[start_idx:end_idx]
            
            # Flatten the subsequence
            subsequences[i] = subseq.flatten()
            
            # Store indices
            subsequence_indices[i] = [start_idx, end_idx - 1]
        
        return subsequences, subsequence_indices
    
    def fit(self, X, y=None):
        """
        Fit the Temporal LOF model to subsequences of the input data.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input time series data.
        
        y : Ignored
            Not used, present for API consistency by convention.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Create subsequences
        subsequences, self.subsequence_indices_ = self.create_subsequences(X)
        
        # Apply LOF to the subsequences
        super().fit(subsequences)
        
        # Store original data shape
        self.original_shape_ = X.shape
        
        return self
    
    def get_anomaly_subsequences(self, threshold_percentile=95):
        """
        Get the anomalous subsequences based on a percentile threshold.
        
        Parameters
        ----------
        threshold_percentile : float, default=95
            Percentile threshold for anomaly detection.
        
        Returns
        -------
        anomaly_indices : ndarray
            Indices of anomalous subsequences.
        
        anomaly_scores : ndarray
            Anomaly scores of the anomalous subsequences.
        """
        check_is_fitted(self, ["scores_"])
        
        threshold = np.percentile(self.scores_, threshold_percentile)
        anomaly_indices = np.where(self.scores_ > threshold)[0]
        anomaly_scores = self.scores_[anomaly_indices]
        
        return anomaly_indices, anomaly_scores
    
    def get_anomaly_timestamps(self, threshold_percentile=95):
        """
        Get the timestamps (start and end indices) of anomalous subsequences.
        
        Parameters
        ----------
        threshold_percentile : float, default=95
            Percentile threshold for anomaly detection.
        
        Returns
        -------
        anomaly_timestamps : ndarray of shape (n_anomalies, 2)
            Start and end indices of anomalous subsequences in the original data.
        
        anomaly_scores : ndarray of shape (n_anomalies,)
            Anomaly scores of the anomalous subsequences.
        """
        check_is_fitted(self, ["subsequence_indices_"])
        
        anomaly_indices, anomaly_scores = self.get_anomaly_subsequences(threshold_percentile)
        anomaly_timestamps = self.subsequence_indices_[anomaly_indices]
        
        return anomaly_timestamps, anomaly_scores
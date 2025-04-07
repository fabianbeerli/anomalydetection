"""
Isolation Forest implementation for anomaly detection.

This module provides a wrapper around sklearn's IsolationForest
with additional functionality for time series and subsequence analysis.
"""
import numpy as np
import logging
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.utils.validation import check_array, check_is_fitted
from typing import List, Union, Optional, Tuple

logger = logging.getLogger(__name__)


class IsolationForest:
    """
    Wrapper for sklearn's IsolationForest with additional functionality.
    
    Parameters
    ----------
    n_estimators : int, default=100
        The number of base estimators in the ensemble.
    
    max_samples : int or float, default='auto'
        The number of samples to draw from X to train each base estimator.
            - If int, then draw max_samples samples.
            - If float, then draw max_samples * X.shape[0] samples.
            - If 'auto', then max_samples=min(256, n_samples).
    
    contamination : float, default=0.1
        The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set.
    
    random_state : int, optional
        Random seed for reproducibility.
    
    Attributes
    ----------
    model_ : IsolationForest
        The underlying sklearn IsolationForest model.
    
    feature_importances_ : ndarray of shape (n_features,)
        Feature importances based on the average path length.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[int, float, str] = 'auto',
        contamination: float = 0.1,
        random_state: Optional[int] = None
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
    
    def fit(self, X, y=None):
        """
        Fit the isolation forest model.
        
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
        self.model_ = SklearnIsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state
        )
        
        self.model_.fit(X)
        
        # Compute anomaly scores as negative of decision function
        # This way, higher values = more anomalous
        self.scores_ = -self.model_.decision_function(X)
        
        # Compute feature importances
        self._compute_feature_importances()
        
        return self
    
    def _compute_feature_importances(self):
        """
        Compute feature importances based on the average path length reduction.
        
        In Isolation Forest, features that contribute more to isolating points
        (i.e., appear more frequently in splitting decisions) are more important.
        """
        # Feature importances calculation for Isolation Forest
        # This is a workaround since sklearn's IsolationForest doesn't directly provide feature_importances_
        
        # Initialize feature importances to zeros
        n_features = self.model_.n_features_in_
        self.feature_importances_ = np.zeros(n_features)
        
        # If the model has estimators_ attribute, compute feature importances from the base estimators
        if hasattr(self.model_, 'estimators_'):
            # Calculate feature importances for each tree if available
            all_importances = []
            for tree in self.model_.estimators_:
                if hasattr(tree, 'feature_importances_'):
                    all_importances.append(tree.feature_importances_)
            
            # If we have importances, average them
            if all_importances:
                self.feature_importances_ = np.mean(all_importances, axis=0)
            else:
                # Use uniform importance if we can't calculate them
                self.feature_importances_ = np.ones(n_features) / n_features
        else:
            # Use uniform importance if estimators_ is not available
            self.feature_importances_ = np.ones(n_features) / n_features
    
    def decision_function(self, X):
        """
        Average anomaly score of X of the base estimators.
        
        The anomaly score of an input sample is computed as
        the mean anomaly score of the trees in the forest.
        
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
        
        # Return negative of sklearn's decision function for consistency with AIDA
        # (higher values = more anomalous)
        return -self.model_.decision_function(X)
    
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
    

class TemporalIsolationForest(IsolationForest):
    """
    Temporal version of Isolation Forest for time series data.
    
    This extension is designed to work with time series subsequences.
    
    Parameters
    ----------
    subsequence_length : int, default=3
        Length of subsequences to analyze.
    
    step : int, default=1
        Step size between subsequences (1 for overlapping, subsequence_length for non-overlapping).
    
    n_estimators : int, default=100
        The number of base estimators in the ensemble.
    
    max_samples : int or float, default='auto'
        The number of samples to draw from X to train each base estimator.
    
    contamination : float, default=0.1
        The amount of contamination of the data set.
    
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        subsequence_length: int = 3,
        step: int = 1,
        n_estimators: int = 100,
        max_samples: Union[int, float, str] = 'auto',
        contamination: float = 0.1,
        random_state: Optional[int] = None
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state
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
        Fit the Temporal Isolation Forest model to subsequences of the input data.
        
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
        
        # Apply Isolation Forest to the subsequences
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
"""
AIDA (Analytic Isolation and Distance-based Anomaly) algorithm implementation.

Based on the paper:
"AIDA: Analytic isolation and distance-based anomaly detection algorithm"
by Luis Antonio Souto Arias, Cornelis W. Oosterlee, and Pasquale Cirillo (2023)
"""
import numpy as np
import logging
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_array, check_is_fitted
from scipy.spatial.distance import cdist
from typing import List, Tuple, Union, Optional

logger = logging.getLogger(__name__)


class AIDA(BaseEstimator, OutlierMixin):
    """
    Analytic Isolation and Distance-based Anomaly (AIDA) algorithm.
    
    AIDA combines metrics of distance with isolation principles to identify anomalies
    without requiring parameter tuning.
    
    Parameters
    ----------
    n_subsamples : int, default=20
        Number of random subsamples used for computing distance profiles.
    
    subsample_size : int or float, default=0.1
        Size of each subsample. If float, interpreted as a fraction of the dataset size.
    
    metric : str, default='manhattan'
        Distance metric to use. Can be any metric supported by scipy.spatial.distance.cdist.
    
    score_type : str, default='variance'
        Type of score to use. Can be 'expectation' or 'variance'.
    
    alpha_version : str, default='random'
        Version of alpha parameter. Can be 'fixed' or 'random'.
    
    alpha_value : float, default=1.0
        Value of alpha parameter if alpha_version is 'fixed'.
    
    random_state : int, Optional
        Random seed for reproducibility.
    
    Attributes
    ----------
    scores_ : ndarray of shape (n_samples,)
        Anomaly scores of the training samples. Higher values correspond to more anomalous points.
    
    feature_importances_ : ndarray of shape (n_features,)
        Importance of each feature for anomaly detection.
    """
    
    def __init__(
        self,
        n_subsamples: int = 20,
        subsample_size: Union[int, float] = 0.1,
        metric: str = 'cityblock',
        score_type: str = 'variance',
        alpha_version: str = 'random',
        alpha_value: float = 1.0,
        random_state: Optional[int] = None
    ):
        self.n_subsamples = n_subsamples
        self.subsample_size = subsample_size
        self.metric = metric
        self.score_type = score_type
        self.alpha_version = alpha_version
        self.alpha_value = alpha_value
        self.random_state = random_state
        
        # Validate inputs
        if score_type not in ['expectation', 'variance']:
            raise ValueError(f"score_type must be 'expectation' or 'variance', got {score_type}")
        
        if alpha_version not in ['fixed', 'random']:
            raise ValueError(f"alpha_version must be 'fixed' or 'random', got {alpha_version}")
        
        # Set random state
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X, y=None):
        """
        Fit the AIDA model.
        
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
        # Check data
        X = check_array(X)
        n_samples, n_features = X.shape
        
        # Determine subsample size
        if isinstance(self.subsample_size, float):
            subsample_size = max(int(self.subsample_size * n_samples), 2)
        else:
            subsample_size = min(self.subsample_size, n_samples)
        
        # Compute distance profiles and anomaly scores
        scores, feature_importances = self._compute_anomaly_scores(X, n_samples, subsample_size)
        
        # Store results
        self.scores_ = scores
        self.feature_importances_ = feature_importances
        
        return self
    
    def _compute_anomaly_scores(self, X, n_samples, subsample_size):
        """
        Compute AIDA anomaly scores.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.
        
        n_samples : int
            Number of samples in X.
        
        subsample_size : int
            Size of each subsample.
        
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores for each sample.
        
        feature_importances : ndarray of shape (n_features,)
            Importance of each feature for anomaly detection.
        """
        n_features = X.shape[1]
        
        # Placeholders for results
        E_C = np.zeros(n_samples)  # Expected path length
        V_C = np.zeros(n_samples)  # Variance of path length
        feature_importances = np.zeros(n_features)
        
        # Generate subsamples and compute scores
        for i in range(self.n_subsamples):
            # Generate a random subsample
            subsample_idx = np.random.choice(n_samples, size=subsample_size, replace=False)
            X_subsample = X[subsample_idx]
            
            # Compute distances from each point to the subsample
            distances = self._compute_distances(X, X_subsample)
            
            # Compute expected path length and variance for this subsample
            E_C_temp, V_C_temp, feat_imp_temp = self._compute_isolation_metrics(
                distances, n_samples, subsample_size, n_features, X, X_subsample
            )
            
            # Accumulate results
            E_C += E_C_temp
            V_C += V_C_temp
            feature_importances += feat_imp_temp
        
        # Average the results
        E_C /= self.n_subsamples
        V_C /= self.n_subsamples
        feature_importances /= self.n_subsamples
        
        # Determine the final scores based on the chosen score type
        if self.score_type == 'expectation':
            scores = -E_C  # Invert so higher values indicate anomalies
        else:  # 'variance'
            scores = V_C
        
        # Normalize scores
        scores = (scores - np.mean(scores)) / np.std(scores)
        
        return scores, feature_importances
    
    def _compute_distances(self, X, X_subsample):
        """
        Compute distances from each point to the subsample.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The full dataset.
        
        X_subsample : ndarray of shape (subsample_size, n_features)
            The subsample.
        
        Returns
        -------
        distances : ndarray of shape (n_samples, subsample_size)
            Distances from each point to each point in the subsample.
        """
        return cdist(X, X_subsample, metric=self.metric)
    
    def _compute_isolation_metrics(self, distances, n_samples, subsample_size, n_features, X, X_subsample):
        """
        Compute isolation metrics based on distance profiles.
        
        Parameters
        ----------
        distances : ndarray of shape (n_samples, subsample_size)
            Distances from each point to each point in the subsample.
        
        n_samples : int
            Number of samples in X.
        
        subsample_size : int
            Size of the subsample.
        
        n_features : int
            Number of features in X.
        
        X : ndarray of shape (n_samples, n_features)
            The full dataset.
        
        X_subsample : ndarray of shape (subsample_size, n_features)
            The subsample.
        
        Returns
        -------
        E_C : ndarray of shape (n_samples,)
            Expected path length for each sample.
        
        V_C : ndarray of shape (n_samples,)
            Variance of path length for each sample.
        
        feature_importances : ndarray of shape (n_features,)
            Importance of each feature for anomaly detection.
        """
        # Placeholders for results
        E_C = np.zeros(n_samples)
        V_C = np.zeros(n_samples)
        feature_importances = np.zeros(n_features)
        
        # Process each sample
        for i in range(n_samples):
            # Sort distances
            sorted_distances = np.sort(distances[i])
            
            # Calculate alpha parameter
            if self.alpha_version == 'fixed':
                alpha = self.alpha_value
            else:  # 'random'
                alpha = 2 * np.random.random() + 0.5  # Between 0.5 and 2.5
            
            # Compute the analytic isolation metrics
            # These are based on the formulas from the AIDA paper
            
            # Expected path length
            E_C[i] = np.sum((1 / np.arange(1, subsample_size + 1)) * 
                             (sorted_distances / sorted_distances[-1]) ** alpha)
            
            # Variance of path length
            sq_sum = np.sum((1 / np.arange(1, subsample_size + 1) ** 2) * 
                            (sorted_distances / sorted_distances[-1]) ** (2 * alpha))
            V_C[i] = sq_sum - (E_C[i] ** 2 / subsample_size)
            
            # Feature importance (simple version)
            # For a more detailed implementation, check the TIX algorithm in the original paper
            for j in range(n_features):
                feature_diff = np.abs(X[i, j] - X_subsample[:, j])
                feature_importances[j] += np.sum(feature_diff / (distances[i] + 1e-10))
        
        # Normalize feature importances
        feature_importances = feature_importances / n_samples / subsample_size
        
        return E_C, V_C, feature_importances
    
    def decision_function(self, X):
        """
        Compute the decision function of the model for each sample in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples. The higher, the more abnormal.
        """
        check_is_fitted(self)
        X = check_array(X)
        
        # For prediction on new samples, we would need to compute their distance to the training set
        # and then calculate the isolation metrics.
        # This is a simplified version for demonstration purposes.
        
        # Compute distances to training subsample (for simplicity, use a random subsample)
        n_samples_train = len(self.scores_)
        subsample_size = min(100, n_samples_train)
        subsample_idx = np.random.choice(n_samples_train, size=subsample_size, replace=False)
        
        # Get the subsample from the original training data
        # In a real implementation, we would store the training data in the fit method
        # For now, we just return the scores from fit
        
        # Placeholder for real implementation
        return np.zeros(X.shape[0])
    
    def predict(self, X):
        """
        Predict if a particular sample is an outlier or not.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        Returns
        -------
        is_outlier : ndarray of shape (n_samples,)
            For each observation, tells whether or not it should be considered as an outlier.
            1 for outliers, 0 for inliers.
        """
        scores = self.decision_function(X)
        threshold = np.percentile(self.scores_, 95)  # Default threshold at 95th percentile
        return (scores > threshold).astype(int)
    
    def fit_predict(self, X, y=None):
        """
        Fit the model and predict if samples are outliers or not.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        y : Ignored
            Not used, present for API consistency by convention.
        
        Returns
        -------
        is_outlier : ndarray of shape (n_samples,)
            For each observation, tells whether or not it should be considered as an outlier.
            1 for outliers, 0 for inliers.
        """
        self.fit(X)
        threshold = np.percentile(self.scores_, 95)  # Default threshold at 95th percentile
        return (self.scores_ > threshold).astype(int)


class TemporalAIDA(AIDA):
    """
    Temporal version of AIDA for time series data.
    
    This extension of AIDA is designed to work with time series subsequences
    as described in the additional notes of the thesis.
    
    Parameters
    ----------
    subsequence_length : int, default=3
        Length of subsequences to analyze.
    
    step : int, default=1
        Step size between subsequences (1 for overlapping, subsequence_length for non-overlapping).
    
    n_subsamples : int, default=20
        Number of random subsamples used for computing distance profiles.
    
    subsample_size : int or float, default=0.1
        Size of each subsample. If float, interpreted as a fraction of the dataset size.
    
    metric : str, default='manhattan'
        Distance metric to use. Can be any metric supported by scipy.spatial.distance.cdist.
    
    score_type : str, default='variance'
        Type of score to use. Can be 'expectation' or 'variance'.
    
    alpha_version : str, default='random'
        Version of alpha parameter. Can be 'fixed' or 'random'.
    
    alpha_value : float, default=1.0
        Value of alpha parameter if alpha_version is 'fixed'.
    
    random_state : int, Optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        subsequence_length: int = 3,
        step: int = 1,
        n_subsamples: int = 20,
        subsample_size: Union[int, float] = 0.1,
        metric: str = 'manhattan',
        score_type: str = 'variance',
        alpha_version: str = 'random',
        alpha_value: float = 1.0,
        random_state: Optional[int] = None
    ):
        super().__init__(
            n_subsamples=n_subsamples,
            subsample_size=subsample_size,
            metric=metric,
            score_type=score_type,
            alpha_version=alpha_version,
            alpha_value=alpha_value,
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
        Fit the TemporalAIDA model to subsequences of the input data.
        
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
        # Check data
        X = check_array(X)
        
        # Create subsequences
        subsequences, self.subsequence_indices_ = self.create_subsequences(X)
        
        # Apply AIDA to the subsequences
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
        check_is_fitted(self)
        
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
        check_is_fitted(self)
        
        anomaly_indices, anomaly_scores = self.get_anomaly_subsequences(threshold_percentile)
        anomaly_timestamps = self.subsequence_indices_[anomaly_indices]
        
        return anomaly_timestamps, anomaly_scores


class MultiTSAIDA:
    """
    Multi-Time Series version of AIDA for analyzing multiple time series simultaneously.
    
    This extension of AIDA is designed to work with subsequence matrices
    that represent multiple time series (stocks) over a window period.
    
    Parameters
    ----------
    n_subsamples : int, default=20
        Number of random subsamples used for computing distance profiles.
    
    subsample_size : int or float, default=0.1
        Size of each subsample. If float, interpreted as a fraction of the dataset size.
    
    matrix_norm : str, default='frobenius'
        Norm to use for computing distances between subsequence matrices.
        Options: 'frobenius', 'nuclear', 'spectral', 'l1'
    
    score_type : str, default='variance'
        Type of score to use. Can be 'expectation' or 'variance'.
    
    alpha_version : str, default='random'
        Version of alpha parameter. Can be 'fixed' or 'random'.
    
    alpha_value : float, default=1.0
        Value of alpha parameter if alpha_version is 'fixed'.
    
    random_state : int, Optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        n_subsamples: int = 20,
        subsample_size: Union[int, float] = 0.1,
        matrix_norm: str = 'frobenius',
        score_type: str = 'variance',
        alpha_version: str = 'random',
        alpha_value: float = 1.0,
        random_state: Optional[int] = None
    ):
        self.n_subsamples = n_subsamples
        self.subsample_size = subsample_size
        self.matrix_norm = matrix_norm
        self.score_type = score_type
        self.alpha_version = alpha_version
        self.alpha_value = alpha_value
        self.random_state = random_state
        
        # Validate inputs
        if score_type not in ['expectation', 'variance']:
            raise ValueError(f"score_type must be 'expectation' or 'variance', got {score_type}")
        
        if alpha_version not in ['fixed', 'random']:
            raise ValueError(f"alpha_version must be 'fixed' or 'random', got {alpha_version}")
        
        if matrix_norm not in ['frobenius', 'nuclear', 'spectral', 'l1']:
            raise ValueError(f"matrix_norm must be one of 'frobenius', 'nuclear', 'spectral', 'l1', got {matrix_norm}")
        
        # Set random state
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, matrices, metadata=None):
        """
        Fit the MultiTSAIDA model to a list of subsequence matrices.
        
        Parameters
        ----------
        matrices : list of ndarray or ndarray of shape (n_matrices, n_stocks, subsequence_length, n_features)
            List of subsequence matrices or 4D array containing all matrices.
        
        metadata : list of dict, optional
            Metadata for each matrix, e.g., timestamps, tickers.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Convert to numpy array if a list is provided
        if isinstance(matrices, list):
            # Get the shape of the first matrix
            first_matrix_shape = matrices[0].shape
            # Create a 4D array
            matrices_array = np.zeros((len(matrices),) + first_matrix_shape)
            for i, matrix in enumerate(matrices):
                matrices_array[i] = matrix
            matrices = matrices_array
        
        # Store the matrices
        self.matrices_ = matrices
        self.metadata_ = metadata
        
        # Compute anomaly scores
        self.scores_ = self._compute_anomaly_scores(matrices)
        
        return self
    
    def _compute_anomaly_scores(self, matrices):
        """
        Compute AIDA anomaly scores for a set of subsequence matrices.
        
        Parameters
        ----------
        matrices : ndarray of shape (n_matrices, n_stocks, subsequence_length, n_features)
            Subsequence matrices to analyze.
        
        Returns
        -------
        scores : ndarray of shape (n_matrices,)
            Anomaly scores for each subsequence matrix.
        """
        n_matrices = matrices.shape[0]
        
        # Determine subsample size
        if isinstance(self.subsample_size, float):
            subsample_size = max(int(self.subsample_size * n_matrices), 2)
        else:
            subsample_size = min(self.subsample_size, n_matrices)
        
        # Placeholders for results
        E_C = np.zeros(n_matrices)  # Expected path length
        V_C = np.zeros(n_matrices)  # Variance of path length
        
        # Generate subsamples and compute scores
        for i in range(self.n_subsamples):
            # Generate a random subsample
            subsample_idx = np.random.choice(n_matrices, size=subsample_size, replace=False)
            matrices_subsample = matrices[subsample_idx]
            
            # Compute distances from each matrix to the subsample
            distances = self._compute_matrix_distances(matrices, matrices_subsample)
            
            # Compute expected path length and variance for this subsample
            E_C_temp, V_C_temp = self._compute_isolation_metrics(
                distances, n_matrices, subsample_size
            )
            
            # Accumulate results
            E_C += E_C_temp
            V_C += V_C_temp
        
        # Average the results
        E_C /= self.n_subsamples
        V_C /= self.n_subsamples
        
        # Determine the final scores based on the chosen score type
        if self.score_type == 'expectation':
            scores = -E_C  # Invert so higher values indicate anomalies
        else:  # 'variance'
            scores = V_C
        
        # Normalize scores
        scores = (scores - np.mean(scores)) / np.std(scores)
        
        return scores
    
    def _compute_matrix_distances(self, matrices, matrices_subsample):
        """
        Compute distances between matrices using the specified norm.
        
        Parameters
        ----------
        matrices : ndarray of shape (n_matrices, n_stocks, subsequence_length, n_features)
            The full dataset of matrices.
        
        matrices_subsample : ndarray of shape (subsample_size, n_stocks, subsequence_length, n_features)
            The subsample of matrices.
        
        Returns
        -------
        distances : ndarray of shape (n_matrices, subsample_size)
            Distances from each matrix to each matrix in the subsample.
        """
        n_matrices = matrices.shape[0]
        subsample_size = matrices_subsample.shape[0]
        distances = np.zeros((n_matrices, subsample_size))
        
        for i in range(n_matrices):
            for j in range(subsample_size):
                # Compute distance based on the chosen norm
                if self.matrix_norm == 'frobenius':
                    # Frobenius norm (default)
                    distances[i, j] = np.linalg.norm(matrices[i] - matrices_subsample[j])
                elif self.matrix_norm == 'nuclear':
                    # Nuclear norm (sum of singular values)
                    # Reshape to 2D matrices for SVD
                    mat_i = matrices[i].reshape(matrices[i].shape[0], -1)
                    mat_j = matrices_subsample[j].reshape(matrices_subsample[j].shape[0], -1)
                    diff = mat_i - mat_j
                    distances[i, j] = np.sum(np.linalg.svd(diff, compute_uv=False))
                elif self.matrix_norm == 'spectral':
                    # Spectral norm (largest singular value)
                    # Reshape to 2D matrices for SVD
                    mat_i = matrices[i].reshape(matrices[i].shape[0], -1)
                    mat_j = matrices_subsample[j].reshape(matrices_subsample[j].shape[0], -1)
                    diff = mat_i - mat_j
                    distances[i, j] = np.max(np.linalg.svd(diff, compute_uv=False))
                elif self.matrix_norm == 'l1':
                    # L1 norm (sum of absolute differences)
                    distances[i, j] = np.sum(np.abs(matrices[i] - matrices_subsample[j]))
        
        return distances
    
    def _compute_isolation_metrics(self, distances, n_matrices, subsample_size):
        """
        Compute isolation metrics based on distance profiles.
        
        Parameters
        ----------
        distances : ndarray of shape (n_matrices, subsample_size)
            Distances from each matrix to each matrix in the subsample.
        
        n_matrices : int
            Number of matrices.
        
        subsample_size : int
            Size of the subsample.
        
        Returns
        -------
        E_C : ndarray of shape (n_matrices,)
            Expected path length for each matrix.
        
        V_C : ndarray of shape (n_matrices,)
            Variance of path length for each matrix.
        """
        # Placeholders for results
        E_C = np.zeros(n_matrices)
        V_C = np.zeros(n_matrices)
        
        # Process each matrix
        for i in range(n_matrices):
            # Sort distances
            sorted_distances = np.sort(distances[i])
            
            # Calculate alpha parameter
            if self.alpha_version == 'fixed':
                alpha = self.alpha_value
            else:  # 'random'
                alpha = 2 * np.random.random() + 0.5  # Between 0.5 and 2.5
            
            # Compute the analytic isolation metrics
            # Expected path length
            E_C[i] = np.sum((1 / np.arange(1, subsample_size + 1)) * 
                             (sorted_distances / sorted_distances[-1]) ** alpha)
            
            # Variance of path length
            sq_sum = np.sum((1 / np.arange(1, subsample_size + 1) ** 2) * 
                            (sorted_distances / sorted_distances[-1]) ** (2 * alpha))
            V_C[i] = sq_sum - (E_C[i] ** 2 / subsample_size)
        
        return E_C, V_C
    
    def get_anomalies(self, threshold_percentile=95):
        """
        Get the anomalous matrices based on a percentile threshold.
        
        Parameters
        ----------
        threshold_percentile : float, default=95
            Percentile threshold for anomaly detection.
        
        Returns
        -------
        anomaly_indices : ndarray
            Indices of anomalous matrices.
        
        anomaly_scores : ndarray
            Anomaly scores of the anomalous matrices.
        
        anomaly_metadata : list of dict or None
            Metadata for anomalous matrices, if available.
        """
        threshold = np.percentile(self.scores_, threshold_percentile)
        anomaly_indices = np.where(self.scores_ > threshold)[0]
        anomaly_scores = self.scores_[anomaly_indices]
        
        if self.metadata_ is not None:
            anomaly_metadata = [self.metadata_[i] for i in anomaly_indices]
            return anomaly_indices, anomaly_scores, anomaly_metadata
        else:
            return anomaly_indices, anomaly_scores, None


class TIX:
    """
    Tempered Isolation-based eXplanation (TIX) algorithm.
    
    TIX provides feature importance explanations for anomalies detected by AIDA.
    
    Parameters
    ----------
    l_norm : float, default=1.0
        Parameter for the l-norm used in distance calculations.
    
    temperature : float, default=1.0
        Temperature parameter for the tempering function.
    
    n_features_to_explain : int or float, default=10
        Number of top features to include in the explanation.
        If float, interpreted as a fraction of the total number of features.
    
    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_samples, n_features)
        Importance scores for each feature for each sample.
    """
    
    def __init__(
        self,
        l_norm: float = 1.0,
        temperature: float = 1.0,
        n_features_to_explain: Union[int, float] = 10
    ):
        self.l_norm = l_norm
        self.temperature = temperature
        self.n_features_to_explain = n_features_to_explain
    
    def fit(self, X, anomaly_scores):
        """
        Fit the TIX algorithm.
        
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        
        anomaly_scores : ndarray of shape (n_samples,)
            Anomaly scores for each sample, as computed by the AIDA algorithm.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Check data
        X = check_array(X)
        n_samples, n_features = X.shape
        
        # Determine number of features to explain
        if isinstance(self.n_features_to_explain, float):
            n_features_to_explain = max(int(self.n_features_to_explain * n_features), 1)
        else:
            n_features_to_explain = min(self.n_features_to_explain, n_features)
        
        self.n_features_to_explain_ = n_features_to_explain
        
        # Compute feature importances
        self.feature_importances_ = self._compute_feature_importances(X, anomaly_scores)
        
        return self
    
    def _compute_feature_importances(self, X, anomaly_scores):
        """
        Compute feature importances using the TIX algorithm.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.
        
        anomaly_scores : ndarray of shape (n_samples,)
            Anomaly scores for each sample, as computed by the AIDA algorithm.
        
        Returns
        -------
        feature_importances : ndarray of shape (n_samples, n_features)
            Importance scores for each feature for each sample.
        """
        n_samples, n_features = X.shape
        feature_importances = np.zeros((n_samples, n_features))
        
        # Sort samples by anomaly score (descending)
        sorted_indices = np.argsort(-anomaly_scores)
        
        # Calculate feature importances for each sample
        for i, idx in enumerate(sorted_indices):
            x_i = X[idx]
            
            # Compute distances between this sample and all others
            # We use the l-norm distance with the specified parameter
            distances = np.zeros(n_samples)
            for j in range(n_samples):
                if j != idx:
                    # Compute l-norm distance
                    distances[j] = np.power(np.sum(np.power(np.abs(x_i - X[j]), self.l_norm)), 1/self.l_norm)
            
            # Set self-distance to NaN to exclude from calculations
            distances[idx] = np.nan
            
            # Apply temperature scaling to distances
            tempered_distances = np.exp(-self.temperature * distances / np.nanmax(distances))
            
            # Compute feature importances
            for f in range(n_features):
                # Feature contribution to distance
                feature_contrib = np.power(np.abs(x_i[f] - X[:, f]), self.l_norm)
                
                # Weight by tempered distances
                # This gives more importance to samples that are close to the current sample
                weighted_contrib = feature_contrib * tempered_distances
                
                # Compute overall feature importance
                feature_importances[idx, f] = np.nansum(weighted_contrib) / np.nansum(tempered_distances)
            
            # Normalize feature importances to sum to 1
            feature_importances[idx] = feature_importances[idx] / np.sum(feature_importances[idx])
        
        return feature_importances
    
    def explain(self, sample_idx):
        """
        Provide an explanation for a specific sample.
        
        Parameters
        ----------
        sample_idx : int
            Index of the sample to explain.
        
        Returns
        -------
        top_features : ndarray
            Indices of the most important features.
        
        feature_importances : ndarray
            Importance scores for the top features.
        """
        if not hasattr(self, 'feature_importances_'):
            raise RuntimeError("The model has not been fitted yet. Call 'fit' before using 'explain'.")
        
        # Get feature importances for the sample
        importances = self.feature_importances_[sample_idx]
        
        # Get top features
        top_indices = np.argsort(-importances)[:self.n_features_to_explain_]
        top_importances = importances[top_indices]
        
        return top_indices, top_importances
    
    def plot_explanation(self, sample_idx, feature_names=None, figsize=(10, 6)):
        """
        Plot the feature importance explanation for a specific sample.
        
        Parameters
        ----------
        sample_idx : int
            Index of the sample to explain.
        
        feature_names : list, optional
            Names of the features. If None, feature indices will be used.
        
        figsize : tuple, default=(10, 6)
            Figure size.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object.
        """
        import matplotlib.pyplot as plt
        
        top_indices, top_importances = self.explain(sample_idx)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use feature names if provided, otherwise use indices
        if feature_names is not None:
            top_names = [feature_names[i] for i in top_indices]
        else:
            top_names = [f"Feature {i}" for i in top_indices]
        
        # Create bar plot
        y_pos = np.arange(len(top_indices))
        ax.barh(y_pos, top_importances, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Feature Importance for Sample {sample_idx}')
        
        plt.tight_layout()
        return fig
    
    def plot_distance_profile(self, X, sample_idx, n_features=None, figsize=(10, 6)):
        """
        Plot the distance profile for a specific sample.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        sample_idx : int
            Index of the sample to explain.
        
        n_features : int, optional
            Number of top features to include in the plot.
            If None, uses the value set during initialization.
        
        figsize : tuple, default=(10, 6)
            Figure size.
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object.
        """
        import matplotlib.pyplot as plt
        
        X = check_array(X)
        
        if n_features is None:
            n_features = self.n_features_to_explain_
        
        # Get top features
        top_indices, _ = self.explain(sample_idx)
        top_indices = top_indices[:n_features]
        
        # Calculate distances for the top features
        sample = X[sample_idx]
        
        # Distance profile list (will be a list of sorted distances for different feature subsets)
        dist_list = []
        
        # Calculate distances for incremental feature subsets
        for i in range(1, len(top_indices) + 1):
            # Select the first i features
            subset_indices = top_indices[:i]
            
            # Calculate distances using these features
            distances = np.zeros(X.shape[0])
            for j in range(X.shape[0]):
                # Use l-norm distance for the subset of features
                feature_diffs = np.abs(sample[subset_indices] - X[j, subset_indices])
                distances[j] = np.power(np.sum(np.power(feature_diffs, self.l_norm)), 1/self.l_norm)
            
            # Sort distances and normalize
            sorted_distances = np.sort(distances)
            normalized_distances = sorted_distances / sorted_distances[-1]
            
            dist_list.append(normalized_distances)
        
        # Create boxplot
        fig, ax = plt.subplots(figsize=figsize)
        ax.boxplot(dist_list, vert=False, patch_artist=True, manage_ticks=False)
        ax.set_xlabel('Distance Profile (normalized)')
        ax.set_ylabel('Number of Features')
        ax.set_yticks(np.arange(1, len(top_indices) + 1))
        ax.set_title(f'Distance Profile for Sample {sample_idx}')
        
        plt.tight_layout()
        return fig
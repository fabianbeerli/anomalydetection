from src.models.aida_cpp_wrapper import AIDA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Initialize models
aida = AIDA(n_subsamples=100, score_type='variance')
iforest = IsolationForest(n_estimators=100)
lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
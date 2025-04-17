"""
Anomaly detection models.

This package includes implementations of:
- Isolation Forest
- Local Outlier Factor (LOF)
- References to AIDA (implemented in C++)
"""

from .isolation_forest import IForest, TemporalIsolationForest
from .lof import LOF, TemporalLOF

# Note: AIDA is implemented in C++ and accessed via subprocess calls

__all__ = [
    'IForest', 'TemporalIsolationForest',
    'LOF', 'TemporalLOF'
]
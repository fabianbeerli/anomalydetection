"""
Configuration parameters for the anomaly detection project.
"""
import os
from datetime import datetime
from pathlib import Path

# Project paths
ROOT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Ensure directories exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Data retrieval parameters
START_DATE = '2023-01-01'
END_DATE = '2025-03-26'
SP500_TICKER = '^GSPC'

# Top 10 S&P 500 constituents by weight (as of 2025)
TOP_SP500_CONSTITUENTS = [
    'AAPL',  # Apple Inc.
    'MSFT',  # Microsoft Corporation
    'AMZN',  # Amazon.com Inc.
    'NVDA',  # NVIDIA Corporation
    'GOOGL', # Alphabet Inc. Class A
    'GOOG',  # Alphabet Inc. Class C
    'META',  # Meta Platforms Inc.
    'BRK-B', # Berkshire Hathaway Inc. Class B
    'JPM',   # JPMorgan Chase & Co.
    'LLY'    # Eli Lilly and Company
]

# Additional 20 stocks will be randomly selected from remaining S&P 500 constituents

# Feature engineering parameters
WINDOW_SIZES = {
    'returns': 10,  # Window size for calculating return z-scores
    'volume': 10    # Window size for calculating volume z-scores
}

# Anomaly detection window around index anomalies (in trading days)
ANOMALY_WINDOW = 20  # ±20 trading days around each identified index anomaly

# News data retrieval
NEWS_SOURCE = 'https://finance.yahoo.com/news/'

# Logging configuration
LOG_LEVEL = 'INFO'
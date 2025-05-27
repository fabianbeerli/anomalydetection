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
# TOP_SP500_CONSTITUENTS = [
#     'MSFT',   # Microsoft
#     'NVDA',   # Nvidia
#     'AAPL',   # Apple Inc.
#     'AMZN',   # Amazon
#     'GOOG',   # Alphabet Inc. (Class C)
#     'GOOGL',  # Alphabet Inc. (Class A)
#     'META',   # Meta Platforms
#     'TSLA',   # Tesla, Inc.ß
#     'BRK-B',  # Berkshire Hathaway
#     'AVGO',   # Broadcom
#     'WMT',    # Walmart
#     'JPM',    # Jpmorgan Chase
#     'V',      # Visa Inc.
#     'LLY',    # Lilly (Eli)
#     'MA',     # Mastercard
#     'NFLX',   # Netflix
#     'COST',   # Costco
#     'XOM',    # Exxonmobil
#     'ORCL',   # Oracle Corporation
#     'PG',     # Procter & Gamble
#     'JNJ',    # Johnson & Johnson
#     'HD',     # Home Depotß
#     'BAC',    # Bank of America
#     'ABBV',   # Abbvie
#     'KO',     # Coca-Cola Company
#     'PLTR',   # Palantir Technologies
#     'PM',     # Philip Morris International
#     'TMUS',   # T-Mobile Us
#     'UNH',    # Unitedhealth Group
#     'CRM',    # Salesforce
# ]

# Top 10 S&P 500 constituents by weight (as of 2025)
TOP_SP500_CONSTITUENTS = [
    'MSFT',   # Microsoft
    'NVDA',   # Nvidia
    'AAPL',   # Apple Inc.
    'AMZN',   # Amazon
    'GOOG',   # Alphabet Inc. (Class C)
    'GOOGL',  # Alphabet Inc. (Class A)
    'META',   # Meta Platforms
    'TSLA',   # Tesla, Inc.
    'BRK-B',  # Berkshire Hathaway
    'AVGO',   # Broadcom
    "HSIC",   # Henry Schein
    "IPG",    # Interpublic Group of Companies
    "HII",    # Huntington Ingalls Industries
    "MGM",    # Mgm Resorts
    "MKTX",   # Marketaxess
    "PARA",   # Paramount Global
    "FRT",    # Federal Realty Investment Trust
    "NCLH",   # Norwegian Cruise Line Holdings
    "TECH",   # Bio-Techne
    "GNRC",   # Generac
    "MTCH",   # Match Group
    "LW",     # Lamb Weston
    "AES",    # Aes Corporation
    "ALB",    # Albemarle Corporation
    "CRL",    # Charles River Laboratories
    "IVZ",    # Invesco
    "MHK",    # Mohawk Industries
    "APA",    # Apa Corporation
    "CZR",    # Caesars Entertainment
    "ENPH",   # Enphase Energy

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
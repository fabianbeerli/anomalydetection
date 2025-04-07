#!/usr/bin/env python
"""
Script to retrieve S&P 500 index and constituent data.
"""
import os
import sys
import logging
from pathlib import Path

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.retrieval import (
    retrieve_sp500_index_data,
    retrieve_constituent_data,
    select_constituent_stocks
)
from src import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to retrieve all necessary data.
    """
    logger.info("Starting data retrieval process")
    
    # Retrieve S&P 500 index data
    logger.info("Retrieving S&P 500 index data")
    sp500_file = retrieve_sp500_index_data()
    
    if sp500_file:
        logger.info(f"S&P 500 index data saved to {sp500_file}")
    else:
        logger.error("Failed to retrieve S&P 500 index data")
    
    # Select constituent stocks
    logger.info("Selecting constituent stocks")
    constituents = select_constituent_stocks(top_n=10, additional_n=20)
    logger.info(f"Selected {len(constituents)} constituent stocks")
    
    # Retrieve constituent stock data
    logger.info("Retrieving constituent stock data")
    constituent_files = retrieve_constituent_data(constituents)
    
    logger.info(f"Retrieved data for {len(constituent_files)} constituent stocks")
    logger.info("Data retrieval completed")


if __name__ == "__main__":
    main()
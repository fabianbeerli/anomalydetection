"""
Data retrieval functions for downloading S&P 500 index and constituent data.
"""
import os
import logging
import random
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
from pathlib import Path

from src import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_sp500_constituents():
    """
    Get the current list of S&P 500 constituent stocks.
    
    Returns:
        list: List of S&P 500 ticker symbols
    """
    try:
        # Use Wikipedia to get the list of S&P 500 components
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        
        # Extract the ticker symbols
        tickers = []
        rows = table.find_all('tr')
        
        for row in rows[1:]:  # Skip the header row
            cells = row.find_all('td')
            if cells:
                ticker = cells[0].text.strip()
                ticker = ticker.replace('.', '-')  # Replace dots with hyphens for yfinance
                tickers.append(ticker)
        
        logger.info(f"Retrieved {len(tickers)} S&P 500 constituents")
        return tickers
    
    except Exception as e:
        logger.error(f"Error retrieving S&P 500 constituents: {e}")
        # Fallback to a predefined list if the retrieval fails
        logger.warning("Using predefined list of S&P 500 constituents")
        return config.TOP_SP500_CONSTITUENTS


def download_ticker_data(ticker, start_date, end_date, interval='1d'):
    """
    Download historical data for a given ticker.
    
    Args:
        ticker (str): Ticker symbol
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        interval (str): Data interval (e.g., '1d' for daily, '1wk' for weekly)
    
    Returns:
        pandas.DataFrame: Historical data for the ticker
    """
    try:
        logger.info(f"Downloading data for {ticker} from {start_date} to {end_date}")
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        
        if data.empty:
            logger.warning(f"No data retrieved for {ticker}")
            return None
        
        # Check if we have the expected columns
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in expected_columns:
            if col not in data.columns:
                logger.warning(f"Column {col} missing from data for {ticker}")
        
        logger.info(f"Downloaded {len(data)} records for {ticker}")
        return data
    
    except Exception as e:
        logger.error(f"Error downloading data for {ticker}: {e}")
        return None


def save_ticker_data(data, ticker, output_dir):
    """
    Save ticker data to a CSV file.
    
    Args:
        data (pandas.DataFrame): Data to save
        ticker (str): Ticker symbol
        output_dir (Path): Directory to save the file
    
    Returns:
        Path: Path to the saved file or None if operation failed
    """
    if data is None or data.empty:
        logger.warning(f"No data to save for {ticker}")
        return None
    
    try:
        # Create the output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Sanitize ticker for filename
        ticker_safe = ticker.replace('^', 'index_').replace('/', '_')
        filename = output_dir / f"{ticker_safe}.csv"
        
        # Save the data
        data.to_csv(filename)
        logger.info(f"Saved data for {ticker} to {filename}")
        return filename
    
    except Exception as e:
        logger.error(f"Error saving data for {ticker}: {e}")
        return None


def retrieve_sp500_index_data():
    """
    Retrieve S&P 500 index data for the specified time period.
    
    Returns:
        Path: Path to the saved file or None if operation failed
    """
    data = download_ticker_data(
        config.SP500_TICKER, 
        config.START_DATE, 
        config.END_DATE
    )
    return save_ticker_data(data, config.SP500_TICKER, config.RAW_DATA_DIR)


def select_constituent_stocks(top_n=10, additional_n=20):
    """
    Select constituent stocks for analysis:
    - Top N stocks by market cap
    - Additional randomly selected stocks
    
    Args:
        top_n (int): Number of top stocks to include
        additional_n (int): Number of additional random stocks to include
    
    Returns:
        list: List of selected ticker symbols
    """
    # Get all S&P 500 constituents
    all_constituents = get_sp500_constituents()
    
    # Use predefined top constituents
    top_constituents = config.TOP_SP500_CONSTITUENTS[:top_n]
    
    # Get remaining constituents
    remaining_constituents = [t for t in all_constituents if t not in top_constituents]
    
    # Randomly select additional constituents
    random.seed(42)  # For reproducibility
    additional_constituents = random.sample(remaining_constituents, min(additional_n, len(remaining_constituents)))
    
    selected_constituents = top_constituents + additional_constituents
    logger.info(f"Selected {len(selected_constituents)} constituent stocks for analysis")
    
    return selected_constituents


def retrieve_constituent_data(constituent_list=None):
    """
    Retrieve data for S&P 500 constituent stocks.
    
    Args:
        constituent_list (list, optional): List of constituents to retrieve.
            If None, selects constituents using the select_constituent_stocks function.
    
    Returns:
        list: List of paths to saved files
    """
    if constituent_list is None:
        constituent_list = select_constituent_stocks()
    
    saved_files = []
    
    for ticker in constituent_list:
        data = download_ticker_data(
            ticker, 
            config.START_DATE, 
            config.END_DATE
        )
        
        file_path = save_ticker_data(data, ticker, config.RAW_DATA_DIR)
        if file_path:
            saved_files.append(file_path)
        
        # Add a small delay to avoid hitting rate limits
        time.sleep(0.5)
    
    logger.info(f"Retrieved data for {len(saved_files)} constituent stocks")
    return saved_files


def retrieve_financial_news(ticker, date, window_days=1):
    """
    Retrieve financial news for a specific ticker around a given date.
    
    Args:
        ticker (str): Ticker symbol
        date (str or datetime): Date to search for news
        window_days (int): Number of days before and after the date to include
    
    Returns:
        list: List of dictionaries containing news information
    """
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')
    
    start_date = date - timedelta(days=window_days)
    end_date = date + timedelta(days=window_days)
    
    news_list = []
    
    try:
        # Fetch the stock object
        stock = yf.Ticker(ticker)
        
        # Get news items
        news_items = stock.news
        
        # Filter by date and add to the list
        for item in news_items:
            news_date = datetime.fromtimestamp(item['providerPublishTime'])
            if start_date <= news_date <= end_date:
                news_list.append({
                    'title': item['title'],
                    'publisher': item['publisher'],
                    'link': item['link'],
                    'publish_time': news_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'summary': item.get('summary', '')
                })
        
        logger.info(f"Retrieved {len(news_list)} news items for {ticker} around {date.strftime('%Y-%m-%d')}")
        return news_list
    
    except Exception as e:
        logger.error(f"Error retrieving news for {ticker}: {e}")
        return []


def save_news_data(news_list, ticker, date, output_dir):
    """
    Save news data to a JSON file.
    
    Args:
        news_list (list): List of dictionaries containing news information
        ticker (str): Ticker symbol
        date (str or datetime): Date associated with the news
        output_dir (Path): Directory to save the file
    
    Returns:
        Path: Path to the saved file or None if operation failed
    """
    if not news_list:
        logger.warning(f"No news data to save for {ticker}")
        return None
    
    try:
        # Create the output directory if it doesn't exist
        news_dir = output_dir / 'news'
        news_dir.mkdir(exist_ok=True, parents=True)
        
        # Format the date
        if isinstance(date, str):
            date_str = date
        else:
            date_str = date.strftime('%Y-%m-%d')
        
        # Sanitize ticker for filename
        ticker_safe = ticker.replace('^', 'index_').replace('/', '_')
        filename = news_dir / f"{ticker_safe}_{date_str}_news.json"
        
        # Save the data
        pd.DataFrame(news_list).to_json(filename, orient='records', indent=2)
        logger.info(f"Saved news data for {ticker} to {filename}")
        return filename
    
    except Exception as e:
        logger.error(f"Error saving news data for {ticker}: {e}")
        return None


def main():
    """
    Main function to retrieve all necessary data.
    """
    logger.info("Starting data retrieval process")
    
    # Retrieve S&P 500 index data
    sp500_file = retrieve_sp500_index_data()
    
    # Retrieve constituent stock data
    constituent_files = retrieve_constituent_data()
    
    logger.info("Data retrieval completed")
    return {
        'sp500_file': sp500_file,
        'constituent_files': constituent_files
    }


if __name__ == "__main__":
    main()
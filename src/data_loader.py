"""
Data Loader Module
Downloads and prepares stock data from Yahoo Finance with comprehensive error handling.
"""

import yfinance as yf
import pandas as pd
from typing import Optional, List
from datetime import datetime, timedelta

# Default stock symbols (10+ popular stocks)
DEFAULT_SYMBOLS = [
    'AAPL',   # Apple Inc.
    'GOOGL',  # Alphabet Inc.
    'MSFT',   # Microsoft Corporation
    'AMZN',   # Amazon.com Inc.
    'TSLA',   # Tesla Inc.
    'NVDA',   # NVIDIA Corporation
    'META',   # Meta Platforms Inc.
    'NFLX',   # Netflix Inc.
    'AMD',    # Advanced Micro Devices Inc.
    'JPM',    # JPMorgan Chase & Co.
    'V'       # Visa Inc.
]


def download_stock_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    max_retries: int = 3
) -> Optional[pd.DataFrame]:
    """
    Download historical stock data from Yahoo Finance.
    
    Parameters
    ----------
    symbol : str
        Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format. If None, fetches all available data.
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. If None, uses current date.
    max_retries : int, default=3
        Number of retry attempts for failed downloads
    
    Returns
    -------
    pd.DataFrame or None
        DataFrame with OHLCV data and datetime index, or None if download fails
    
    Examples
    --------
    >>> data = download_stock_data('AAPL')
    >>> print(data.head())
    """
    
    # Set default end date to today if not provided
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # If no start date, fetch all available data (yfinance handles this)
    if start_date is None:
        start_date = '1900-01-01'  # Very old date to get all available data
    
    print(f"📥 Downloading data for {symbol}...")
    print(f"   Date range: {start_date} to {end_date}")
    
    for attempt in range(max_retries):
        try:
            # Download data using yfinance
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True  # Adjust for splits and dividends
            )
            
            # Flatten MultiIndex columns if present (fix for yfinance v0.2+)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Check if data is empty
            if data.empty:
                raise ValueError(f"No data found for symbol '{symbol}'. Please check if the symbol is valid.")
            
            # Validate required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Handle missing values
            if data.isnull().any().any():
                print(f"   ⚠️  Found missing values. Applying forward fill...")
                data = data.fillna(method='ffill').fillna(method='bfill')
            
            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            print(f"   ✅ Successfully downloaded {len(data)} rows of data")
            print(f"   📊 Date range: {data.index[0].date()} to {data.index[-1].date()}")
            
            return data
            
        except Exception as e:
            print(f"   ❌ Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            
            if attempt < max_retries - 1:
                print(f"   🔄 Retrying...")
                continue
            else:
                print(f"   ⛔ All retry attempts failed for {symbol}")
                return None
    
    return None


def get_available_symbols() -> List[str]:
    """
    Get list of pre-configured stock symbols.
    
    Returns
    -------
    List[str]
        List of stock ticker symbols
    """
    return DEFAULT_SYMBOLS.copy()


def validate_symbol(symbol: str) -> bool:
    """
    Validate if a stock symbol exists by attempting to fetch recent data.
    
    Parameters
    ----------
    symbol : str
        Stock ticker symbol to validate
    
    Returns
    -------
    bool
        True if symbol is valid, False otherwise
    """
    try:
        # Try to fetch last 5 days of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        data = yf.download(
            symbol,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False
        )
        
        return not data.empty
        
    except Exception:
        return False


if __name__ == "__main__":
    # Example usage
    print("Stock Data Loader - Example Usage\n")
    
    # Test with a valid symbol
    symbol = "AAPL"
    data = download_stock_data(symbol)
    
    if data is not None:
        print(f"\n📈 Sample data for {symbol}:")
        print(data.tail())
        print(f"\n📊 Data shape: {data.shape}")
        print(f"📅 Columns: {list(data.columns)}")
    
    # Show available symbols
    print(f"\n📋 Available symbols: {', '.join(get_available_symbols())}")

"""
Feature Engineering Module
Creates technical indicators and target variable for stock direction prediction.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def calculate_sma(data: pd.DataFrame, windows: list = [10, 20, 50]) -> pd.DataFrame:
    """
    Calculate Simple Moving Averages.
    
    Parameters
    ----------
    data : pd.DataFrame
        Stock data with 'Close' column
    windows : list, default=[10, 20, 50]
        List of window sizes for SMA calculation
    
    Returns
    -------
    pd.DataFrame
        Data with added SMA columns
    """
    df = data.copy()
    
    for window in windows:
        df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
    
    return df


def calculate_ema(data: pd.DataFrame, windows: list = [12, 26]) -> pd.DataFrame:
    """
    Calculate Exponential Moving Averages.
    
    Parameters
    ----------
    data : pd.DataFrame
        Stock data with 'Close' column
    windows : list, default=[12, 26]
        List of window sizes for EMA calculation
    
    Returns
    -------
    pd.DataFrame
        Data with added EMA columns
    """
    df = data.copy()
    
    for window in windows:
        df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    
    return df


def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI).
    
    Parameters
    ----------
    data : pd.DataFrame
        Stock data with 'Close' column
    period : int, default=14
        Period for RSI calculation
    
    Returns
    -------
    pd.DataFrame
        Data with added RSI column
    """
    df = data.copy()
    
    # Calculate price changes
    delta = df['Close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    df['RSI'] = rsi
    
    return df


def calculate_momentum(data: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """
    Calculate price momentum (rate of change).
    
    Parameters
    ----------
    data : pd.DataFrame
        Stock data with 'Close' column
    period : int, default=10
        Period for momentum calculation
    
    Returns
    -------
    pd.DataFrame
        Data with added Momentum column
    """
    df = data.copy()
    
    # Momentum = Current Price - Price N periods ago
    df['Momentum'] = df['Close'].diff(period)
    
    return df


def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily percentage returns.
    
    Parameters
    ----------
    data : pd.DataFrame
        Stock data with 'Close' column
    
    Returns
    -------
    pd.DataFrame
        Data with added Returns column
    """
    df = data.copy()
    
    # Daily returns = (Close - Previous Close) / Previous Close
    df['Returns'] = df['Close'].pct_change()
    
    return df


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to the dataset.
    
    Parameters
    ----------
    data : pd.DataFrame
        Stock data with OHLCV columns
    
    Returns
    -------
    pd.DataFrame
        Data with all technical indicators added
    """
    print("🔧 Engineering features...")
    
    df = data.copy()
    
    # Add all indicators
    df = calculate_sma(df, windows=[10, 20, 50])
    print("   ✅ Added SMA (10, 20, 50)")
    
    df = calculate_ema(df, windows=[12, 26])
    print("   ✅ Added EMA (12, 26)")
    
    df = calculate_rsi(df, period=14)
    print("   ✅ Added RSI (14)")
    
    df = calculate_momentum(df, period=10)
    print("   ✅ Added Momentum (10)")
    
    df = calculate_returns(df)
    print("   ✅ Added Daily Returns")
    
    # Add volume-based features
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    print("   ✅ Added Volume SMA (20)")
    
    return df


def create_target(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary target variable for direction prediction.
    
    Target = 1 if next day's close > current close (Up)
    Target = 0 if next day's close <= current close (Down)
    
    Parameters
    ----------
    data : pd.DataFrame
        Stock data with 'Close' column
    
    Returns
    -------
    pd.DataFrame
        Data with added 'Target' column
    """
    df = data.copy()
    
    # Shift close price by -1 to get next day's price
    df['Next_Close'] = df['Close'].shift(-1)
    
    # Create binary target: 1 for Up, 0 for Down
    df['Target'] = (df['Next_Close'] > df['Close']).astype(int)
    
    print("🎯 Created target variable:")
    print(f"   Up days (1): {df['Target'].sum()}")
    print(f"   Down days (0): {(df['Target'] == 0).sum()}")
    
    # Drop the helper column
    df = df.drop('Next_Close', axis=1)
    
    return df


def prepare_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix and target vector for model training.
    
    Parameters
    ----------
    data : pd.DataFrame
        Stock data with technical indicators and target
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Feature matrix (X) and target vector (y)
    """
    df = data.copy()
    
    # Add all technical indicators
    df = add_technical_indicators(df)
    
    # Create target variable
    df = create_target(df)
    
    # Drop rows with NaN values (from indicator calculations)
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    
    print(f"\n🧹 Cleaned data:")
    print(f"   Dropped {dropped_rows} rows with NaN values")
    print(f"   Final dataset: {len(df)} rows")
    
    # Select feature columns (exclude OHLCV and target)
    feature_columns = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Target']]
    
    X = df[feature_columns]
    y = df['Target']
    
    print(f"\n📊 Feature matrix shape: {X.shape}")
    print(f"📋 Features: {list(X.columns)}")
    
    return X, y


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering - Example Usage\n")
    
    # Create sample data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    sample_data = pd.DataFrame({
        'Open': np.random.randn(len(dates)).cumsum() + 100,
        'High': np.random.randn(len(dates)).cumsum() + 102,
        'Low': np.random.randn(len(dates)).cumsum() + 98,
        'Close': np.random.randn(len(dates)).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # Prepare features
    X, y = prepare_features(sample_data)
    
    print(f"\n📈 Sample features:")
    print(X.head())
    print(f"\n🎯 Sample targets:")
    print(y.head())

"""
Asset data structure and loading functionality.

This module replicates the C++ asset struct and load_asset_data function
for loading historical price data from CSV files.
"""

import pandas as pd
from datetime import datetime
from typing import List
import os


class Asset:
    """
    Python equivalent of the C++ asset struct.
    
    Attributes:
        id (str): Asset identifier
        historical_close_price (List[float]): Training period close prices
        validation_close_price (List[float]): Validation period close prices  
        complete_close_price (List[float]): Complete period close prices
    """
    
    def __init__(self, asset_id: str):
        self.id = asset_id
        self.historical_close_price: List[float] = []
        self.validation_close_price: List[float] = []
        self.complete_close_price: List[float] = []
    
    def __repr__(self):
        return f"Asset(id='{self.id}', historical_prices={len(self.historical_close_price)}, validation_prices={len(self.validation_close_price)})"


def load_asset_data(path: str, asset_id: str, 
                   training_start_date: datetime,
                   training_end_date: datetime,
                   validation_start_date: datetime,
                   validation_end_date: datetime) -> Asset:
    """
    Load asset data from CSV file, equivalent to C++ load_asset_data function.
    
    Args:
        path: Path to CSV file
        asset_id: Asset identifier
        training_start_date: Start of training period
        training_end_date: End of training period
        validation_start_date: Start of validation period
        validation_end_date: End of validation period
    
    Returns:
        Asset object with loaded price data
    """
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Asset file not found: {path}")
    
    # Create asset object
    asset = Asset(asset_id)
    
    # Read CSV file using pandas
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file {path}: {e}")
    
    # Expected columns: Date, Open, High, Low, Close, Volume, Adj Close
    if len(df.columns) < 7:
        raise ValueError(f"CSV file {path} does not have expected columns")
    
    # Rename columns for clarity
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj_Close']
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date (ascending)
    df = df.sort_values('Date')
    
    # Filter data for complete period
    complete_mask = (df['Date'] >= training_start_date) & (df['Date'] <= validation_end_date)
    complete_data = df[complete_mask].copy()
    
    if len(complete_data) == 0:
        return asset
    
    # Filter for validation period
    validation_mask = (df['Date'] >= validation_start_date) & (df['Date'] <= validation_end_date)
    validation_data = df[validation_mask].copy()
    
    # Filter for training period
    training_mask = (df['Date'] >= training_start_date) & (df['Date'] <= training_end_date)
    training_data = df[training_mask].copy()
    
    # Extract close prices (using Adj_Close as in C++ version)
    asset.complete_close_price = complete_data['Adj_Close'].tolist()
    asset.validation_close_price = validation_data['Adj_Close'].tolist()
    asset.historical_close_price = training_data['Adj_Close'].tolist()
    
    # Reverse lists to match C++ behavior (using boost::copy with reversed)
    asset.complete_close_price.reverse()
    asset.validation_close_price.reverse()
    asset.historical_close_price.reverse()
    
    return asset


def calculate_returns(prices: List[float]) -> List[float]:
    """
    Calculate returns from price data.
    
    Args:
        prices: List of price values
    
    Returns:
        List of returns (percentage changes)
    """
    if len(prices) < 2:
        return []
    
    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] != 0:
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        else:
            returns.append(0.0)
    
    return returns


def validate_asset_data(asset: Asset) -> bool:
    """
    Validate that asset data is properly loaded.
    
    Args:
        asset: Asset object to validate
    
    Returns:
        True if data is valid, False otherwise
    """
    if not asset.historical_close_price or not asset.validation_close_price:
        return False
    
    # Check for negative prices
    if any(price < 0 for price in asset.historical_close_price + asset.validation_close_price):
        return False
    
    # Check for reasonable price ranges (not all zeros, not extremely large)
    all_prices = asset.historical_close_price + asset.validation_close_price
    if all(price == 0 for price in all_prices):
        return False
    
    if any(price > 1e6 for price in all_prices):  # Suspiciously large prices
        return False
    
    return True 
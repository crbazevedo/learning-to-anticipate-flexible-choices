#!/usr/bin/env python3
"""
FTSE Data Downloader

This script downloads updated FTSE index data to fill the gap between
the original 2012 data and current date, avoiding infinite loops and
handling errors gracefully.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class FTSEDataDownloader:
    """Robust FTSE data downloader with error handling and rate limiting."""
    
    def __init__(self, output_dir: str = "data/ftse-updated"):
        """Initialize the downloader."""
        self.output_dir = output_dir
        self.rate_limit_delay = 1.0  # 1 second between requests
        self.max_retries = 3
        self.session_timeout = 30
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Main FTSE indices to download
        self.ftse_indices = {
            '^FTSE': 'FTSE 100',
            '^FTMC': 'FTSE 250', 
            '^FTAS': 'FTSE All-Share',
            '^FTSC': 'FTSE Small Cap',
            '^FTLC': 'FTSE Large Cap',
            '^FTMC': 'FTSE Mid Cap',
            '^FTSE': 'FTSE 100 (Primary)',
            '^GDAXI': 'DAX (German)',  # For comparison
            '^FCHI': 'CAC 40 (French)',  # For comparison
        }
        
        # Alternative symbols if primary fails
        self.alternative_symbols = {
            '^FTSE': ['FTSE.L', 'UKX.L', 'FTSE100.L'],
            '^FTMC': ['FTMC.L', 'FTSE250.L'],
            '^FTAS': ['FTAS.L', 'FTSEALL.L'],
        }
    
    def download_with_retry(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Download data with retry logic and error handling.
        
        Args:
            symbol: Stock/index symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                print(f"  Attempting to download {symbol} (attempt {attempt + 1}/{self.max_retries})")
                
                # Create ticker object
                ticker = yf.Ticker(symbol)
                
                # Download data with timeout
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval='1d',
                    timeout=self.session_timeout
                )
                
                if df.empty:
                    print(f"    ⚠️  No data returned for {symbol}")
                    return None
                
                # Reset index to get Date as column
                df = df.reset_index()
                
                # Rename columns to match original format
                df = df.rename(columns={
                    'Date': 'Date',
                    'Open': 'Open',
                    'High': 'High', 
                    'Low': 'Low',
                    'Close': 'Close',
                    'Volume': 'Volume',
                    'Dividends': 'Dividends',
                    'Stock Splits': 'Stock Splits'
                })
                
                # Add Adj Close if not present
                if 'Adj Close' not in df.columns:
                    df['Adj Close'] = df['Close']
                
                # Select only required columns
                required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
                df = df[required_columns]
                
                # Convert Date to string format
                df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
                
                print(f"    ✅ Successfully downloaded {len(df)} rows for {symbol}")
                return df
                
            except Exception as e:
                print(f"    ❌ Error downloading {symbol} (attempt {attempt + 1}): {str(e)}")
                if attempt < self.max_retries - 1:
                    print(f"    ⏳ Waiting {self.rate_limit_delay * 2} seconds before retry...")
                    time.sleep(self.rate_limit_delay * 2)
                else:
                    print(f"    💀 Failed to download {symbol} after {self.max_retries} attempts")
                    return None
        
        return None
    
    def download_ftse_index(self, symbol: str, name: str, start_date: str, end_date: str) -> bool:
        """
        Download a specific FTSE index with fallback options.
        
        Args:
            symbol: Primary symbol to try
            name: Human-readable name
            start_date: Start date
            end_date: End date
            
        Returns:
            True if successful, False otherwise
        """
        print(f"\n📊 Downloading {name} ({symbol})")
        print(f"   Period: {start_date} to {end_date}")
        
        # Try primary symbol first
        df = self.download_with_retry(symbol, start_date, end_date)
        
        if df is not None:
            # Save the data
            filename = f"{name.replace(' ', '_').replace('(', '').replace(')', '')}_{start_date}_{end_date}.csv"
            filepath = os.path.join(self.output_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"   💾 Saved to: {filepath}")
            return True
        
        # Try alternative symbols if available
        if symbol in self.alternative_symbols:
            print(f"   🔄 Trying alternative symbols for {symbol}...")
            for alt_symbol in self.alternative_symbols[symbol]:
                print(f"   Trying alternative: {alt_symbol}")
                df = self.download_with_retry(alt_symbol, start_date, end_date)
                if df is not None:
                    filename = f"{name.replace(' ', '_').replace('(', '').replace(')', '')}_{start_date}_{end_date}.csv"
                    filepath = os.path.join(self.output_dir, filename)
                    df.to_csv(filepath, index=False)
                    print(f"   💾 Saved to: {filepath}")
                    return True
        
        print(f"   💀 Failed to download {name} ({symbol})")
        return False
    
    def download_all_indices(self, start_date: str = "2012-11-21", end_date: str = None) -> Dict[str, bool]:
        """
        Download all FTSE indices with proper error handling.
        
        Args:
            start_date: Start date for downloads
            end_date: End date (defaults to today)
            
        Returns:
            Dictionary of download results
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print("=" * 60)
        print("FTSE DATA DOWNLOADER")
        print("=" * 60)
        print(f"Start Date: {start_date}")
        print(f"End Date: {end_date}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Rate Limit: {self.rate_limit_delay} seconds between requests")
        print(f"Max Retries: {self.max_retries}")
        print("=" * 60)
        
        results = {}
        successful_downloads = 0
        
        for symbol, name in self.ftse_indices.items():
            success = self.download_ftse_index(symbol, name, start_date, end_date)
            results[symbol] = success
            
            if success:
                successful_downloads += 1
            
            # Rate limiting between downloads
            if symbol != list(self.ftse_indices.keys())[-1]:  # Not the last one
                print(f"   ⏳ Rate limiting: waiting {self.rate_limit_delay} seconds...")
                time.sleep(self.rate_limit_delay)
        
        # Summary
        print("\n" + "=" * 60)
        print("DOWNLOAD SUMMARY")
        print("=" * 60)
        print(f"Total indices attempted: {len(self.ftse_indices)}")
        print(f"Successful downloads: {successful_downloads}")
        print(f"Failed downloads: {len(self.ftse_indices) - successful_downloads}")
        print(f"Success rate: {successful_downloads/len(self.ftse_indices)*100:.1f}%")
        
        if successful_downloads > 0:
            print(f"\n📁 Downloaded files saved to: {self.output_dir}")
        
        return results
    
    def create_sample_data(self, num_files: int = 5) -> None:
        """
        Create sample data files for testing when real downloads fail.
        
        Args:
            num_files: Number of sample files to create
        """
        print(f"\n🔧 Creating {num_files} sample data files for testing...")
        
        # Generate sample data
        start_date = datetime(2012, 11, 21)
        end_date = datetime.now()
        
        for i in range(num_files):
            # Generate random price data
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            base_price = 5000 + i * 500  # Different base prices for each file
            
            # Create realistic price movements
            np.random.seed(42 + i)  # Different seed for each file
            returns = np.random.normal(0.0005, 0.015, len(dates))  # Daily returns
            prices = [base_price]
            
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 100))  # Minimum price of 100
            
            # Create OHLCV data
            data = []
            for j, (date, price) in enumerate(zip(dates, prices)):
                # Generate OHLC from close price
                daily_volatility = np.random.uniform(0.005, 0.02)
                open_price = price * (1 + np.random.uniform(-daily_volatility, daily_volatility))
                high_price = max(open_price, price) * (1 + np.random.uniform(0, daily_volatility))
                low_price = min(open_price, price) * (1 - np.random.uniform(0, daily_volatility))
                volume = np.random.randint(1000000, 10000000)
                
                data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Open': round(open_price, 2),
                    'High': round(high_price, 2),
                    'Low': round(low_price, 2),
                    'Close': round(price, 2),
                    'Volume': volume,
                    'Adj Close': round(price, 2)
                })
            
            df = pd.DataFrame(data)
            
            # Save sample file
            filename = f"sample_ftse_index_{i+1}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            filepath = os.path.join(self.output_dir, filename)
            df.to_csv(filepath, index=False)
            print(f"   Created: {filename}")
        
        print(f"   ✅ Sample data created in: {self.output_dir}")

def main():
    """Main function to run the FTSE data downloader."""
    downloader = FTSEDataDownloader()
    
    try:
        # Try to download real data
        results = downloader.download_all_indices()
        
        # If no successful downloads, create sample data
        successful_count = sum(results.values())
        if successful_count == 0:
            print("\n⚠️  No real data could be downloaded. Creating sample data for testing...")
            downloader.create_sample_data()
        else:
            print(f"\n✅ Successfully downloaded {successful_count} indices!")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        print("Creating sample data instead...")
        downloader.create_sample_data()

if __name__ == "__main__":
    main() 
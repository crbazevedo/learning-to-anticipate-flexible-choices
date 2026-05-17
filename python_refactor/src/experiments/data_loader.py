"""
Data Loader Module

Handles loading and preprocessing of financial data for portfolio optimization experiments.
"""

import pandas as pd
from typing import Dict, List, Any
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Data loader for portfolio optimization experiments."""
    
    def __init__(self):
        """Initialize the data loader."""
        self.data_cache = {}
    
    def load_asset_data(self, asset_files: List[str], date_range: Dict[str, str], 
                       assets: List[str]) -> pd.DataFrame:
        """
        Load asset price data from CSV files.
        
        Args:
            asset_files: List of CSV file paths
            date_range: Dictionary with 'start' and 'end' dates
            assets: List of asset names to include
            
        Returns:
            DataFrame with asset returns
        """
        if not asset_files:
            return pd.DataFrame()
        
        # Load and combine all asset data
        all_data = []
        
        for file_path in asset_files:
            try:
                # Load CSV file
                df = pd.read_csv(file_path)
                
                # Convert date column
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Filter by date range if specified
                if date_range:
                    start_date = pd.to_datetime(date_range.get('start', df['Date'].min()))
                    end_date = pd.to_datetime(date_range.get('end', df['Date'].max()))
                    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
                
                # Calculate returns
                df['Return'] = df['Close'].pct_change()
                
                # Add to combined data
                all_data.append(df[['Date', 'Return']])
                
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {str(e)}")
                continue
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Pivot to get assets as columns
        returns_df = combined_df.pivot(index='Date', columns=None, values='Return')
        
        # Rename columns to asset names
        if assets and len(assets) <= len(returns_df.columns):
            returns_df.columns = assets[:len(returns_df.columns)]
        
        return returns_df.fillna(0)
    
    def load_market_data(self, market_files: List[str], date_range: Dict[str, str]) -> pd.DataFrame:
        """
        Load market index data.
        
        Args:
            market_files: List of market data file paths
            date_range: Dictionary with 'start' and 'end' dates
            
        Returns:
            DataFrame with market returns
        """
        if not market_files:
            return pd.DataFrame()
        
        # Load market data (similar to asset data)
        return self.load_asset_data(market_files, date_range, ['Market'])
    
    def load_portfolio_data(self, data_dir: str = "data/ftse-updated") -> Dict[str, pd.DataFrame]:
        """
        Load portfolio data from the generated FTSE dataset.
        
        Args:
            data_dir: Directory containing the data files
            
        Returns:
            Dictionary with asset and market data
        """
        data_path = Path(data_dir)
        
        if not data_path.exists():
            print(f"Warning: Data directory {data_dir} does not exist")
            return {}
        
        # Find all CSV files
        csv_files = list(data_path.glob("*.csv"))
        
        if not csv_files:
            print(f"Warning: No CSV files found in {data_dir}")
            return {}
        
        # Load data
        asset_data = {}
        market_data = {}
        
        for file_path in csv_files:
            try:
                # Load CSV file
                df = pd.read_csv(file_path)
                
                # Convert date column
                df['Date'] = pd.to_datetime(df['Date'])
                
                # Calculate returns
                df['Return'] = df['Close'].pct_change()
                
                # Determine if it's an asset or market index
                filename = file_path.stem
                
                if 'FTSE' in filename or 'sample_ftse' in filename:
                    # Market index
                    market_data[filename] = df[['Date', 'Return']]
                else:
                    # Individual asset
                    asset_data[filename] = df[['Date', 'Return']]
                
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {str(e)}")
                continue
        
        return {
            'assets': asset_data,
            'market': market_data
        }
    
    def preprocess_data(self, data: Dict[str, pd.DataFrame], 
                       preprocessing_config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        Preprocess the loaded data according to configuration.
        
        Args:
            data: Raw data dictionary
            preprocessing_config: Preprocessing configuration
            
        Returns:
            Preprocessed data dictionary
        """
        processed_data = {}
        
        for data_type, df in data.items():
            if df.empty:
                continue
            
            # Apply preprocessing steps
            processed_df = df.copy()
            
            # Remove outliers if specified
            if preprocessing_config.get('remove_outliers', False):
                processed_df = self._remove_outliers(processed_df)
            
            # Fill missing values
            if preprocessing_config.get('fill_missing', True):
                processed_df = processed_df.fillna(method='ffill').fillna(0)
            
            # Normalize if specified
            if preprocessing_config.get('normalize', False):
                processed_df = self._normalize_data(processed_df)
            
            # Filter by date range
            date_range = preprocessing_config.get('date_range')
            if date_range:
                start_date = pd.to_datetime(date_range.get('start', processed_df.index.min()))
                end_date = pd.to_datetime(date_range.get('end', processed_df.index.max()))
                processed_df = processed_df[(processed_df.index >= start_date) & 
                                         (processed_df.index <= end_date)]
            
            processed_data[data_type] = processed_df
        
        return processed_data
    
    def _remove_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Remove outliers from the data."""
        if method == 'iqr':
            # Use IQR method
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Replace outliers with bounds
            for col in df.columns:
                if col != 'Date':
                    df[col] = df[col].clip(lower=lower_bound[col], upper=upper_bound[col])
        
        return df
    
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the data."""
        # Z-score normalization
        for col in df.columns:
            if col != 'Date':
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    df[col] = (df[col] - mean_val) / std_val
        
        return df
    
    def get_data_summary(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate summary statistics for the loaded data.
        
        Args:
            data: Data dictionary
            
        Returns:
            Summary statistics dictionary
        """
        summary = {}
        
        for data_type, df in data.items():
            if df.empty:
                continue
            
            summary[data_type] = {
                'shape': df.shape,
                'date_range': {
                    'start': df.index.min().isoformat() if not df.empty else None,
                    'end': df.index.max().isoformat() if not df.empty else None
                },
                'columns': list(df.columns),
                'missing_values': df.isnull().sum().to_dict(),
                'statistics': {
                    'mean': df.mean().to_dict(),
                    'std': df.std().to_dict(),
                    'min': df.min().to_dict(),
                    'max': df.max().to_dict()
                }
            }
        
        return summary 
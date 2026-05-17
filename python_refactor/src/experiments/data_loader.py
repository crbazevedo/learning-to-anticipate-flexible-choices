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

        # W9-2: glob-expand entries containing wildcard chars.
        # The paper-window data ships as 98 per-asset CSVs at
        # legacy-cpp/executable/data/ftse-original/table*.csv;
        # before W9-2 the loader treated 'table*.csv' as a literal
        # filename and silently loaded nothing.
        from pathlib import Path
        expanded: list[str] = []
        for entry in asset_files:
            p = Path(entry)
            if any(ch in p.name for ch in "*?["):
                matches = sorted(p.parent.glob(p.name))
                expanded.extend(str(m) for m in matches)
            else:
                expanded.append(entry)

        # Load and combine all asset data
        all_data = []

        for file_path in expanded:
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

                # W9-2: tag each row with an asset_id derived from the
                # filename stem; required for the post-concat pivot below
                # (pre-W9-2 columns=None → KeyError(None) at pivot time).
                df['asset_id'] = Path(file_path).stem

                # Add to combined data
                all_data.append(df[['Date', 'asset_id', 'Return']])

            except Exception as e:
                print(f"Warning: Could not load {file_path}: {str(e)}")
                continue

        if not all_data:
            return pd.DataFrame()

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # W9-4 hotfix to W9-2: real data has duplicate (Date, asset_id)
        # tuples (e.g. FTSE_100_20121121_20241231.csv has 2 rows for each
        # year-end day — a regular EOD plus a consolidated-summary entry).
        # Synthetic test fixtures didn't surface this; smoke-test did.
        # Keep last observation per (Date, asset_id) — consolidated summary
        # if present, otherwise the only row.
        # Sub-papercut #15 receipt: "synthesized fixtures must match real substrate."
        combined_df = combined_df.drop_duplicates(subset=['Date', 'asset_id'], keep='last')

        # W9-2: pivot on asset_id (the tag added above). Pre-W9-2 this
        # was pivot(columns=None) → KeyError(None) in modern pandas.
        returns_df = combined_df.pivot(index='Date', columns='asset_id', values='Return')

        # Apply the `assets` filter when supplied: keep only those columns
        # that match (and preserve the requested order). When `assets` is
        # empty / None, return all loaded asset columns.
        if assets:
            kept = [a for a in assets if a in returns_df.columns]
            if kept:
                returns_df = returns_df[kept]

        # W13-1 (closes W11-2 gap at source): pct_change crossing a
        # zero-price quote produces Inf; data-source quirks produce
        # one-day "returns" of 97999x. Sanitize at the SOURCE so
        # every downstream consumer (experiment_manager, portfolio_
        # evaluator, etc.) receives finite-bounded returns. Pre-W13-1
        # the sanitation lived only in portfolio_evaluator → algorithm's
        # _initialize_population received Inf → mean_ROI=Inf →
        # Portfolio.ROI=Inf → hypervolume=NaN (W12-CARRY-1 family).
        import numpy as _np
        returns_df = (returns_df
                       .replace([_np.inf, -_np.inf], 0.0)
                       .clip(lower=-1.0, upper=1.0)
                       .fillna(0))
        return returns_df
    
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
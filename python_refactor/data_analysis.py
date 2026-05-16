#!/usr/bin/env python3
"""
Data Analysis Script for FTSE CSV Files

This script analyzes the CSV data files to understand their structure,
time periods, and characteristics to determine the best approach for
downloading updated data.
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FTSEDataAnalyzer:
    """Analyzer for FTSE CSV data files."""
    
    def __init__(self, data_dir: str = "../legacy-cpp/executable/data/ftse-original"):
        """Initialize the analyzer with data directory."""
        self.data_dir = data_dir
        self.csv_files = []
        self.data_summary = {}
        
    def scan_files(self) -> List[str]:
        """Scan for CSV files in the data directory."""
        pattern = os.path.join(self.data_dir, "table (*).csv")
        self.csv_files = sorted(glob.glob(pattern), key=lambda x: int(x.split('(')[1].split(')')[0]))
        print(f"Found {len(self.csv_files)} CSV files")
        return self.csv_files
    
    def analyze_single_file(self, file_path: str) -> Dict:
        """Analyze a single CSV file."""
        try:
            df = pd.read_csv(file_path)
            
            # Basic info
            file_info = {
                'file': os.path.basename(file_path),
                'rows': len(df),
                'columns': list(df.columns),
                'start_date': df['Date'].min(),
                'end_date': df['Date'].max(),
                'date_range_days': (pd.to_datetime(df['Date'].max()) - pd.to_datetime(df['Date'].min())).days,
                'avg_volume': df['Volume'].mean(),
                'avg_close': df['Close'].mean(),
                'price_range': df['Close'].max() - df['Close'].min(),
                'has_missing': df.isnull().any().any(),
                'missing_count': df.isnull().sum().sum()
            }
            
            # Check if it looks like an index vs individual stock
            if df['Close'].mean() > 1000:  # Likely an index
                file_info['type'] = 'index'
            else:
                file_info['type'] = 'stock'
                
            return file_info
            
        except Exception as e:
            return {
                'file': os.path.basename(file_path),
                'error': str(e)
            }
    
    def analyze_all_files(self) -> pd.DataFrame:
        """Analyze all CSV files and return summary."""
        if not self.csv_files:
            self.scan_files()
        
        results = []
        for file_path in self.csv_files:
            result = self.analyze_single_file(file_path)
            results.append(result)
        
        self.data_summary = pd.DataFrame(results)
        return self.data_summary
    
    def identify_assets(self) -> Dict:
        """Identify different types of assets in the dataset."""
        if self.data_summary.empty:
            self.analyze_all_files()
        
        # Group by type
        indices = self.data_summary[self.data_summary['type'] == 'index']
        stocks = self.data_summary[self.data_summary['type'] == 'stock']
        
        print(f"\nAsset Analysis:")
        print(f"Indices: {len(indices)} files")
        print(f"Individual Stocks: {len(stocks)} files")
        
        # Analyze price ranges for indices
        if not indices.empty:
            print(f"\nIndex Analysis:")
            for _, row in indices.iterrows():
                print(f"  {row['file']}: {row['start_date']} to {row['end_date']} "
                      f"(Close range: {row['avg_close']:.0f} ± {row['price_range']/2:.0f})")
        
        # Analyze stocks
        if not stocks.empty:
            print(f"\nStock Analysis (sample):")
            sample_stocks = stocks.head(5)
            for _, row in sample_stocks.iterrows():
                print(f"  {row['file']}: {row['start_date']} to {row['end_date']} "
                      f"(Avg Close: {row['avg_close']:.2f})")
        
        return {
            'indices': indices,
            'stocks': stocks,
            'total_files': len(self.data_summary)
        }
    
    def analyze_time_periods(self) -> Dict:
        """Analyze the time periods covered by the data."""
        if self.data_summary.empty:
            self.analyze_all_files()
        
        # Convert dates to datetime
        self.data_summary['start_date_dt'] = pd.to_datetime(self.data_summary['start_date'])
        self.data_summary['end_date_dt'] = pd.to_datetime(self.data_summary['end_date'])
        
        # Overall time range
        overall_start = self.data_summary['start_date_dt'].min()
        overall_end = self.data_summary['end_date_dt'].max()
        
        print(f"\nTime Period Analysis:")
        print(f"Overall period: {overall_start.date()} to {overall_end.date()}")
        print(f"Total days: {(overall_end - overall_start).days}")
        print(f"Years covered: {(overall_end - overall_start).days / 365.25:.1f}")
        
        # Check for gaps
        date_ranges = []
        for _, row in self.data_summary.iterrows():
            if 'error' not in row:
                date_ranges.append((row['start_date_dt'], row['end_date_dt']))
        
        # Find common period
        common_start = max(r[0] for r in date_ranges)
        common_end = min(r[1] for r in date_ranges)
        
        print(f"Common period (all assets): {common_start.date()} to {common_end.date()}")
        print(f"Common period days: {(common_end - common_start).days}")
        
        return {
            'overall_start': overall_start,
            'overall_end': overall_end,
            'common_start': common_start,
            'common_end': common_end,
            'total_days': (overall_end - overall_start).days
        }
    
    def check_data_quality(self) -> Dict:
        """Check data quality and identify issues."""
        if self.data_summary.empty:
            self.analyze_all_files()
        
        quality_issues = []
        
        # Check for files with errors
        if 'error' in self.data_summary.columns:
            error_files = self.data_summary[self.data_summary['error'].notna()]
            if not error_files.empty:
                quality_issues.append(f"Files with errors: {len(error_files)}")
        
        # Check for missing data
        missing_data = self.data_summary[self.data_summary['missing_count'] > 0]
        if not missing_data.empty:
            quality_issues.append(f"Files with missing data: {len(missing_data)}")
        
        # Check for very short time periods
        short_periods = self.data_summary[self.data_summary['date_range_days'] < 30]
        if not short_periods.empty:
            quality_issues.append(f"Files with very short periods (<30 days): {len(short_periods)}")
        
        # Check for zero volume days
        zero_volume_files = []
        for file_path in self.csv_files[:5]:  # Check first 5 files
            try:
                df = pd.read_csv(file_path)
                zero_volume_count = (df['Volume'] == 0).sum()
                if zero_volume_count > 0:
                    zero_volume_files.append(os.path.basename(file_path))
            except:
                pass
        
        if zero_volume_files:
            quality_issues.append(f"Files with zero volume days: {len(zero_volume_files)}")
        
        print(f"\nData Quality Analysis:")
        if quality_issues:
            for issue in quality_issues:
                print(f"  ⚠️  {issue}")
        else:
            print(f"  ✅ No major quality issues detected")
        
        return {
            'issues': quality_issues,
            'error_files': len(error_files) if 'error_files' in locals() else 0,
            'missing_data_files': len(missing_data) if 'missing_data_files' in locals() else 0
        }
    
    def suggest_download_strategy(self) -> Dict:
        """Suggest strategy for downloading updated data."""
        time_analysis = self.analyze_time_periods()
        asset_analysis = self.identify_assets()
        
        print(f"\nDownload Strategy Recommendations:")
        
        # Calculate gap
        gap_start = time_analysis['overall_end']
        gap_end = datetime.now()
        gap_days = (gap_end - gap_start).days
        
        print(f"  📅 Data gap: {gap_days} days ({gap_days/365.25:.1f} years)")
        print(f"  📊 Need to download data from {gap_start.date()} to {gap_end.date()}")
        
        # Identify likely indices
        indices = asset_analysis['indices']
        if not indices.empty:
            print(f"\n  🏢 Likely Indices to update:")
            for _, row in indices.iterrows():
                print(f"    - {row['file']} (FTSE index variant)")
        
        # Suggest data sources
        print(f"\n  📈 Recommended Data Sources:")
        print(f"    - Yahoo Finance API (yfinance)")
        print(f"    - Alpha Vantage API")
        print(f"    - Quandl/Refinitiv")
        print(f"    - Bloomberg Terminal (if available)")
        
        # Suggest approach
        print(f"\n  🔄 Recommended Approach:")
        print(f"    1. Use yfinance to download FTSE indices")
        print(f"    2. Handle discontinued/new assets")
        print(f"    3. Maintain same data format (Date,Open,High,Low,Close,Volume,Adj Close)")
        print(f"    4. Validate data quality and consistency")
        
        return {
            'gap_days': gap_days,
            'gap_start': gap_start,
            'gap_end': gap_end,
            'indices_to_update': len(indices),
            'stocks_to_update': len(asset_analysis['stocks'])
        }
    
    def generate_report(self) -> str:
        """Generate a comprehensive analysis report."""
        print("=" * 60)
        print("FTSE DATA ANALYSIS REPORT")
        print("=" * 60)
        
        # Scan files
        self.scan_files()
        
        # Analyze all files
        self.analyze_all_files()
        
        # Run all analyses
        asset_analysis = self.identify_assets()
        time_analysis = self.analyze_time_periods()
        quality_analysis = self.check_data_quality()
        strategy = self.suggest_download_strategy()
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total files analyzed: {len(self.csv_files)}")
        print(f"Data period: {time_analysis['overall_start'].date()} to {time_analysis['overall_end'].date()}")
        print(f"Data gap: {strategy['gap_days']} days")
        print(f"Indices identified: {len(asset_analysis['indices'])}")
        print(f"Individual stocks: {len(asset_analysis['stocks'])}")
        
        return "Analysis complete"

def main():
    """Main function to run the analysis."""
    analyzer = FTSEDataAnalyzer()
    report = analyzer.generate_report()
    print(f"\n{report}")

if __name__ == "__main__":
    main() 
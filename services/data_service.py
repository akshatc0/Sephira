"""
Data service for loading and querying sentiment data from CSV.
"""

import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataService:
    """Service for accessing and querying sentiment data."""
    
    def __init__(self, csv_path: Path):
        """
        Initialize data service with CSV file path.
        
        Args:
            csv_path: Path to the CSV file containing sentiment data
        """
        self.csv_path = csv_path
        self.df: Optional[pd.DataFrame] = None
        self.countries: List[str] = []
        self.date_range: Tuple[str, str] = ("", "")
        
        self._load_data()
    
    def _load_data(self):
        """Load data from CSV file."""
        try:
            logger.info(f"Loading data from {self.csv_path}")
            self.df = pd.read_csv(self.csv_path)
            
            # Convert date column to datetime
            if 'date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'])
            
            # Get available countries (exclude index and date columns)
            exclude_cols = ['Unnamed: 0', 'date']
            self.countries = [col for col in self.df.columns if col not in exclude_cols]
            
            # Get date range
            if 'date' in self.df.columns and not self.df['date'].isna().all():
                min_date = self.df['date'].min()
                max_date = self.df['date'].max()
                self.date_range = (
                    min_date.strftime("%Y-%m-%d"),
                    max_date.strftime("%Y-%m-%d")
                )
            
            logger.info(f"Loaded data: {len(self.df)} rows, {len(self.countries)} countries")
            logger.info(f"Date range: {self.date_range[0]} to {self.date_range[1]}")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def get_countries(self) -> List[str]:
        """Get list of available countries."""
        return self.countries.copy()
    
    def get_date_range(self) -> Tuple[str, str]:
        """Get available date range."""
        return self.date_range
    
    def get_country_data(self, country: str, start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get time series data for a specific country.
        
        Args:
            country: Country name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with date and sentiment values
        """
        if country not in self.countries:
            raise ValueError(f"Country '{country}' not found in dataset")
        
        if self.df is None:
            raise RuntimeError("Data not loaded")
        
        # Filter by country
        result_df = self.df[['date', country]].copy()
        result_df = result_df.rename(columns={country: 'sentiment'})
        
        # Filter by date range
        if start_date:
            start_dt = pd.to_datetime(start_date)
            result_df = result_df[result_df['date'] >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            result_df = result_df[result_df['date'] <= end_dt]
        
        # Remove rows with NaN values
        result_df = result_df.dropna(subset=['sentiment'])
        
        return result_df.sort_values('date')
    
    def get_multiple_countries_data(self, countries: List[str],
                                   start_date: Optional[str] = None,
                                   end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get time series data for multiple countries.
        
        Args:
            countries: List of country names
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with date and sentiment values for each country
        """
        if self.df is None:
            raise RuntimeError("Data not loaded")
        
        for country in countries:
            if country not in self.countries:
                raise ValueError(f"Country '{country}' not found in dataset")
        
        # Select columns
        cols = ['date'] + countries
        result_df = self.df[cols].copy()
        
        # Filter by date range
        if start_date:
            start_dt = pd.to_datetime(start_date)
            result_df = result_df[result_df['date'] >= start_dt]
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            result_df = result_df[result_df['date'] <= end_dt]
        
        # Remove rows where all country values are NaN
        country_cols = [c for c in countries if c in result_df.columns]
        result_df = result_df.dropna(subset=country_cols, how='all')
        
        return result_df.sort_values('date')
    
    def get_summary_statistics(self, country: str, start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary statistics for a country.
        
        Args:
            country: Country name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            Dictionary with summary statistics
        """
        data = self.get_country_data(country, start_date, end_date)
        
        if data.empty:
            return {
                "country": country,
                "data_points": 0,
                "mean": None,
                "min": None,
                "max": None,
                "trend": None
            }
        
        values = data['sentiment'].dropna()
        
        # Calculate trend (slope of linear fit)
        trend = None
        if len(values) > 1:
            x = range(len(values))
            slope = pd.Series(x).corr(values)
            if slope > 0.1:
                trend = "increasing"
            elif slope < -0.1:
                trend = "decreasing"
            else:
                trend = "stable"
        
        return {
            "country": country,
            "data_points": len(values),
            "mean": float(values.mean()) if not values.empty else None,
            "min": float(values.min()) if not values.empty else None,
            "max": float(values.max()) if not values.empty else None,
            "trend": trend
        }
    
    def get_data_summary(self, countries: List[str], start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> str:
        """
        Get a text summary of data for LLM context.
        
        Args:
            countries: List of country names
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            Text summary suitable for LLM prompts
        """
        summaries = []
        
        for country in countries:
            stats = self.get_summary_statistics(country, start_date, end_date)
            
            summary = f"{country}: "
            if stats["data_points"] == 0:
                summary += "No data available for this period"
            else:
                trend_desc = stats["trend"] or "unknown"
                summary += f"{stats['data_points']} data points, overall trend: {trend_desc}"
            
            summaries.append(summary)
        
        date_info = ""
        if start_date and end_date:
            date_info = f"Period: {start_date} to {end_date}. "
        
        return date_info + " | ".join(summaries)


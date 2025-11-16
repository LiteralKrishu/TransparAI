"""
Real API Integration Module for TransparAI
Integrates with multiple government API endpoints
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import time
from functools import wraps
import warnings

warnings.filterwarnings('ignore')


class RateLimiter:
    """Rate limiter decorator for API calls"""
    
    def __init__(self, calls: int, period: int):
        """
        Initialize rate limiter
        
        Args:
            calls: Number of calls allowed
            period: Time period in seconds
        """
        self.calls = calls
        self.period = period
        self.calls_made = []
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            self.calls_made = [t for t in self.calls_made if t > now - self.period]
            
            if len(self.calls_made) >= self.calls:
                sleep_time = self.period - (now - self.calls_made[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            self.calls_made.append(now)
            return func(*args, **kwargs)
        
        return wrapper


class GemGovAPI:
    """Government e-Marketplace (GeM) API Integration"""
    
    BASE_URL = "https://api.gem.gov.in/public/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize GeM API client
        
        Args:
            api_key: Optional API key for authentication
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'TransparAI-SFLC.in-Hackathon',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
    
    @RateLimiter(calls=10, period=60)
    def get_contracts(self, filters: Optional[Dict] = None,
                     page: int = 1, limit: int = 100) -> Optional[Dict]:
        """
        Fetch contracts from GeM API
        
        Args:
            filters: Dictionary with filter parameters
            page: Page number
            limit: Records per page
            
        Returns:
            JSON response with contract data
        """
        try:
            params = {
                'page': page,
                'limit': limit
            }
            
            if filters:
                params.update(filters)
            
            response = self.session.get(
                f"{self.BASE_URL}/contracts",
                headers=self.headers,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"GeM API Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"GeM API Request Error: {str(e)}")
            return None
    
    @RateLimiter(calls=10, period=60)
    def get_contract_details(self, contract_id: str) -> Optional[Dict]:
        """
        Fetch specific contract details
        
        Args:
            contract_id: Contract ID
            
        Returns:
            Contract details
        """
        try:
            response = self.session.get(
                f"{self.BASE_URL}/contracts/{contract_id}",
                headers=self.headers,
                timeout=30
            )
            
            return response.json() if response.status_code == 200 else None
            
        except Exception as e:
            print(f"Error fetching contract {contract_id}: {str(e)}")
            return None
    
    @RateLimiter(calls=5, period=60)
    def search_vendors(self, vendor_name: str) -> Optional[Dict]:
        """
        Search for vendors
        
        Args:
            vendor_name: Vendor name to search
            
        Returns:
            Search results
        """
        try:
            response = self.session.get(
                f"{self.BASE_URL}/vendors/search",
                headers=self.headers,
                params={'name': vendor_name},
                timeout=30
            )
            
            return response.json() if response.status_code == 200 else None
            
        except Exception as e:
            print(f"Error searching vendors: {str(e)}")
            return None
    
    @RateLimiter(calls=5, period=60)
    def get_vendor_details(self, vendor_id: str) -> Optional[Dict]:
        """
        Fetch vendor details
        
        Args:
            vendor_id: Vendor ID
            
        Returns:
            Vendor details
        """
        try:
            response = self.session.get(
                f"{self.BASE_URL}/vendors/{vendor_id}",
                headers=self.headers,
                timeout=30
            )
            
            return response.json() if response.status_code == 200 else None
            
        except Exception as e:
            print(f"Error fetching vendor {vendor_id}: {str(e)}")
            return None


class DataGovInAPI:
    """Data.gov.in API Integration"""
    
    BASE_URL = "https://api.data.gov.in/resource"
    
    def __init__(self, api_key: str):
        """
        Initialize Data.gov.in API client
        
        Args:
            api_key: API key for authentication
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'TransparAI-SFLC.in-Hackathon',
            'Accept': 'application/json'
        }
    
    @RateLimiter(calls=5, period=60)
    def get_resource(self, resource_id: str, filters: Optional[Dict] = None,
                    limit: int = 100, offset: int = 0) -> Optional[Dict]:
        """
        Fetch data from government resource
        
        Args:
            resource_id: Resource ID
            filters: Optional filters
            limit: Records limit
            offset: Record offset
            
        Returns:
            Resource data
        """
        try:
            params = {
                'api-key': self.api_key,
                'format': 'json',
                'limit': limit,
                'offset': offset
            }
            
            if filters:
                params.update(filters)
            
            response = self.session.get(
                f"{self.BASE_URL}/{resource_id}",
                headers=self.headers,
                params=params,
                timeout=30
            )
            
            return response.json() if response.status_code == 200 else None
            
        except Exception as e:
            print(f"Data.gov.in API Error: {str(e)}")
            return None


class ProcurementDataAggregator:
    """Aggregates data from multiple sources"""
    
    def __init__(self, gem_api: Optional[GemGovAPI] = None,
                 datagov_api: Optional[DataGovInAPI] = None):
        """
        Initialize data aggregator
        
        Args:
            gem_api: GeM API client
            datagov_api: Data.gov.in API client
        """
        self.gem_api = gem_api
        self.datagov_api = datagov_api
    
    def fetch_and_aggregate_data(self, start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None,
                                max_records: int = 500) -> pd.DataFrame:
        """
        Fetch and aggregate data from all sources
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            max_records: Maximum records to fetch
            
        Returns:
            Aggregated DataFrame
        """
        all_data = []
        
        # Fetch from GeM API
        if self.gem_api:
            gem_data = self._fetch_from_gem(max_records)
            if gem_data is not None:
                all_data.append(gem_data)
        
        # Fetch from Data.gov.in
        if self.datagov_api:
            datagov_data = self._fetch_from_datagov(max_records)
            if datagov_data is not None:
                all_data.append(datagov_data)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Filter by date if provided
            if start_date and end_date:
                if 'post_date' in combined_df.columns:
                    combined_df = combined_df[
                        (combined_df['post_date'] >= start_date) &
                        (combined_df['post_date'] <= end_date)
                    ]
            
            return combined_df
        
        return pd.DataFrame()
    
    def _fetch_from_gem(self, max_records: int) -> Optional[pd.DataFrame]:
        """Fetch data from GeM API"""
        try:
            records = []
            page = 1
            
            while len(records) < max_records:
                response = self.gem_api.get_contracts(page=page, limit=100)
                
                if not response or 'records' not in response:
                    break
                
                for record in response.get('records', []):
                    records.append({
                        'contract_id': record.get('id', 'N/A'),
                        'vendor_name': record.get('vendor_name', 'N/A'),
                        'contract_value': float(record.get('value', 0)),
                        'post_date': pd.to_datetime(record.get('post_date')),
                        'award_date': pd.to_datetime(record.get('award_date')),
                        'category': record.get('category', 'N/A'),
                        'status': record.get('status', 'N/A'),
                        'source': 'GeM'
                    })
                
                if len(records) >= max_records:
                    break
                
                page += 1
            
            return pd.DataFrame(records) if records else None
            
        except Exception as e:
            print(f"Error fetching from GeM: {str(e)}")
            return None
    
    def _fetch_from_datagov(self, max_records: int) -> Optional[pd.DataFrame]:
        """Fetch data from Data.gov.in"""
        try:
            # This would require specific resource IDs from data.gov.in
            # Implementation depends on available procurement datasets
            return None
            
        except Exception as e:
            print(f"Error fetching from Data.gov.in: {str(e)}")
            return None


class DataValidator:
    """Validates and cleans procurement data"""
    
    @staticmethod
    def validate_contract_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate contract data quality
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []
        
        required_fields = ['contract_id', 'vendor_name', 'contract_value',
                          'post_date', 'award_date']
        
        for field in required_fields:
            if field not in df.columns:
                errors.append(f"Missing required field: {field}")
            elif df[field].isnull().sum() > 0:
                errors.append(f"Missing values in field: {field}")
        
        if 'contract_value' in df.columns:
            if (df['contract_value'] < 0).any():
                errors.append("Negative contract values found")
        
        if 'post_date' in df.columns and 'award_date' in df.columns:
            invalid_dates = (df['award_date'] < df['post_date']).sum()
            if invalid_dates > 0:
                errors.append(f"Award date before post date in {invalid_dates} records")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def clean_contract_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize contract data
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Convert dates
        date_columns = ['post_date', 'award_date', 'contract_date']
        for col in date_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
        
        # Clean numeric columns
        numeric_columns = ['contract_value', 'estimated_value', 'bid_price']
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Remove rows with critical missing values
        required_fields = ['contract_id', 'vendor_name', 'contract_value']
        df_clean = df_clean.dropna(subset=required_fields)
        
        # Calculate processing days if dates available
        if 'post_date' in df_clean.columns and 'award_date' in df_clean.columns:
            df_clean['processing_days'] = (
                df_clean['award_date'] - df_clean['post_date']
            ).dt.days
        
        return df_clean
    
    @staticmethod
    def get_data_quality_report(df: pd.DataFrame) -> Dict:
        """
        Generate data quality report
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        report = {
            'total_records': len(df),
            'duplicate_records': df.duplicated().sum(),
            'unique_vendors': df['vendor_name'].nunique() if 'vendor_name' in df.columns else 0,
            'date_range': None,
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        if 'post_date' in df.columns:
            report['date_range'] = {
                'start': df['post_date'].min().isoformat() if df['post_date'].min() else None,
                'end': df['post_date'].max().isoformat() if df['post_date'].max() else None
            }
        
        return report

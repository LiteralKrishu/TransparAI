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
                                max_records: int = 500) -> Tuple[pd.DataFrame, list]:
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
        errors = []

        # Fetch from GeM API
        if self.gem_api:
            gem_data, gem_errors = self._fetch_from_gem(max_records)
            if gem_errors:
                errors.extend(gem_errors)
            if gem_data is not None and not gem_data.empty:
                all_data.append(gem_data)

        # Fetch from Data.gov.in
        if self.datagov_api:
            datagov_data, datagov_errors = self._fetch_from_datagov(max_records)
            if datagov_errors:
                errors.extend(datagov_errors)
            if datagov_data is not None and not datagov_data.empty:
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

            return combined_df, errors

        # nothing fetched
        return pd.DataFrame(), errors
    
    def _fetch_from_gem(self, max_records: int) -> Tuple[Optional[pd.DataFrame], list]:
        """Fetch data from GeM API with defensive parsing

        Returns tuple (DataFrame or None, list_of_error_messages)
        """
        errors = []
        try:
            records = []
            page = 1

            while len(records) < max_records:
                response = self.gem_api.get_contracts(page=page, limit=100)

                if not response:
                    errors.append(f"GeM response empty or error on page {page}")
                    break

                # response may use different keys depending on API version
                candidates = []
                if isinstance(response, dict):
                    # try common keys
                    if 'records' in response:
                        candidates = response.get('records', [])
                    elif 'data' in response:
                        candidates = response.get('data', [])
                    elif 'result' in response:
                        # some APIs wrap in result->data
                        r = response.get('result')
                        if isinstance(r, dict) and 'data' in r:
                            candidates = r.get('data', [])
                        elif isinstance(r, list):
                            candidates = r
                    elif 'contracts' in response:
                        candidates = response.get('contracts', [])

                if not candidates:
                    # nothing to parse
                    break

                for record in candidates:
                    try:
                        contract_id = record.get('id') or record.get('contract_id') or record.get('contractNo') or 'N/A'
                        vendor_name = record.get('vendor_name') or record.get('vendor') or record.get('supplier') or 'N/A'
                        value = record.get('value') or record.get('contract_value') or record.get('estimated_value') or 0

                        post_date = record.get('post_date') or record.get('publish_date') or record.get('created_at')
                        award_date = record.get('award_date') or record.get('award_on') or record.get('awarded_at')

                        records.append({
                            'contract_id': contract_id,
                            'vendor_name': vendor_name,
                            'contract_value': pd.to_numeric(value, errors='coerce') or 0,
                            'post_date': post_date,
                            'award_date': award_date,
                            'category': record.get('category', 'N/A'),
                            'status': record.get('status', 'N/A'),
                            'source': 'GeM'
                        })
                    except Exception as rec_e:
                        errors.append(f"Error parsing GeM record: {str(rec_e)}")

                if len(records) >= max_records:
                    break

                page += 1

            df = pd.DataFrame(records) if records else pd.DataFrame()

            # convert dates safely
            for col in ['post_date', 'award_date']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')

            # ensure numeric
            if 'contract_value' in df.columns:
                df['contract_value'] = pd.to_numeric(df['contract_value'], errors='coerce').fillna(0)

            return df, errors

        except Exception as e:
            errors.append(f"Error fetching from GeM: {str(e)}")
            return pd.DataFrame(), errors
    
    def _fetch_from_datagov(self, max_records: int) -> Tuple[Optional[pd.DataFrame], list]:
        """Fetch data from Data.gov.in

        Returns tuple (DataFrame, list_of_errors)
        """
        errors = []
        try:
            # This would require specific resource IDs from data.gov.in
            # Implementation depends on available procurement datasets
            # For now return empty dataframe and a helpful message
            errors.append("Data.gov.in integration not configured (no resource_id).")
            return pd.DataFrame(), errors

        except Exception as e:
            errors.append(f"Error fetching from Data.gov.in: {str(e)}")
            return pd.DataFrame(), errors


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

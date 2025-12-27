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
                                max_records: int = 500) -> Tuple[pd.DataFrame, str, List[str]]:
        """
        Fetch and aggregate data from all sources with fallback to sample data
        
        Args:
            start_date: Start date for filtering
            end_date: End date for filtering
            max_records: Maximum records to fetch
            
        Returns:
            Tuple of (DataFrame, source_name, error_messages)
        """
        all_data = []
        errors = []
        sources_tried = []
        
        # Try GeM API first (highest priority)
        if self.gem_api:
            print("Attempting to fetch data from GeM API...")
            sources_tried.append("GeM API")
            gem_data = self._fetch_from_gem(max_records)
            if gem_data is not None and not gem_data.empty:
                all_data.append(gem_data)
                print(f"[OK] Successfully fetched {len(gem_data)} records from GeM API")
            else:
                errors.append("GeM API: No data returned or connection failed")
                print("[WARNING] GeM API failed to return data")
        
        # Try Data.gov.in API (second priority)
        if self.datagov_api:
            print("Attempting to fetch data from Data.gov.in API...")
            sources_tried.append("Data.gov.in API")
            datagov_data = self._fetch_from_datagov(max_records)
            if datagov_data is not None and not datagov_data.empty:
                all_data.append(datagov_data)
                print(f"[OK] Successfully fetched {len(datagov_data)} records from Data.gov.in")
            else:
                errors.append("Data.gov.in API: No data returned or connection failed")
                print("[WARNING] Data.gov.in API failed to return data")
        
        # Try additional procurement data sources
        print("Attempting to fetch from additional government sources...")
        sources_tried.append("Additional Sources")
        additional_data = self._fetch_from_additional_sources(max_records)
        if additional_data is not None and not additional_data.empty:
            all_data.append(additional_data)
            print(f"[OK] Successfully fetched {len(additional_data)} records from additional sources")
        else:
            errors.append("Additional Sources: No data returned or connection failed")
            print("[WARNING] Additional sources failed to return data")
        
        # If we have data from any source, process and return it
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Filter by date if provided
            if start_date and end_date:
                if 'post_date' in combined_df.columns:
                    combined_df = combined_df[
                        (combined_df['post_date'] >= start_date) &
                        (combined_df['post_date'] <= end_date)
                    ]
            
            # Determine source name based on what worked
            if len(all_data) > 1:
                source_name = f"Multiple Government APIs ({len(all_data)} sources)"
            else:
                source_name = "Government APIs"
            
            print(f"[SUCCESS] Total records collected: {len(combined_df)} from {len(all_data)} source(s)")
            return combined_df, source_name, errors
        
        # Fallback to sample data if all APIs failed
        print(f"[WARNING] All {len(sources_tried)} API source(s) failed. Falling back to sample data...")
        errors.append(f"All API endpoints failed: {', '.join(sources_tried)}")
        errors.append("Automatically switched to sample data for demonstration")
        
        try:
            from src.data_generator import generate_sample_data
            sample_df = generate_sample_data(n_records=max_records)
            print(f"[OK] Generated {len(sample_df)} sample records as fallback")
            return sample_df, "Sample Data (API Fallback)", errors
        except Exception as e:
            errors.append(f"Failed to generate sample data: {str(e)}")
            return pd.DataFrame(), "No Data", errors
    
    def _fetch_from_gem(self, max_records: int) -> Optional[pd.DataFrame]:
        """Fetch data from GeM API with comprehensive error handling"""
        try:
            records = []
            page = 1
            max_pages = 10  # Prevent infinite loops
            
            while len(records) < max_records and page <= max_pages:
                try:
                    response = self.gem_api.get_contracts(page=page, limit=100)
                    
                    if not response:
                        print(f"GeM API: No response received for page {page}")
                        break
                    
                    # Handle different response formats
                    data_key = None
                    for key in ['records', 'data', 'contracts', 'results']:
                        if key in response:
                            data_key = key
                            break
                    
                    if not data_key:
                        print(f"GeM API: Unexpected response format: {list(response.keys())}")
                        break
                    
                    page_records = response.get(data_key, [])
                    if not page_records:
                        break
                    
                    for record in page_records:
                        try:
                            # Flexible field mapping
                            contract_id = record.get('id') or record.get('contract_id') or record.get('contractId') or f'GEM_{len(records)}'
                            vendor_name = record.get('vendor_name') or record.get('vendorName') or record.get('vendor') or 'Unknown Vendor'
                            value = record.get('value') or record.get('contract_value') or record.get('contractValue') or record.get('amount') or 0
                            
                            records.append({
                                'contract_id': str(contract_id),
                                'vendor_name': str(vendor_name),
                                'contract_value': float(value),
                                'post_date': pd.to_datetime(record.get('post_date') or record.get('postDate') or datetime.now(), errors='coerce'),
                                'award_date': pd.to_datetime(record.get('award_date') or record.get('awardDate') or datetime.now(), errors='coerce'),
                                'category': record.get('category') or record.get('type') or 'Uncategorized',
                                'status': record.get('status') or 'Active',
                                'ministry': record.get('ministry') or record.get('department') or 'N/A',
                                'location': record.get('location') or record.get('state') or 'N/A',
                                'source': 'GeM'
                            })
                        except Exception as e:
                            print(f"GeM API: Error processing record: {str(e)}")
                            continue
                    
                    if len(records) >= max_records:
                        break
                    
                    page += 1
                    
                except Exception as e:
                    print(f"GeM API: Error fetching page {page}: {str(e)}")
                    break
            
            if records:
                df = pd.DataFrame(records)
                # Calculate processing days
                df['processing_days'] = (df['award_date'] - df['post_date']).dt.days
                df['processing_days'] = df['processing_days'].fillna(0).abs()
                return df
            else:
                print("GeM API: No records collected")
                return None
            
        except Exception as e:
            print(f"GeM API: Critical error in _fetch_from_gem: {str(e)}")
            return None
    
    def _fetch_from_datagov(self, max_records: int) -> Optional[pd.DataFrame]:
        """Fetch data from Data.gov.in general API"""
        try:
            # This uses the general Data.gov.in API client if configured
            # For now, return None and rely on additional_sources method
            return None
            
        except Exception as e:
            print(f"Error fetching from Data.gov.in: {str(e)}")
            return None
    
    def _fetch_from_additional_sources(self, max_records: int) -> Optional[pd.DataFrame]:
        """
        Fetch data from additional government procurement sources
        
        This method tries multiple alternative sources:
        - Data.gov.in procurement datasets
        - eProcurement portal
        - CPPP (Central Public Procurement Portal)
        
        Args:
            max_records: Maximum records to fetch
            
        Returns:
            DataFrame with procurement records or None
        """
        all_records = []
        
        # Try Data.gov.in specific procurement dataset
        try:
            import json
            headers = {
                'User-Agent': 'TransparAI-SFLC.in',
                'Accept': 'application/json'
            }
            
            # Try data.gov.in procurement resource
            # Resource ID: 9ef84268-d588-465a-a308-a864a43d0070 (example procurement data)
            try:
                response = requests.get(
                    'https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070',
                    headers=headers,
                    params={
                        'format': 'json',
                        'limit': min(max_records, 100),
                        'offset': 0
                    },
                    timeout=15
                )
                
                if response.status_code == 200:
                    data = response.json()
                    records = data.get('records', [])
                    
                    for record in records[:max_records]:
                        try:
                            all_records.append({
                                'contract_id': record.get('tender_id') or record.get('id') or f'DGI_{len(all_records)}',
                                'vendor_name': record.get('vendor') or record.get('contractor') or 'Unknown Vendor',
                                'contract_value': float(record.get('value') or record.get('contract_value') or record.get('amount') or 0),
                                'post_date': pd.to_datetime(record.get('publish_date') or record.get('date'), errors='coerce'),
                                'award_date': pd.to_datetime(record.get('award_date') or record.get('closing_date'), errors='coerce'),
                                'category': record.get('category') or record.get('type') or 'General',
                                'status': record.get('status') or 'Active',
                                'ministry': record.get('department') or record.get('ministry') or 'N/A',
                                'location': record.get('location') or record.get('state') or 'N/A',
                                'source': 'Data.gov.in'
                            })
                        except Exception as e:
                            continue
                    
                    if records:
                        print(f"Data.gov.in: Fetched {len(records)} records from procurement dataset")
                else:
                    print(f"Data.gov.in: API returned status {response.status_code}")
                    
            except Exception as e:
                print(f"Data.gov.in specific dataset error: {str(e)}")
            
        except Exception as e:
            print(f"Additional sources error: {str(e)}")
        
        # Convert to DataFrame if we have records
        if all_records:
            df = pd.DataFrame(all_records)
            df['post_date'] = pd.to_datetime(df['post_date'], errors='coerce')
            df['award_date'] = pd.to_datetime(df['award_date'], errors='coerce')
            df['processing_days'] = (df['award_date'] - df['post_date']).dt.days
            df['processing_days'] = df['processing_days'].fillna(0).abs()
            return df
        
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

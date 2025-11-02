import pandas as pd
import numpy as np
from datetime import datetime
import requests
import json

class DataProcessor:
    """Utility class for data processing operations"""
    
    @staticmethod
    def clean_procurement_data(df):
        """Clean and preprocess procurement data"""
        df = df.drop_duplicates(subset=['contract_id'])
        df['contract_value'] = df['contract_value'].fillna(0)
        df['vendor_name'] = df['vendor_name'].fillna('Unknown')
        
        date_columns = ['post_date', 'award_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df
    
    @staticmethod
    def calculate_metrics(df):
        """Calculate key procurement metrics"""
        metrics = {
            'total_contracts': len(df),
            'total_value': df['contract_value'].sum(),
            'avg_contract_value': df['contract_value'].mean(),
            'unique_vendors': df['vendor_name'].nunique(),
            'avg_processing_days': df['processing_days'].mean() if 'processing_days' in df.columns else 0
        }
        return metrics

class APIClient:
    """Client for government API interactions"""
    
    def __init__(self, config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TransparAI-SFLC.in/1.0',
            'Accept': 'application/json'
        })
    
    def fetch_gem_contracts(self, limit=100):
        """Fetch contracts from GeM API"""
        try:
            url = f"{self.config['api_endpoints']['gem']}?limit={limit}"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching GeM data: {e}")
            return None

class VisualizationHelper:
    """Helper class for creating consistent visualizations"""
    
    def __init__(self, primary_color="#1486c9"):
        self.primary_color = primary_color
        self.color_scheme = [primary_color, '#ff6b6b', '#51cf66', '#fcc419', '#ae3ec9']
    
    def create_metric_card(self, value, title, delta=None):
        """Create standardized metric card data"""
        return {
            'value': value,
            'title': title,
            'delta': delta
        }

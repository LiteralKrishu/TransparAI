"""
Data Generator Module for TransparAI
Generates realistic sample procurement data for testing and demonstration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from typing import Optional


def generate_sample_data(n_records: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic sample procurement data
    
    Args:
        n_records: Number of records to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with sample procurement data
    """
    np.random.seed(seed)
    
    vendors = [
        "Tata Consultancy Services", "Infosys Limited", "Wipro Limited", 
        "HCL Technologies", "Tech Mahindra", "Larsen & Toubro Infotech",
        "Bharat Electronics Limited", "ITC Limited", "Reliance Industries",
        "Adani Enterprises", "Small Scale Vendor Pvt Ltd", "Medium Enterprise Solutions",
        "National IT Services", "Digital India Corp", "Public Sector Undertaking IT",
        "Accenture India", "Capgemini India", "IBM India", "Deloitte India",
        "Cognizant", "Mindtree", "KPMG India"
    ]
    
    categories = [
        "IT Infrastructure", "Software Development", "Consulting Services",
        "Hardware Procurement", "Network Security", "Cloud Services",
        "Digital Transformation", "Maintenance & Support", "Training & Development",
        "Data Analytics", "Business Process Outsourcing", "Infrastructure"
    ]
    
    ministries = [
        "Ministry of Electronics & IT", "Ministry of Defence", "Ministry of Health",
        "Ministry of Education", "Ministry of Transport", "Ministry of Finance",
        "Ministry of Railways", "Ministry of Energy", "Ministry of Communications"
    ]
    
    locations = [
        "New Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata",
        "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow"
    ]
    
    data = []
    base_date = datetime.now() - timedelta(days=730)
    
    for i in range(n_records):
        base_value = np.random.lognormal(14, 1.2)
        contract_value = max(50000, base_value)
        
        # Add some high-value outliers (5%)
        if np.random.random() < 0.05:
            contract_value *= np.random.uniform(5, 20)
        
        post_date = base_date + timedelta(days=np.random.randint(0, 730))
        
        # Processing time (1-90 days normally, with outliers)
        if np.random.random() < 0.05:
            processing_days = np.random.randint(91, 365)
        else:
            processing_days = np.random.randint(1, 90)
        
        award_date = post_date + timedelta(days=processing_days)
        
        vendor = np.random.choice(vendors)
        category = np.random.choice(categories)
        
        # Create some correlations - certain vendors get certain categories
        if vendor in ["Tata Consultancy Services", "Infosys Limited", "Wipro Limited"]:
            if np.random.random() < 0.7:
                category = np.random.choice(["IT Infrastructure", "Software Development", "Cloud Services"])
        
        data.append({
            'contract_id': f'CON{100000 + i}',
            'vendor_name': vendor,
            'category': category,
            'contract_value': contract_value,
            'post_date': post_date,
            'award_date': award_date,
            'ministry': np.random.choice(ministries),
            'location': np.random.choice(locations),
            'status': np.random.choice(['Completed', 'Ongoing', 'Awarded']),
            'processing_days': processing_days,
            'estimated_value': contract_value * np.random.uniform(0.8, 1.5),
            'actual_cost': contract_value * np.random.uniform(0.9, 1.1)
        })
    
    df = pd.DataFrame(data)
    
    # Ensure data types
    df['post_date'] = pd.to_datetime(df['post_date'])
    df['award_date'] = pd.to_datetime(df['award_date'])
    df['contract_value'] = df['contract_value'].astype(float)
    df['estimated_value'] = df['estimated_value'].astype(float)
    df['actual_cost'] = df['actual_cost'].astype(float)
    
    return df.sort_values('post_date').reset_index(drop=True)


def generate_monthly_data(n_months: int = 24) -> pd.DataFrame:
    """
    Generate aggregated monthly procurement data
    
    Args:
        n_months: Number of months to generate
        
    Returns:
        DataFrame with monthly aggregated data
    """
    monthly_data = []
    
    for month_offset in range(n_months):
        month_date = datetime.now() - timedelta(days=30 * (n_months - month_offset))
        
        monthly_data.append({
            'month': month_date.strftime('%Y-%m'),
            'total_contracts': np.random.randint(50, 200),
            'total_value': np.random.uniform(5e7, 5e8),
            'avg_processing_days': np.random.uniform(15, 45),
            'unique_vendors': np.random.randint(10, 40),
            'anomalies_detected': np.random.randint(0, 20)
        })
    
    return pd.DataFrame(monthly_data)


def generate_vendor_performance_data(n_vendors: int = 20) -> pd.DataFrame:
    """
    Generate vendor performance metrics
    
    Args:
        n_vendors: Number of vendors
        
    Returns:
        DataFrame with vendor performance data
    """
    vendor_names = [
        "Vendor A", "Vendor B", "Vendor C", "Vendor D", "Vendor E",
        "Vendor F", "Vendor G", "Vendor H", "Vendor I", "Vendor J",
        "Vendor K", "Vendor L", "Vendor M", "Vendor N", "Vendor O",
        "Vendor P", "Vendor Q", "Vendor R", "Vendor S", "Vendor T"
    ]
    
    vendor_data = []
    
    for i in range(min(n_vendors, len(vendor_names))):
        vendor_data.append({
            'vendor_name': vendor_names[i],
            'total_contracts': np.random.randint(5, 100),
            'total_value': np.random.uniform(1e7, 1e9),
            'avg_value_per_contract': np.random.uniform(1e6, 5e7),
            'on_time_delivery_rate': np.random.uniform(75, 100),
            'cost_overrun_ratio': np.random.uniform(0.9, 1.15),
            'quality_score': np.random.uniform(70, 100),
            'repeat_contract_count': np.random.randint(0, 20),
            'performance_rating': np.random.choice(['A+', 'A', 'B+', 'B', 'C'])
        })
    
    return pd.DataFrame(vendor_data)


def generate_category_analysis_data() -> pd.DataFrame:
    """
    Generate procurement analysis by category
    
    Returns:
        DataFrame with category-wise analysis
    """
    categories = [
        "IT Services", "Infrastructure", "Consulting", "Hardware",
        "Software", "Services", "Security", "Training"
    ]
    
    category_data = []
    
    for category in categories:
        category_data.append({
            'category': category,
            'total_contracts': np.random.randint(20, 150),
            'total_spend': np.random.uniform(5e7, 5e8),
            'avg_contract_value': np.random.uniform(1e6, 1e8),
            'unique_vendors': np.random.randint(5, 30),
            'hhi_index': np.random.uniform(500, 3000),
            'efficiency_score': np.random.uniform(60, 95),
            'anomaly_percentage': np.random.uniform(2, 15)
        })
    
    return pd.DataFrame(category_data)


def save_sample_data():
    """Generate and save sample data"""
    df = generate_sample_data(500)
    
    df.to_csv('data/sample_procurement_data.csv', index=False)
    
    json_data = df.to_dict('records', orient='records')
    with open('data/sample_procurement_data.json', 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    
    print(f"Generated sample data with {len(df)} records")
    print(f"Total contract value: INR {df['contract_value'].sum()/1e7:.2f} Crores")
    print(f"Average contract value: INR {df['contract_value'].mean()/1e5:.2f} Lakhs")
    print(f"Unique vendors: {df['vendor_name'].nunique()}")
    print(f"Date range: {df['post_date'].min()} to {df['post_date'].max()}")


if __name__ == "__main__":
    save_sample_data()


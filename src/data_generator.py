import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def generate_sample_data(num_records=200):
    """Generate realistic sample procurement data for India"""
    
    np.random.seed(42)
    
    vendors = [
        "Tata Consultancy Services", "Infosys Limited", "Wipro Limited", 
        "HCL Technologies", "Tech Mahindra", "Larsen & Toubro Infotech",
        "Bharat Electronics Limited", "ITC Limited", "Reliance Industries",
        "Adani Enterprises", "Small Scale Vendor Pvt Ltd", "Medium Enterprise Solutions",
        "National IT Services", "Digital India Corp", "Public Sector Undertaking IT"
    ]
    
    categories = [
        "IT Infrastructure", "Software Development", "Consulting Services",
        "Hardware Procurement", "Network Security", "Cloud Services",
        "Digital Transformation", "Maintenance & Support", "Training & Development"
    ]
    
    ministries = [
        "Ministry of Electronics & IT", "Ministry of Defence", "Ministry of Health",
        "Ministry of Education", "Ministry of Transport", "Ministry of Finance"
    ]
    
    locations = [
        "New Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata",
        "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow"
    ]
    
    data = []
    base_date = datetime.now() - timedelta(days=365)
    
    for i in range(num_records):
        base_value = np.random.lognormal(14, 1.0)
        contract_value = max(100000, base_value)
        
        if np.random.random() < 0.05:
            contract_value *= np.random.uniform(10, 50)
        
        post_date = base_date + timedelta(days=np.random.randint(0, 365))
        award_delay = timedelta(days=np.random.randint(7, 120))
        award_date = post_date + award_delay
        
        data.append({
            'contract_id': f'GEM/2024/{5000 + i}',
            'vendor_name': np.random.choice(vendors),
            'category': np.random.choice(categories),
            'contract_value': contract_value,
            'post_date': post_date.strftime('%Y-%m-%d'),
            'award_date': award_date.strftime('%Y-%m-%d'),
            'ministry': np.random.choice(ministries),
            'location': np.random.choice(locations),
            'processing_days': award_delay.days
        })
    
    return pd.DataFrame(data)

def save_sample_data():
    """Generate and save sample data"""
    df = generate_sample_data(250)
    
    df.to_csv('data/sample_procurement_data.csv', index=False)
    
    json_data = df.to_dict('records')
    with open('data/sample_procurement_data.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Generated sample data with {len(df)} records")
    print(f"Total contract value: INR {df['contract_value'].sum()/1e7:.2f} Crores")
    print(f"Average contract value: INR {df['contract_value'].mean()/1e5:.2f} Lakhs")
    print(f"Unique vendors: {df['vendor_name'].nunique()}")

if __name__ == "__main__":
    save_sample_data()

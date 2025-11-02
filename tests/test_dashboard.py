import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from utils.helpers import DataProcessor, APIClient

class TestDataProcessor:
    """Test cases for DataProcessor class"""
    
    def setup_method(self):
        self.processor = DataProcessor()
        self.sample_data = pd.DataFrame({
            'contract_id': ['C001', 'C002', 'C003'],
            'vendor_name': ['Vendor A', 'Vendor B', None],
            'contract_value': [1000000, 2000000, None],
            'post_date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'award_date': ['2024-01-15', '2024-01-20', '2024-01-25']
        })
    
    def test_clean_data(self):
        """Test data cleaning functionality"""
        cleaned_data = self.processor.clean_procurement_data(self.sample_data)
        
        assert len(cleaned_data) == 3
        assert cleaned_data['vendor_name'].isna().sum() == 0
        assert cleaned_data['contract_value'].isna().sum() == 0
    
    def test_calculate_metrics(self):
        """Test metrics calculation"""
        metrics = self.processor.calculate_metrics(self.sample_data)
        
        assert 'total_contracts' in metrics
        assert 'total_value' in metrics
        assert metrics['total_contracts'] == 3

def test_anomaly_detection():
    """Test anomaly detection logic"""
    from sklearn.ensemble import IsolationForest
    
    values = np.concatenate([np.random.normal(1000000, 200000, 99), [50000000]])
    X = values.reshape(-1, 1)
    
    clf = IsolationForest(contamination=0.1, random_state=42)
    predictions = clf.fit_predict(X)
    
    assert predictions[-1] == -1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

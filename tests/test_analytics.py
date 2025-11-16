"""
Comprehensive Testing Module for TransparAI
Unit tests, integration tests, and performance validation
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.analytics import (
    AnomalyDetection, StatisticalAnalysis, ClusteringAnalysis,
    VendorAnalysis, CollusionDetection, EfficiencyAnalysis
)
from src.financial_analysis import (
    FinancialValuation, ROIAnalysis, CostEfficiencyAnalysis, BudgetForecasting
)
from src.api_integration import DataValidator


class TestDataGeneration:
    """Generate test data"""
    
    @staticmethod
    def create_sample_dataframe(n_records: int = 100) -> pd.DataFrame:
        """Create sample procurement data"""
        np.random.seed(42)
        
        vendors = ["Vendor A", "Vendor B", "Vendor C", "Vendor D", "Vendor E"]
        categories = ["IT", "Infrastructure", "Consulting", "Hardware", "Services"]
        ministries = ["Defence", "Education", "Health", "Transport", "IT"]
        
        data = {
            'contract_id': [f'CON{1000+i}' for i in range(n_records)],
            'vendor_name': np.random.choice(vendors, n_records),
            'category': np.random.choice(categories, n_records),
            'contract_value': np.random.lognormal(14, 1.2, n_records),
            'post_date': [datetime.now() - timedelta(days=np.random.randint(0, 365))
                         for _ in range(n_records)],
            'award_date': [datetime.now() - timedelta(days=np.random.randint(0, 365))
                          for _ in range(n_records)],
            'ministry': np.random.choice(ministries, n_records),
            'location': np.random.choice(['Delhi', 'Mumbai', 'Bangalore'], n_records)
        }
        
        df = pd.DataFrame(data)
        df['processing_days'] = (df['award_date'] - df['post_date']).dt.days.abs()
        
        return df


class TestAnomalyDetection(unittest.TestCase):
    """Test anomaly detection functions"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        cls.df = TestDataGeneration.create_sample_dataframe(100)
    
    def test_zscore_detection(self):
        """Test Z-score anomaly detection"""
        anomalies = AnomalyDetection.zscore_detection(
            self.df['contract_value'],
            threshold=3.0
        )
        
        self.assertIsInstance(anomalies, np.ndarray)
        self.assertTrue(len(anomalies) > 0)
        self.assertLessEqual(anomalies.sum(), len(anomalies))
    
    def test_iqr_detection(self):
        """Test IQR anomaly detection"""
        anomalies = AnomalyDetection.iqr_detection(self.df['contract_value'])
        
        self.assertIsInstance(anomalies, (pd.Series, np.ndarray))
        self.assertTrue(len(anomalies) > 0)
    
    def test_isolation_forest_detection(self):
        """Test Isolation Forest anomaly detection"""
        features = ['contract_value', 'processing_days']
        anomalies = AnomalyDetection.isolation_forest_detection(
            self.df,
            features=features,
            contamination=0.1
        )
        
        self.assertIsInstance(anomalies, np.ndarray)
        self.assertEqual(len(anomalies), len(self.df))
        self.assertAlmostEqual(anomalies.sum() / len(anomalies), 0.1, places=1)


class TestStatisticalAnalysis(unittest.TestCase):
    """Test statistical analysis functions"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        cls.df = TestDataGeneration.create_sample_dataframe(100)
    
    def test_normality_test(self):
        """Test Shapiro-Wilk normality test"""
        result = StatisticalAnalysis.normality_test(self.df['contract_value'])
        
        self.assertIn('statistic', result)
        self.assertIn('p_value', result)
        self.assertIn('is_normal', result)
    
    def test_ttest_analysis(self):
        """Test t-test analysis"""
        group1 = self.df[self.df['vendor_name'] == 'Vendor A']['contract_value']
        group2 = self.df[self.df['vendor_name'] == 'Vendor B']['contract_value']
        
        if len(group1) > 0 and len(group2) > 0:
            result = StatisticalAnalysis.ttest_analysis(group1, group2)
            
            self.assertIn('t_statistic', result)
            self.assertIn('p_value', result)
            self.assertIn('significant', result)
    
    def test_confidence_interval(self):
        """Test confidence interval calculation"""
        ci = StatisticalAnalysis.confidence_interval(self.df['contract_value'])
        
        self.assertEqual(len(ci), 2)
        self.assertLess(ci[0], ci[1])
    
    def test_correlation_matrix(self):
        """Test correlation matrix calculation"""
        corr = StatisticalAnalysis.correlation_matrix(
            self.df[['contract_value', 'processing_days']]
        )
        
        self.assertIsInstance(corr, pd.DataFrame)
        self.assertEqual(corr.shape, (2, 2))
    
    def test_forecast_accuracy(self):
        """Test forecast accuracy metrics"""
        actual = np.array([100, 200, 300, 400, 500])
        predicted = np.array([110, 195, 305, 390, 510])
        
        result = StatisticalAnalysis.forecast_accuracy(actual, predicted)
        
        self.assertIn('mae', result)
        self.assertIn('rmse', result)
        self.assertIn('mape', result)
        self.assertIn('r_squared', result)


class TestVendorAnalysis(unittest.TestCase):
    """Test vendor analysis functions"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        cls.df = TestDataGeneration.create_sample_dataframe(100)
    
    def test_herfindahl_index(self):
        """Test HHI calculation"""
        hhi = VendorAnalysis.herfindahl_index(self.df)
        
        self.assertGreaterEqual(hhi, 0)
        self.assertLessEqual(hhi, 10000)
    
    def test_vendor_concentration_ratio(self):
        """Test concentration ratio calculation"""
        cr = VendorAnalysis.vendor_concentration_ratio(self.df, top_n=3)
        
        self.assertGreaterEqual(cr, 0)
        self.assertLessEqual(cr, 100)
    
    def test_vendor_diversity_index(self):
        """Test diversity index calculation"""
        diversity = VendorAnalysis.vendor_diversity_index(self.df)
        
        self.assertGreaterEqual(diversity, 0)
        self.assertLessEqual(diversity, 1)
    
    def test_vendor_repeat_analysis(self):
        """Test vendor repeat analysis"""
        result = VendorAnalysis.vendor_repeat_analysis(self.df)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('contract_count', result.columns)
        self.assertIn('total_value', result.columns)


class TestCollusionDetection(unittest.TestCase):
    """Test collusion detection functions"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        cls.df = TestDataGeneration.create_sample_dataframe(100)
    
    def test_bid_clustering(self):
        """Test bid clustering detection"""
        result = CollusionDetection.detect_bid_clustering(self.df)
        
        self.assertIn('similar_bids_count', result)
        self.assertIn('similar_bids', result)
        self.assertIn('suspicious', result)
    
    def test_price_patterns(self):
        """Test price pattern detection"""
        result = CollusionDetection.detect_price_patterns(self.df)
        
        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)


class TestEfficiencyAnalysis(unittest.TestCase):
    """Test efficiency analysis functions"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        cls.df = TestDataGeneration.create_sample_dataframe(100)
    
    def test_efficiency_score(self):
        """Test efficiency scoring"""
        scores = EfficiencyAnalysis.efficiency_score(self.df)
        
        self.assertEqual(len(scores), len(self.df))
        self.assertTrue((scores >= 0).all())
        self.assertTrue((scores <= 1).all())
    
    def test_competitive_bidding(self):
        """Test competitive bidding analysis"""
        result = EfficiencyAnalysis.competitive_bidding_analysis(self.df)
        
        self.assertIsInstance(result, dict)


class TestFinancialValuation(unittest.TestCase):
    """Test financial valuation functions"""
    
    def test_npv_calculation(self):
        """Test NPV calculation"""
        cash_flows = [100, 150, 200, 250]
        npv = FinancialValuation.calculate_npv(cash_flows, 0.1, 500)
        
        self.assertIsInstance(npv, (int, float))
    
    def test_irr_calculation(self):
        """Test IRR calculation"""
        cash_flows = [100, 150, 200, 250]
        irr = FinancialValuation.calculate_irr(cash_flows, 500)
        
        self.assertIsInstance(irr, (int, float))
        self.assertGreater(irr, -1)
    
    def test_payback_period(self):
        """Test payback period calculation"""
        cash_flows = [100, 150, 200, 250]
        payback = FinancialValuation.calculate_payback_period(cash_flows, 300)
        
        self.assertIsInstance(payback, (int, float))
        self.assertGreaterEqual(payback, 0)
    
    def test_profitability_index(self):
        """Test profitability index calculation"""
        cash_flows = [100, 150, 200, 250]
        pi = FinancialValuation.calculate_profitability_index(cash_flows, 0.1, 500)
        
        self.assertIsInstance(pi, (int, float))
        self.assertGreaterEqual(pi, 0)


class TestROIAnalysis(unittest.TestCase):
    """Test ROI analysis functions"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        cls.df = TestDataGeneration.create_sample_dataframe(100)
    
    def test_roi_calculation(self):
        """Test ROI calculation"""
        roi = ROIAnalysis.calculate_roi(1000, 5000)
        
        self.assertAlmostEqual(roi, 20)
    
    def test_annualized_roi(self):
        """Test annualized ROI calculation"""
        annualized = ROIAnalysis.calculate_annualized_roi(50, 2)
        
        self.assertIsInstance(annualized, (int, float))
        self.assertGreater(annualized, 0)
    
    def test_roi_by_vendor(self):
        """Test ROI calculation by vendor"""
        result = ROIAnalysis.calculate_roi_by_vendor(self.df)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('vendor', result.columns)


class TestCostEfficiency(unittest.TestCase):
    """Test cost efficiency analysis"""
    
    def test_cost_variance(self):
        """Test cost variance calculation"""
        result = CostEfficiencyAnalysis.calculate_cost_variance(950, 1000)
        
        self.assertEqual(result['variance'], 50)
        self.assertAlmostEqual(result['variance_pct'], 5)
    
    def test_cost_benefit_analysis(self):
        """Test cost-benefit analysis"""
        benefits = [1000, 1100, 1200]
        costs = [500, 400, 300]
        
        result = CostEfficiencyAnalysis.cost_benefit_analysis(benefits, costs)
        
        self.assertIn('benefit_cost_ratio', result)
        self.assertIn('feasible', result)


class TestDataValidator(unittest.TestCase):
    """Test data validation functions"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data"""
        cls.df = TestDataGeneration.create_sample_dataframe(100)
    
    def test_validate_contract_data(self):
        """Test contract data validation"""
        is_valid, errors = DataValidator.validate_contract_data(self.df)
        
        self.assertTrue(is_valid or len(errors) > 0)
    
    def test_clean_contract_data(self):
        """Test data cleaning"""
        df_clean = DataValidator.clean_contract_data(self.df)
        
        self.assertLessEqual(len(df_clean), len(self.df))
        self.assertFalse(df_clean.duplicated().any())
    
    def test_data_quality_report(self):
        """Test data quality report generation"""
        report = DataValidator.get_data_quality_report(self.df)
        
        self.assertIn('total_records', report)
        self.assertIn('unique_vendors', report)
        self.assertIn('missing_values', report)


def run_all_tests(verbosity: int = 2) -> unittest.TestResult:
    """
    Run all tests
    
    Args:
        verbosity: Verbosity level
        
    Returns:
        Test result
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    test_classes = [
        TestAnomalyDetection,
        TestStatisticalAnalysis,
        TestVendorAnalysis,
        TestCollusionDetection,
        TestEfficiencyAnalysis,
        TestFinancialValuation,
        TestROIAnalysis,
        TestCostEfficiency,
        TestDataValidator
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


if __name__ == '__main__':
    run_all_tests()

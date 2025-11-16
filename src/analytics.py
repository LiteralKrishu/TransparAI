"""
Advanced Analytics Functions for TransparAI
Includes anomaly detection, clustering, and statistical analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')


class AnomalyDetection:
    """Z-score and IQR-based anomaly detection"""
    
    @staticmethod
    def zscore_detection(data: pd.Series, threshold: float = 3.0) -> np.ndarray:
        """
        Detect anomalies using Z-score method
        
        Args:
            data: pandas Series of values
            threshold: z-score threshold (default 3.0)
            
        Returns:
            Boolean array where True indicates anomaly
        """
        z_scores = np.abs(stats.zscore(data.dropna()))
        return z_scores > threshold
    
    @staticmethod
    def iqr_detection(data: pd.Series) -> np.ndarray:
        """
        Detect anomalies using Interquartile Range (IQR) method
        
        Args:
            data: pandas Series of values
            
        Returns:
            Boolean array where True indicates anomaly
        """
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return (data < lower_bound) | (data > upper_bound)
    
    @staticmethod
    def isolation_forest_detection(df: pd.DataFrame, features: List[str], 
                                   contamination: float = 0.1) -> np.ndarray:
        """
        Detect anomalies using Isolation Forest
        
        Args:
            df: DataFrame containing data
            features: List of feature columns to use
            contamination: Expected contamination rate
            
        Returns:
            Boolean array where True indicates anomaly
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features].fillna(df[features].mean()))
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(X_scaled)
        
        return predictions == -1


class StatisticalAnalysis:
    """Statistical validation and analysis functions"""
    
    @staticmethod
    def normality_test(data: pd.Series) -> Dict:
        """
        Perform Shapiro-Wilk normality test
        
        Args:
            data: pandas Series
            
        Returns:
            Dictionary with test statistics
        """
        stat, p_value = stats.shapiro(data.dropna())
        
        return {
            'statistic': stat,
            'p_value': p_value,
            'is_normal': p_value > 0.05,
            'interpretation': 'Normal distribution' if p_value > 0.05 else 'Not normal'
        }
    
    @staticmethod
    def ttest_analysis(group1: pd.Series, group2: pd.Series, 
                      equal_var: bool = False) -> Dict:
        """
        Perform independent t-test between two groups
        
        Args:
            group1: First group data
            group2: Second group data
            equal_var: Whether to assume equal variances
            
        Returns:
            Dictionary with test results
        """
        t_stat, p_value = stats.ttest_ind(group1.dropna(), group2.dropna(), 
                                         equal_var=equal_var)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': abs(t_stat) / np.sqrt(len(group1) + len(group2))
        }
    
    @staticmethod
    def confidence_interval(data: pd.Series, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for mean
        
        Args:
            data: pandas Series
            confidence: Confidence level (default 0.95 for 95%)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        n = len(data.dropna())
        mean = data.mean()
        se = stats.sem(data.dropna())
        margin = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        
        return (mean - margin, mean + margin)
    
    @staticmethod
    def correlation_matrix(df: pd.DataFrame, numeric_only: bool = True) -> pd.DataFrame:
        """
        Calculate correlation matrix
        
        Args:
            df: DataFrame
            numeric_only: Include only numeric columns
            
        Returns:
            Correlation matrix
        """
        return df.corr(numeric_only=numeric_only)
    
    @staticmethod
    def forecast_accuracy(actual: np.ndarray, predicted: np.ndarray) -> Dict:
        """
        Calculate forecast accuracy metrics
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary with accuracy metrics
        """
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r_squared': 1 - (np.sum((actual - predicted) ** 2) / 
                            np.sum((actual - np.mean(actual)) ** 2))
        }


class ClusteringAnalysis:
    """DBSCAN clustering for pattern detection"""
    
    @staticmethod
    def dbscan_clustering(df: pd.DataFrame, features: List[str], 
                         eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
        """
        Perform DBSCAN clustering
        
        Args:
            df: DataFrame containing data
            features: List of feature columns
            eps: Maximum distance between samples
            min_samples: Minimum samples in a neighborhood
            
        Returns:
            Array of cluster labels
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features].fillna(df[features].mean()))
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_scaled)
        
        return clusters


class VendorAnalysis:
    """Vendor concentration and behavior analysis"""
    
    @staticmethod
    def herfindahl_index(df: pd.DataFrame, value_column: str, 
                        vendor_column: str = 'vendor_name') -> float:
        """
        Calculate Herfindahl-Hirschman Index (HHI) for market concentration
        
        Args:
            df: DataFrame
            value_column: Column with values to sum
            vendor_column: Column with vendor names
            
        Returns:
            HHI value (0-10000 scale)
        """
        vendor_values = df.groupby(vendor_column)[value_column].sum()
        total_value = vendor_values.sum()
        
        market_shares = (vendor_values / total_value) * 100
        hhi = np.sum(market_shares ** 2)
        
        return hhi
    
    @staticmethod
    def vendor_concentration_ratio(df: pd.DataFrame, value_column: str, 
                                  top_n: int = 4, 
                                  vendor_column: str = 'vendor_name') -> float:
        """
        Calculate concentration ratio (CR-n)
        
        Args:
            df: DataFrame
            value_column: Column with values
            top_n: Number of top vendors
            vendor_column: Column with vendor names
            
        Returns:
            Concentration ratio (0-100 scale)
        """
        vendor_values = df.groupby(vendor_column)[value_column].sum()
        total_value = vendor_values.sum()
        
        top_vendors_value = vendor_values.nlargest(top_n).sum()
        cr = (top_vendors_value / total_value) * 100
        
        return cr
    
    @staticmethod
    def vendor_diversity_index(df: pd.DataFrame, value_column: str, 
                              vendor_column: str = 'vendor_name') -> float:
        """
        Calculate vendor diversity using Shannon entropy
        
        Args:
            df: DataFrame
            value_column: Column with values
            vendor_column: Column with vendor names
            
        Returns:
            Diversity index (0-1 scale, higher = more diverse)
        """
        vendor_values = df.groupby(vendor_column)[value_column].sum()
        total_value = vendor_values.sum()
        
        market_shares = vendor_values / total_value
        entropy = -np.sum(market_shares * np.log(market_shares + 1e-10))
        
        max_entropy = np.log(len(market_shares))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy
    
    @staticmethod
    def vendor_repeat_analysis(df: pd.DataFrame, 
                              vendor_column: str = 'vendor_name') -> pd.DataFrame:
        """
        Analyze vendor repeat contract patterns
        
        Args:
            df: DataFrame
            vendor_column: Column with vendor names
            
        Returns:
            DataFrame with vendor contract statistics
        """
        vendor_stats = df.groupby(vendor_column).agg({
            'contract_id': 'count',
            'contract_value': ['sum', 'mean', 'std'],
            'processing_days': 'mean'
        }).round(2)
        
        vendor_stats.columns = ['contract_count', 'total_value', 'avg_value', 
                               'value_std', 'avg_processing_days']
        vendor_stats = vendor_stats.sort_values('total_value', ascending=False)
        
        return vendor_stats


class CollusionDetection:
    """Pattern detection for potential collusion"""
    
    @staticmethod
    def detect_bid_clustering(df: pd.DataFrame, value_column: str,
                             threshold: float = 0.05) -> Dict:
        """
        Detect suspicious bid clustering
        
        Args:
            df: DataFrame
            value_column: Column with contract values
            threshold: Threshold for similarity (default 5%)
            
        Returns:
            Dictionary with clustering analysis results
        """
        values = df[value_column].values
        
        similar_bids = []
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                diff_ratio = abs(values[i] - values[j]) / max(values[i], values[j])
                if diff_ratio < threshold:
                    similar_bids.append({
                        'bid_1': values[i],
                        'bid_2': values[j],
                        'difference_ratio': diff_ratio
                    })
        
        return {
            'similar_bids_count': len(similar_bids),
            'similar_bids': similar_bids,
            'suspicious': len(similar_bids) > len(values) * 0.1
        }
    
    @staticmethod
    def detect_price_patterns(df: pd.DataFrame, value_column: str,
                             vendor_column: str = 'vendor_name') -> Dict:
        """
        Detect suspicious price patterns by vendor
        
        Args:
            df: DataFrame
            value_column: Column with contract values
            vendor_column: Column with vendor names
            
        Returns:
            Dictionary with pattern analysis
        """
        suspicious_vendors = {}
        
        for vendor in df[vendor_column].unique():
            vendor_df = df[df[vendor_column] == vendor]
            values = vendor_df[value_column].values
            
            if len(values) > 1:
                cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
                
                suspicious_vendors[vendor] = {
                    'contract_count': len(values),
                    'coefficient_of_variation': cv,
                    'suspicious': cv < 0.1
                }
        
        return suspicious_vendors


class EfficiencyAnalysis:
    """Procurement efficiency scoring and analysis"""
    
    @staticmethod
    def efficiency_score(df: pd.DataFrame) -> pd.Series:
        """
        Calculate overall efficiency score
        
        Args:
            df: DataFrame with procurement data
            
        Returns:
            Series with efficiency scores
        """
        scores = pd.Series(index=df.index, dtype=float)
        
        # Processing time efficiency (lower is better)
        processing_score = 1 - (df['processing_days'] / df['processing_days'].max())
        
        # Value efficiency (consistency is good)
        for vendor in df['vendor_name'].unique():
            vendor_mask = df['vendor_name'] == vendor
            vendor_values = df.loc[vendor_mask, 'contract_value']
            
            if len(vendor_values) > 1:
                cv = vendor_values.std() / vendor_values.mean()
                consistency_score = 1 - min(cv, 1.0)
            else:
                consistency_score = 0.5
            
            scores[vendor_mask] = (processing_score[vendor_mask] * 0.6 + 
                                 consistency_score * 0.4)
        
        return scores
    
    @staticmethod
    def competitive_bidding_analysis(df: pd.DataFrame) -> Dict:
        """
        Analyze competitive bidding health
        
        Args:
            df: DataFrame
            
        Returns:
            Dictionary with bidding analysis
        """
        categories = df.groupby('category').apply(
            lambda x: {
                'avg_processing_days': x['processing_days'].mean(),
                'contract_count': len(x),
                'unique_vendors': x['vendor_name'].nunique(),
                'avg_value': x['contract_value'].mean(),
                'competition_ratio': x['vendor_name'].nunique() / max(len(x), 1)
            }
        )
        
        return categories.to_dict()

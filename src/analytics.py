"""
Advanced Analytics Functions for TransparAI
Includes anomaly detection, clustering, and statistical analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from typing import Tuple, Dict, List


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
        values = pd.to_numeric(data, errors='coerce').to_numpy(dtype=float)
        mask = np.isfinite(values)
        flags = np.zeros(len(values), dtype=bool)
        if mask.sum() >= 2 and np.std(values[mask]) > 0:
            flags[mask] = np.abs(stats.zscore(values[mask])) > threshold
        return flags

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
                                   contamination: float = 0.1, random_state: int = 42) -> np.ndarray:
        """
        Detect anomalies using Isolation Forest

        Args:
            df: DataFrame containing data
            features: List of feature columns to use
            contamination: Expected contamination rate

        Returns:
            Boolean array where True indicates anomaly
        """
        numeric = df[features].apply(pd.to_numeric, errors='coerce')
        if numeric.empty or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError('Analysis needs finite values in every selected feature')
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric)

        if len(df) < 3:
            return np.zeros(len(df), dtype=bool)
        iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
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
        numeric = df[features].apply(pd.to_numeric, errors='coerce')
        if numeric.empty or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError('Analysis needs finite values in every selected feature')
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_scaled)

        return clusters


class VendorAnalysis:
    """Vendor concentration and behavior analysis"""

    @staticmethod
    def herfindahl_index(df: pd.DataFrame, value_column: str = 'contract_value',
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
        # Basic validations
        if df is None:
            raise ValueError("DataFrame `df` is None")

        # If the expected value column is missing, try to infer a numeric value column
        if value_column not in df.columns:
            fallback_candidates = [c for c in df.columns if 'value' in c.lower() or 'amount' in c.lower()]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            chosen = None
            for cand in ['contract_value'] + fallback_candidates + numeric_cols:
                if cand in df.columns:
                    chosen = cand
                    break

            if chosen is None:
                raise KeyError(f"Value column '{value_column}' not found and no numeric fallback available")
            value_column = chosen

        vendor_values = df.groupby(vendor_column)[value_column].sum()
        total_value = vendor_values.sum()

        if total_value == 0 or np.isnan(total_value):
            return 0.0

        market_shares = (vendor_values / total_value) * 100
        hhi = float(np.sum(market_shares ** 2))

        return hhi

    @staticmethod
    def vendor_concentration_ratio(df: pd.DataFrame, value_column: str = 'contract_value',
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
        if df is None:
            raise ValueError("DataFrame `df` is None")

        if value_column not in df.columns:
            fallback_candidates = [c for c in df.columns if 'value' in c.lower() or 'amount' in c.lower()]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            chosen = None
            for cand in ['contract_value'] + fallback_candidates + numeric_cols:
                if cand in df.columns:
                    chosen = cand
                    break

            if chosen is None:
                raise KeyError(f"Value column '{value_column}' not found and no numeric fallback available")
            value_column = chosen

        vendor_values = df.groupby(vendor_column)[value_column].sum()
        total_value = vendor_values.sum()

        if total_value == 0 or np.isnan(total_value):
            return 0.0

        top_vendors_value = vendor_values.nlargest(top_n).sum()
        cr = float((top_vendors_value / total_value) * 100)

        return cr

    @staticmethod
    def vendor_diversity_index(df: pd.DataFrame, value_column: str = 'contract_value',
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
        if df is None:
            raise ValueError("DataFrame `df` is None")

        if value_column not in df.columns:
            fallback_candidates = [c for c in df.columns if 'value' in c.lower() or 'amount' in c.lower()]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            chosen = None
            for cand in ['contract_value'] + fallback_candidates + numeric_cols:
                if cand in df.columns:
                    chosen = cand
                    break

            if chosen is None:
                raise KeyError(f"Value column '{value_column}' not found and no numeric fallback available")
            value_column = chosen

        vendor_values = df.groupby(vendor_column)[value_column].sum()
        total_value = vendor_values.sum()

        if total_value == 0 or np.isnan(total_value):
            return 0.0

        market_shares = vendor_values / total_value
        entropy = -np.sum(market_shares * np.log(market_shares + 1e-10))

        max_entropy = np.log(len(market_shares)) if len(market_shares) > 0 else 1.0
        normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

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
    def detect_bid_clustering(df: pd.DataFrame, value_column: str = 'contract_value',
                             threshold: float = 0.05, max_pairs: int = 100) -> Dict:
        """Count similar bid prices only within a tender and identical item.

        Contract awards alone cannot establish competing bids. At most max_pairs
        example pairs are retained; sorted-window counting avoids pairwise scans.
        """
        result = {'similar_bids_count': 0, 'similar_bids': [], 'suspicious': False,
                  'available': False, 'reason': 'Needs tender_id, item_id, bid_price, vendor_name, quantity and price_basis',
                  'truncated': False}
        required = ['tender_id', 'item_id', 'bid_price', 'vendor_name', 'quantity', 'price_basis']
        if not set(required).issubset(df.columns):
            return result
        if not 0 < threshold < 1 or max_pairs < 0:
            raise ValueError('Invalid bid comparison limits')
        bids = df.dropna(subset=required).copy()
        bids['bid_price'] = pd.to_numeric(bids.bid_price, errors='coerce')
        bids['quantity'] = pd.to_numeric(bids.quantity, errors='coerce')
        bids = bids[np.isfinite(bids.bid_price) & (bids.bid_price > 0) & np.isfinite(bids.quantity) & (bids.quantity > 0)]
        # One price per vendor per tender/item is required for a comparable set.
        keys = ['tender_id', 'item_id', 'quantity', 'price_basis', 'vendor_name']
        if bids.duplicated(keys).any():
            result['reason'] = 'Multiple bids per vendor and item; resolve revisions before comparison'
            return result
        result.update(available=True, reason='Similar prices are review signals, not evidence of collusion')
        for _, group in bids.groupby(['tender_id', 'item_id', 'quantity', 'price_basis'], sort=True):
            group = group.sort_values('bid_price')
            values = group.bid_price.to_numpy()
            for j, value in enumerate(values):
                left = int(np.searchsorted(values, value * (1 - threshold), side='right'))
                count = max(0, j - left)
                result['similar_bids_count'] += count
                room = max_pairs - len(result['similar_bids'])
                for i in range(left, min(j, left + max(room, 0))):
                    result['similar_bids'].append({'bid_1': float(values[i]), 'bid_2': float(value),
                                                  'difference_ratio': float((value-values[i])/value)})
        result['truncated'] = result['similar_bids_count'] > len(result['similar_bids'])
        return result

    @staticmethod
    def detect_price_patterns(df: pd.DataFrame, value_column: str = 'contract_value',
                             vendor_column: str = 'vendor_name') -> Dict:
        """Describe variation within vendor/category; do not infer collusion."""
        patterns = {}
        if 'category' not in df:
            return patterns
        for (vendor, category), group in df.groupby([vendor_column, 'category']):
            values = pd.to_numeric(group[value_column], errors='coerce')
            values = values[np.isfinite(values) & (values > 0)]
            if len(values) > 1:
                cv = float(values.std(ddof=0) / values.mean())
                patterns[f'{vendor} / {category}'] = {'contract_count': len(values),
                    'coefficient_of_variation': cv, 'low_variation': cv < 0.1,
                    'suspicious': False}
        return patterns


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
        days = pd.to_numeric(df['processing_days'], errors='coerce')
        days = days.where(np.isfinite(days) & (days >= 0))
        max_days = days.max()
        processing_score = (1 - days / max_days) if max_days > 0 else days.where(days.isna(), 1.0)

        # Value efficiency (consistency is good)
        for vendor in df['vendor_name'].unique():
            vendor_mask = df['vendor_name'] == vendor
            vendor_values = df.loc[vendor_mask, 'contract_value']

            if len(vendor_values) > 1:
                cv = vendor_values.std() / vendor_values.mean() if vendor_values.mean() > 0 else 0
                consistency_score = 1 - min(cv, 1.0)
            else:
                consistency_score = 0.5

            scores[vendor_mask] = (processing_score[vendor_mask] * 0.6 +
                                 consistency_score * 0.4)

        return scores.clip(0, 1)

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

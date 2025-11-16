"""
Financial Valuation and Analysis Module for TransparAI
NPV, ROI, Payback Period, and Cost Efficiency Calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class FinancialValuation:
    """NPV and investment analysis calculations"""
    
    @staticmethod
    def calculate_npv(cash_flows: List[float], discount_rate: float,
                     initial_investment: float = 0) -> float:
        """
        Calculate Net Present Value (NPV)
        
        Args:
            cash_flows: List of future cash flows
            discount_rate: Discount rate (e.g., 0.1 for 10%)
            initial_investment: Initial investment amount
            
        Returns:
            NPV value
        """
        npv = -initial_investment
        for t, cf in enumerate(cash_flows, start=1):
            npv += cf / ((1 + discount_rate) ** t)
        
        return npv
    
    @staticmethod
    def calculate_irr(cash_flows: List[float], initial_investment: float,
                     iterations: int = 100) -> float:
        """
        Calculate Internal Rate of Return (IRR) using Newton-Raphson method
        
        Args:
            cash_flows: List of future cash flows
            initial_investment: Initial investment amount
            iterations: Number of iterations for convergence
            
        Returns:
            IRR value
        """
        flows = [-initial_investment] + cash_flows
        
        # Initial guess
        rate = 0.1
        
        for _ in range(iterations):
            npv = sum(cf / (1 + rate) ** t for t, cf in enumerate(flows))
            npv_derivative = sum(-t * cf / (1 + rate) ** (t + 1) 
                                for t, cf in enumerate(flows))
            
            if abs(npv_derivative) < 1e-6:
                break
                
            rate = rate - npv / npv_derivative
        
        return rate
    
    @staticmethod
    def calculate_payback_period(cash_flows: List[float],
                                initial_investment: float) -> float:
        """
        Calculate Payback Period
        
        Args:
            cash_flows: List of cash flows
            initial_investment: Initial investment
            
        Returns:
            Payback period in periods
        """
        cumulative = 0
        
        for period, cf in enumerate(cash_flows, start=1):
            cumulative += cf
            
            if cumulative >= initial_investment:
                # Linear interpolation for fractional period
                excess = cumulative - initial_investment
                return period - (excess / cf)
        
        return float('inf')
    
    @staticmethod
    def calculate_profitability_index(cash_flows: List[float],
                                     discount_rate: float,
                                     initial_investment: float) -> float:
        """
        Calculate Profitability Index (PI)
        
        Args:
            cash_flows: List of future cash flows
            discount_rate: Discount rate
            initial_investment: Initial investment
            
        Returns:
            Profitability Index
        """
        pv_inflows = sum(cf / (1 + discount_rate) ** (t + 1) 
                        for t, cf in enumerate(cash_flows))
        
        return pv_inflows / initial_investment if initial_investment > 0 else 0


class ROIAnalysis:
    """Return on Investment Analysis"""
    
    @staticmethod
    def calculate_roi(profit: float, investment: float) -> float:
        """
        Calculate simple ROI
        
        Args:
            profit: Profit amount
            investment: Investment amount
            
        Returns:
            ROI percentage
        """
        return (profit / investment * 100) if investment != 0 else 0
    
    @staticmethod
    def calculate_annualized_roi(total_roi: float, years: float) -> float:
        """
        Calculate annualized ROI
        
        Args:
            total_roi: Total ROI percentage
            years: Number of years
            
        Returns:
            Annualized ROI percentage
        """
        if years == 0:
            return 0
        
        return (((1 + total_roi / 100) ** (1 / years)) - 1) * 100
    
    @staticmethod
    def calculate_roi_by_vendor(df: pd.DataFrame, cost_column: str = 'contract_value',
                               vendor_column: str = 'vendor_name') -> pd.DataFrame:
        """
        Calculate ROI metrics by vendor
        
        Args:
            df: DataFrame with procurement data
            cost_column: Column with costs
            vendor_column: Column with vendor names
            
        Returns:
            DataFrame with ROI by vendor
        """
        vendor_roi = pd.DataFrame()
        
        for vendor in df[vendor_column].unique():
            vendor_df = df[df[vendor_column] == vendor]
            
            total_cost = vendor_df[cost_column].sum()
            contract_count = len(vendor_df)
            avg_cost = vendor_df[cost_column].mean()
            cost_variance = vendor_df[cost_column].var()
            
            vendor_roi = pd.concat([vendor_roi, pd.DataFrame({
                'vendor': [vendor],
                'total_cost': [total_cost],
                'contract_count': [contract_count],
                'avg_cost': [avg_cost],
                'cost_variance': [cost_variance],
                'cost_efficiency': [1 - (cost_variance / avg_cost ** 2) if avg_cost > 0 else 0]
            })], ignore_index=True)
        
        return vendor_roi.sort_values('total_cost', ascending=False)
    
    @staticmethod
    def calculate_cost_per_unit(df: pd.DataFrame, cost_column: str = 'contract_value',
                               quantity_column: str = 'contract_count') -> pd.Series:
        """
        Calculate cost per unit (cost efficiency)
        
        Args:
            df: DataFrame
            cost_column: Column with costs
            quantity_column: Column with quantities
            
        Returns:
            Series with cost per unit
        """
        return df[cost_column] / df[quantity_column].replace(0, 1)


class CostEfficiencyAnalysis:
    """Cost efficiency and optimization analysis"""
    
    @staticmethod
    def calculate_cost_variance(actual_cost: float, budgeted_cost: float) -> Dict:
        """
        Calculate cost variance and percentage
        
        Args:
            actual_cost: Actual cost incurred
            budgeted_cost: Budgeted cost
            
        Returns:
            Dictionary with variance metrics
        """
        variance = budgeted_cost - actual_cost
        variance_pct = (variance / budgeted_cost * 100) if budgeted_cost > 0 else 0
        
        return {
            'variance': variance,
            'variance_pct': variance_pct,
            'status': 'Under Budget' if variance > 0 else 'Over Budget',
            'efficiency': min(100, (budgeted_cost / actual_cost * 100)) if actual_cost > 0 else 100
        }
    
    @staticmethod
    def cost_benefit_analysis(benefits: List[float], costs: List[float],
                             discount_rate: float = 0.1) -> Dict:
        """
        Perform cost-benefit analysis
        
        Args:
            benefits: List of benefit values over time
            costs: List of cost values over time
            discount_rate: Discount rate
            
        Returns:
            Dictionary with CBA metrics
        """
        pv_benefits = sum(b / (1 + discount_rate) ** (t + 1) 
                         for t, b in enumerate(benefits))
        pv_costs = sum(c / (1 + discount_rate) ** (t + 1) 
                      for t, c in enumerate(costs))
        
        bcr = pv_benefits / pv_costs if pv_costs > 0 else 0
        net_benefit = pv_benefits - pv_costs
        
        return {
            'pv_benefits': pv_benefits,
            'pv_costs': pv_costs,
            'benefit_cost_ratio': bcr,
            'net_benefit': net_benefit,
            'feasible': bcr > 1
        }
    
    @staticmethod
    def identify_cost_outliers(df: pd.DataFrame, cost_column: str = 'contract_value',
                              threshold_std: float = 2.0) -> pd.DataFrame:
        """
        Identify cost outliers
        
        Args:
            df: DataFrame
            cost_column: Column with costs
            threshold_std: Number of standard deviations for threshold
            
        Returns:
            DataFrame with outlier records
        """
        mean_cost = df[cost_column].mean()
        std_cost = df[cost_column].std()
        
        lower_bound = mean_cost - threshold_std * std_cost
        upper_bound = mean_cost + threshold_std * std_cost
        
        outliers = df[(df[cost_column] < lower_bound) | 
                     (df[cost_column] > upper_bound)]
        
        return outliers.copy()
    
    @staticmethod
    def category_cost_analysis(df: pd.DataFrame, cost_column: str = 'contract_value',
                              category_column: str = 'category') -> pd.DataFrame:
        """
        Analyze costs by category
        
        Args:
            df: DataFrame
            cost_column: Column with costs
            category_column: Column with categories
            
        Returns:
            DataFrame with category cost analysis
        """
        category_analysis = df.groupby(category_column).agg({
            cost_column: ['sum', 'mean', 'std', 'min', 'max', 'count'],
            'vendor_name': 'nunique'
        }).round(2)
        
        category_analysis.columns = ['total_cost', 'avg_cost', 'cost_std',
                                    'min_cost', 'max_cost', 'contract_count',
                                    'unique_vendors']
        
        category_analysis['cost_per_vendor'] = (
            category_analysis['total_cost'] / 
            category_analysis['unique_vendors'].replace(0, 1)
        ).round(2)
        
        return category_analysis.sort_values('total_cost', ascending=False)
    
    @staticmethod
    def procurement_cycle_cost(df: pd.DataFrame,
                              processing_days_column: str = 'processing_days',
                              cost_column: str = 'contract_value') -> Dict:
        """
        Analyze costs related to procurement cycle duration
        
        Args:
            df: DataFrame
            processing_days_column: Column with processing days
            cost_column: Column with costs
            
        Returns:
            Dictionary with cycle cost metrics
        """
        avg_processing = df[processing_days_column].mean()
        avg_cost = df[cost_column].mean()
        
        # Cost per day (operational cost)
        cost_per_day = avg_cost / avg_processing if avg_processing > 0 else 0
        
        # Fast-tracked contracts (< median processing time)
        fast_tracked = df[df[processing_days_column] < df[processing_days_column].median()]
        slow_tracked = df[df[processing_days_column] >= df[processing_days_column].median()]
        
        return {
            'avg_processing_days': avg_processing,
            'avg_contract_value': avg_cost,
            'cost_per_day': cost_per_day,
            'fast_tracked_avg_cost': fast_tracked[cost_column].mean() if len(fast_tracked) > 0 else 0,
            'slow_tracked_avg_cost': slow_tracked[cost_column].mean() if len(slow_tracked) > 0 else 0,
            'time_cost_correlation': df[processing_days_column].corr(df[cost_column])
        }


class BudgetForecasting:
    """Budget forecasting and projections"""
    
    @staticmethod
    def simple_moving_average(data: pd.Series, window: int = 3) -> pd.Series:
        """
        Calculate simple moving average
        
        Args:
            data: Time series data
            window: Window size
            
        Returns:
            Series with moving averages
        """
        return data.rolling(window=window, min_periods=1).mean()
    
    @staticmethod
    def exponential_smoothing(data: pd.Series, alpha: float = 0.3) -> pd.Series:
        """
        Calculate exponential smoothing forecast
        
        Args:
            data: Time series data
            alpha: Smoothing factor (0-1)
            
        Returns:
            Series with smoothed values
        """
        smoothed = [data.iloc[0]]
        
        for i in range(1, len(data)):
            smoothed.append(alpha * data.iloc[i] + (1 - alpha) * smoothed[-1])
        
        return pd.Series(smoothed, index=data.index)
    
    @staticmethod
    def linear_trend_forecast(df: pd.DataFrame, value_column: str,
                             time_column: str = 'award_date',
                             forecast_periods: int = 12) -> Dict:
        """
        Forecast using linear trend
        
        Args:
            df: DataFrame
            value_column: Column with values to forecast
            time_column: Column with time data
            forecast_periods: Number of periods to forecast
            
        Returns:
            Dictionary with forecast results
        """
        # Prepare data
        df_sorted = df.sort_values(time_column)
        df_sorted['time_index'] = np.arange(len(df_sorted))
        
        x = df_sorted['time_index'].values
        y = df_sorted[value_column].values
        
        # Calculate linear regression
        coefficients = np.polyfit(x, y, 1)
        poly = np.poly1d(coefficients)
        
        # Generate forecast
        future_x = np.arange(len(df_sorted), len(df_sorted) + forecast_periods)
        forecast = poly(future_x)
        
        # Calculate R-squared
        y_pred = poly(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'forecast': forecast.tolist(),
            'slope': coefficients[0],
            'intercept': coefficients[1],
            'r_squared': r_squared,
            'historical_mean': np.mean(y),
            'historical_std': np.std(y)
        }

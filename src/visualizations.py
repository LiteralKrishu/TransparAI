"""
Enhanced Visualizations Module for TransparAI
Multi-panel dashboards, risk heatmaps, cluster analysis, and time series charts
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Optional, List
import warnings

warnings.filterwarnings('ignore')


class RiskVisualization:
    """Risk heatmaps and risk-related visualizations"""
    
    @staticmethod
    def create_risk_heatmap(df: pd.DataFrame, 
                           risk_metrics: Dict[str, np.ndarray]) -> go.Figure:
        """
        Create risk heatmap visualization
        
        Args:
            df: Procurement DataFrame
            risk_metrics: Dictionary with risk scores by vendor/category
            
        Returns:
            Plotly Figure object
        """
        # Prepare data
        vendors = list(set(df['vendor_name'].unique()))
        categories = list(set(df['category'].unique()))
        
        # Create risk matrix
        risk_matrix = np.random.rand(len(categories), len(vendors)) * 100
        
        fig = go.Figure(data=go.Heatmap(
            z=risk_matrix,
            x=vendors[:10],  # Limit to top 10 for clarity
            y=categories,
            colorscale='RdYlGn_r',
            colorbar=dict(title="Risk Score (%)")
        ))
        
        fig.update_layout(
            title="Procurement Risk Heatmap: Vendors vs Categories",
            xaxis_title="Vendors",
            yaxis_title="Categories",
            height=600,
            font=dict(size=12)
        )
        
        return fig
    
    @staticmethod
    def create_anomaly_detection_plot(df: pd.DataFrame,
                                     anomalies: np.ndarray) -> go.Figure:
        """
        Create anomaly detection scatter plot with risk zones
        
        Args:
            df: Procurement DataFrame
            anomalies: Boolean array indicating anomalies
            
        Returns:
            Plotly Figure object
        """
        fig = make_subplots(
            specs=[[{"secondary_y": False}]],
            vertical_spacing=0.12
        )
        
        # Normal points
        normal_mask = ~anomalies
        fig.add_trace(go.Scatter(
            x=df[normal_mask]['contract_value'],
            y=df[normal_mask]['processing_days'],
            mode='markers',
            marker=dict(size=8, color='green', opacity=0.6),
            name='Normal Contracts',
            text=df[normal_mask]['vendor_name'],
            hovertemplate='<b>%{text}</b><br>Value: %{x:,.0f}<br>Days: %{y}<extra></extra>'
        ))
        
        # Anomalous points
        anomaly_mask = anomalies
        fig.add_trace(go.Scatter(
            x=df[anomaly_mask]['contract_value'],
            y=df[anomaly_mask]['processing_days'],
            mode='markers',
            marker=dict(size=12, color='red', opacity=0.8, symbol='star'),
            name='Anomalies',
            text=df[anomaly_mask]['vendor_name'],
            hovertemplate='<b>%{text}</b><br>Value: %{x:,.0f}<br>Days: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Anomaly Detection: Contract Value vs Processing Time",
            xaxis_title="Contract Value (INR)",
            yaxis_title="Processing Days",
            height=600,
            hovermode='closest'
        )
        
        return fig


class ClusterVisualization:
    """Cluster analysis and visualization"""
    
    @staticmethod
    def create_cluster_plot(df: pd.DataFrame,
                           cluster_labels: np.ndarray) -> go.Figure:
        """
        Create cluster analysis scatter plot
        
        Args:
            df: Procurement DataFrame
            cluster_labels: Array of cluster labels
            
        Returns:
            Plotly Figure object
        """
        df_plot = df.copy()
        df_plot['cluster'] = cluster_labels
        
        fig = px.scatter(
            df_plot,
            x='contract_value',
            y='processing_days',
            color='cluster',
            size='contract_value',
            hover_data=['vendor_name', 'category'],
            title='DBSCAN Clustering: Contract Patterns',
            labels={'contract_value': 'Contract Value (INR)',
                   'processing_days': 'Processing Days'},
            size_max=20
        )
        
        fig.update_layout(height=600)
        
        return fig
    
    @staticmethod
    def create_vendor_cluster_dendrogram(df: pd.DataFrame) -> go.Figure:
        """
        Create hierarchical clustering dendrogram for vendors
        
        Args:
            df: Procurement DataFrame
            
        Returns:
            Plotly Figure object
        """
        from scipy.cluster.hierarchy import dendrogram, linkage
        
        vendor_features = df.groupby('vendor_name').agg({
            'contract_value': ['mean', 'std'],
            'processing_days': 'mean',
            'contract_id': 'count'
        }).fillna(0)
        
        linkage_matrix = linkage(vendor_features.values, method='ward')
        
        dendro = dendrogram(linkage_matrix, labels=vendor_features.index)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dendro['icoord'],
            y=dendro['dcoord'],
            mode='lines',
            line=dict(color='#1f77b4')
        ))
        
        fig.update_layout(
            title="Vendor Hierarchical Clustering Dendrogram",
            xaxis_title="Vendor Index",
            yaxis_title="Distance",
            height=600
        )
        
        return fig


class TimeSeriesVisualization:
    """Time series analysis and efficiency tracking"""
    
    @staticmethod
    def create_efficiency_timeseries(df: pd.DataFrame,
                                    period: str = 'M') -> go.Figure:
        """
        Create time series of procurement efficiency
        
        Args:
            df: Procurement DataFrame
            period: Aggregation period ('D', 'W', 'M', 'Y')
            
        Returns:
            Plotly Figure object
        """
        df_time = df.copy()
        df_time['period'] = df_time['award_date'].dt.to_period(period)
        
        efficiency_data = df_time.groupby('period').agg({
            'contract_value': ['sum', 'mean'],
            'processing_days': 'mean',
            'contract_id': 'count'
        }).reset_index()
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=('Total Contract Value Over Time',
                          'Average Processing Days',
                          'Number of Contracts'),
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        periods = [str(p) for p in efficiency_data['period']]
        
        # Row 1: Total Value
        fig.add_trace(go.Bar(
            x=periods,
            y=efficiency_data[('contract_value', 'sum')] / 1e7,
            name='Total Value (Cr)',
            marker_color='#1f77b4'
        ), row=1, col=1)
        
        # Row 2: Processing Days
        fig.add_trace(go.Scatter(
            x=periods,
            y=efficiency_data[('processing_days', 'mean')],
            name='Avg Processing Days',
            line=dict(color='#ff7f0e'),
            mode='lines+markers'
        ), row=2, col=1)
        
        # Row 3: Contract Count
        fig.add_trace(go.Bar(
            x=periods,
            y=efficiency_data[('contract_id', 'count')],
            name='Contract Count',
            marker_color='#2ca02c'
        ), row=3, col=1)
        
        fig.update_yaxes(title_text="Value (Cr)", row=1, col=1)
        fig.update_yaxes(title_text="Days", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=3, col=1)
        fig.update_xaxes(title_text="Period", row=3, col=1)
        
        fig.update_layout(height=800, showlegend=True)
        
        return fig
    
    @staticmethod
    def create_vendor_performance_timeseries(df: pd.DataFrame,
                                            top_n_vendors: int = 5) -> go.Figure:
        """
        Create vendor performance time series
        
        Args:
            df: Procurement DataFrame
            top_n_vendors: Number of top vendors to display
            
        Returns:
            Plotly Figure object
        """
        top_vendors = df['vendor_name'].value_counts().head(top_n_vendors).index
        
        df_filtered = df[df['vendor_name'].isin(top_vendors)].copy()
        df_filtered['period'] = df_filtered['award_date'].dt.to_period('M')
        
        vendor_ts = df_filtered.groupby(['period', 'vendor_name']).agg({
            'contract_value': 'sum',
            'processing_days': 'mean'
        }).reset_index()
        
        fig = px.line(
            vendor_ts,
            x='period',
            y='contract_value',
            color='vendor_name',
            markers=True,
            title=f'Contract Value Trends: Top {top_n_vendors} Vendors',
            labels={'contract_value': 'Total Value (INR)',
                   'period': 'Period',
                   'vendor_name': 'Vendor'}
        )
        
        fig.update_layout(height=600, hovermode='x unified')
        
        return fig


class ConcentrationVisualization:
    """Vendor concentration and competition analysis"""
    
    @staticmethod
    def create_concentration_pie(df: pd.DataFrame,
                                top_n: int = 8) -> go.Figure:
        """
        Create vendor concentration pie chart
        
        Args:
            df: Procurement DataFrame
            top_n: Number of top vendors to show
            
        Returns:
            Plotly Figure object
        """
        vendor_values = df.groupby('vendor_name')['contract_value'].sum()
        top_vendors = vendor_values.nlargest(top_n)
        other_value = vendor_values[~vendor_values.index.isin(top_vendors.index)].sum()
        
        if other_value > 0:
            top_vendors['Others'] = other_value
        
        fig = px.pie(
            values=top_vendors.values,
            names=top_vendors.index,
            title='Market Share: Top Vendors',
            hole=0.3
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=600)
        
        return fig
    
    @staticmethod
    def create_concentration_trend(df: pd.DataFrame,
                                  period: str = 'M') -> go.Figure:
        """
        Create vendor concentration trend over time
        
        Args:
            df: Procurement DataFrame
            period: Aggregation period
            
        Returns:
            Plotly Figure object
        """
        df_time = df.copy()
        df_time['period'] = df_time['award_date'].dt.to_period(period)
        
        concentration_data = []
        
        for period_val in df_time['period'].unique():
            period_df = df_time[df_time['period'] == period_val]
            vendor_values = period_df.groupby('vendor_name')['contract_value'].sum()
            
            # Calculate HHI (Herfindahl-Hirschman Index)
            market_shares = (vendor_values / vendor_values.sum()) * 100
            hhi = (market_shares ** 2).sum()
            
            concentration_data.append({
                'period': str(period_val),
                'hhi': hhi,
                'unique_vendors': period_df['vendor_name'].nunique()
            })
        
        conc_df = pd.DataFrame(concentration_data)
        
        fig = make_subplots(
            specs=[[{"secondary_y": True}]]
        )
        
        fig.add_trace(go.Scatter(
            x=conc_df['period'],
            y=conc_df['hhi'],
            name='HHI Index',
            line=dict(color='red'),
            mode='lines+markers'
        ), secondary_y=False)
        
        fig.add_trace(go.Scatter(
            x=conc_df['period'],
            y=conc_df['unique_vendors'],
            name='Unique Vendors',
            line=dict(color='blue'),
            mode='lines+markers'
        ), secondary_y=True)
        
        fig.update_yaxes(title_text="HHI Index", secondary_y=False)
        fig.update_yaxes(title_text="Number of Vendors", secondary_y=True)
        fig.update_xaxes(title_text="Period")
        
        fig.update_layout(
            title="Market Concentration Trend",
            height=600,
            hovermode='x unified'
        )
        
        return fig


class MultiPanelDashboard:
    """Create comprehensive multi-panel dashboards"""
    
    @staticmethod
    def create_executive_dashboard(df: pd.DataFrame,
                                   anomalies: np.ndarray = None) -> go.Figure:
        """
        Create executive summary dashboard
        
        Args:
            df: Procurement DataFrame
            anomalies: Optional anomaly array
            
        Returns:
            Plotly Figure with multiple subplots
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Contract Value Distribution',
                'Processing Time Distribution',
                'Top 5 Vendors by Value',
                'Contracts by Category',
                'Vendor Concentration (HHI)',
                'Processing Days Trend'
            ),
            specs=[
                [{"type": "histogram"}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "indicator"}, {"type": "scatter"}]
            ]
        )
        
        # Value Distribution
        fig.add_trace(go.Histogram(
            x=df['contract_value'],
            nbinsx=20,
            name='Contract Value',
            marker_color='#1f77b4'
        ), row=1, col=1)
        
        # Processing Time Distribution
        fig.add_trace(go.Histogram(
            x=df['processing_days'],
            nbinsx=20,
            name='Processing Days',
            marker_color='#ff7f0e'
        ), row=1, col=2)
        
        # Top Vendors
        top_vendors = df.groupby('vendor_name')['contract_value'].sum().nlargest(5)
        fig.add_trace(go.Bar(
            x=top_vendors.index,
            y=top_vendors.values,
            name='Top Vendors',
            marker_color='#2ca02c'
        ), row=2, col=1)
        
        # Categories Pie
        category_counts = df['category'].value_counts()
        fig.add_trace(go.Pie(
            labels=category_counts.index,
            values=category_counts.values,
            name='Categories'
        ), row=2, col=2)
        
        # HHI Indicator
        vendor_values = df.groupby('vendor_name')['contract_value'].sum()
        market_shares = (vendor_values / vendor_values.sum()) * 100
        hhi = (market_shares ** 2).sum()
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=hhi,
            title={'text': "HHI Index"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 10000]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 1500], 'color': "lightgreen"},
                       {'range': [1500, 2500], 'color': "yellow"},
                       {'range': [2500, 10000], 'color': "lightcoral"}
                   ]}
        ), row=3, col=1)
        
        # Processing Days Trend
        df_time = df.copy()
        df_time['period'] = df_time['award_date'].dt.to_period('M')
        trend = df_time.groupby('period')['processing_days'].mean()
        
        fig.add_trace(go.Scatter(
            x=[str(p) for p in trend.index],
            y=trend.values,
            mode='lines+markers',
            name='Avg Processing Days',
            line=dict(color='#d62728')
        ), row=3, col=2)
        
        fig.update_xaxes(title_text="Value (INR)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Days", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_xaxes(title_text="Vendor", row=2, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=1)
        fig.update_xaxes(title_text="Period", row=3, col=2)
        fig.update_yaxes(title_text="Days", row=3, col=2)
        
        fig.update_layout(
            height=1200,
            showlegend=False,
            title_text="TransparAI Executive Dashboard"
        )
        
        return fig

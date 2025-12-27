"""
TransparAI - Enhanced Procurement Analytics Dashboard
Main application integrating all analytics, financial, API, and visualization modules
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Import all custom modules
from src.analytics import (
    AnomalyDetection, StatisticalAnalysis, VendorAnalysis,
    CollusionDetection, EfficiencyAnalysis
)
from src.financial_analysis import (
    FinancialValuation, ROIAnalysis, CostEfficiencyAnalysis, BudgetForecasting
)
from src.api_integration import DataValidator, GemGovAPI, ProcurementDataAggregator
from src.visualizations import (
    RiskVisualization, ClusterVisualization, TimeSeriesVisualization,
    ConcentrationVisualization, MultiPanelDashboard
)
from src.data_generator import generate_sample_data


# Page configuration
st.set_page_config(
    page_title="TransparAI - Procurement Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;500;700&display=swap');

* {
    font-family: 'Noto Sans', sans-serif;
}

.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1486c9;
    text-align: center;
    margin-bottom: 1rem;
}

.sub-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #1486c9;
    margin: 1.5rem 0 1rem 0;
    border-bottom: 2px solid #1486c9;
    padding-bottom: 0.5rem;
}

.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.risk-high {
    background-color: #ffebee;
    color: #c62828;
}

.risk-medium {
    background-color: #fff3e0;
    color: #e65100;
}

.risk-low {
    background-color: #e8f5e9;
    color: #2e7d32;
}

.footer {
    text-align: center;
    margin-top: 3rem;
    padding: 2rem;
    background-color: #f8f9fa;
    border-radius: 10px;
    color: #666;
    border-top: 2px solid #1486c9;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state with sample data as default
if 'data' not in st.session_state:
    try:
        st.session_state.data = generate_sample_data(n_records=500)
        st.session_state.data_source = "Sample Data"
    except Exception as e:
        st.session_state.data = None
        st.session_state.data_source = None

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False


def load_data(data_source: str):
    """Load procurement data from selected source"""
    try:
        if data_source == "Sample Data":
            st.session_state.data = generate_sample_data(n_records=500)
            st.session_state.data_source = "Sample Data"
            st.success("✓ Sample data loaded successfully")
        elif data_source == "CSV File":
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file is not None:
                st.session_state.data = pd.read_csv(uploaded_file)
                st.session_state.data_source = "CSV File"
                st.success("✓ CSV file loaded successfully")
        elif data_source == "Government APIs":
            st.info("🔄 Fetching data from Government APIs...")
            gem_api = GemGovAPI()
            aggregator = ProcurementDataAggregator(gem_api=gem_api)
            
            # Get data with error tracking
            df, source_name, errors = aggregator.fetch_and_aggregate_data(max_records=500)
            
            # Check if we got valid data
            if df is not None and not df.empty:
                st.session_state.data = df
                st.session_state.data_source = source_name
                
                # Show warnings if API had issues but fallback worked
                if errors:
                    st.warning("⚠️ API Connection Issues Detected")
                    with st.expander("Show API Error Details"):
                        for error in errors:
                            st.write(f"• {error}")
                
                # Show success message
                if "Fallback" in source_name:
                    st.info(f"ℹ️ Using {source_name} ({len(df)} records)")
                else:
                    st.success(f"✓ Data loaded from {source_name} ({len(df)} records)")
            else:
                # Complete failure - no data at all
                st.error("❌ Failed to load data from any source")
                if errors:
                    st.error("Error details:")
                    for error in errors:
                        st.write(f"• {error}")
                
                # Fallback to sample data as last resort
                st.warning("🔄 Loading sample data as fallback...")
                st.session_state.data = generate_sample_data(n_records=500)
                st.session_state.data_source = "Sample Data (Emergency Fallback)"
                st.info("✓ Sample data loaded successfully")
                
    except Exception as e:
        st.error(f"❌ Unexpected error loading data: {str(e)}")
        # Emergency fallback
        try:
            st.warning("🔄 Attempting to load sample data as fallback...")
            st.session_state.data = generate_sample_data(n_records=500)
            st.session_state.data_source = "Sample Data (Error Recovery)"
            st.success("✓ Sample data loaded successfully")
        except Exception as fallback_error:
            st.error(f"❌ Critical error: Could not load any data: {str(fallback_error)}")


def display_data_quality_metrics():
    """Display data quality assessment"""
    if st.session_state.data is None:
        st.warning("No data loaded")
        return
    
    df = st.session_state.data
    report = DataValidator.get_data_quality_report(df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", report['total_records'], 
                 help="Total number of procurement records")
    
    with col2:
        st.metric("Unique Vendors", report['unique_vendors'],
                 help="Number of unique vendors in dataset")
    
    with col3:
        st.metric("Duplicate Records", report['duplicate_records'],
                 help="Number of duplicate records detected")
    
    with col4:
        completeness = ((len(df) - sum(report['missing_values'].values())) / len(df) * 100)
        st.metric("Data Completeness", f"{completeness:.1f}%",
                 help="Percentage of non-null values")


def perform_anomaly_analysis():
    """Perform comprehensive anomaly detection"""
    if st.session_state.data is None:
        st.warning("No data loaded for analysis")
        return
    
    df = st.session_state.data
    
    st.markdown("### <div class='sub-header'>🔍 Anomaly Detection & Risk Analysis</div>",
               unsafe_allow_html=True)
    
    # Detection method selection
    detection_methods = st.multiselect(
        "Select anomaly detection methods:",
        ["Z-Score", "IQR", "Isolation Forest"],
        default=["Z-Score", "Isolation Forest"]
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if "Z-Score" in detection_methods:
            z_anomalies = AnomalyDetection.zscore_detection(df['contract_value'], threshold=3.0)
            st.write(f"**Z-Score Anomalies:** {z_anomalies.sum()} detected")
        
        if "IQR" in detection_methods:
            iqr_anomalies = AnomalyDetection.iqr_detection(df['contract_value'])
            st.write(f"**IQR Anomalies:** {iqr_anomalies.sum()} detected")
        
        if "Isolation Forest" in detection_methods:
            iso_anomalies = AnomalyDetection.isolation_forest_detection(
                df, ['contract_value', 'processing_days'], contamination=0.1
            )
            st.write(f"**Isolation Forest Anomalies:** {iso_anomalies.sum()} detected")
            
            # Display anomaly visualization
            fig = RiskVisualization.create_anomaly_detection_plot(df, iso_anomalies)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        anomaly_rate = (iso_anomalies.sum() / len(df) * 100) if "Isolation Forest" in detection_methods else 0
        
        if anomaly_rate > 15:
            risk_color = "🔴 HIGH"
        elif anomaly_rate > 8:
            risk_color = "🟡 MEDIUM"
        else:
            risk_color = "🟢 LOW"
        
        st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%", delta=risk_color)


def perform_vendor_analysis():
    """Perform vendor analysis and concentration metrics"""
    if st.session_state.data is None:
        st.warning("No data loaded for analysis")
        return
    
    df = st.session_state.data
    
    st.markdown("### <div class='sub-header'>🏢 Vendor Analysis & Market Concentration</div>",
               unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        hhi = VendorAnalysis.herfindahl_index(df)
        if hhi > 2500:
            status = "🔴 Highly Concentrated"
        elif hhi > 1500:
            status = "🟡 Moderately Concentrated"
        else:
            status = "🟢 Competitive"
        
        st.metric("HHI Index", f"{hhi:.0f}", delta=status)
    
    with col2:
        cr4 = VendorAnalysis.vendor_concentration_ratio(df, top_n=4)
        st.metric("CR-4 (Top 4 Vendors)", f"{cr4:.1f}%",
                 help="Market share of top 4 vendors")
    
    with col3:
        diversity = VendorAnalysis.vendor_diversity_index(df)
        st.metric("Diversity Index", f"{diversity:.3f}",
                 help="Shannon entropy (0-1 scale)")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = ConcentrationVisualization.create_concentration_pie(df, top_n=8)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_heatmap = RiskVisualization.create_risk_heatmap(df, {})
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Vendor details
    st.markdown("#### Top Vendors Analysis")
    vendor_stats = VendorAnalysis.vendor_repeat_analysis(df).head(10)
    st.dataframe(vendor_stats, use_container_width=True)


def perform_collusion_analysis():
    """Detect potential collusion patterns"""
    if st.session_state.data is None:
        st.warning("No data loaded for analysis")
        return
    
    df = st.session_state.data
    
    st.markdown("### <div class='sub-header'>🚨 Collusion Pattern Detection</div>",
               unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bid clustering
        bid_result = CollusionDetection.detect_bid_clustering(df, threshold=0.05)
        
        st.subheader("Suspicious Bid Clustering")
        st.metric("Similar Bids Found", bid_result['similar_bids_count'])
        
        if bid_result['suspicious']:
            st.warning("⚠️ Suspicious bid clustering patterns detected!")
        else:
            st.success("✓ No suspicious bid clustering detected")
    
    with col2:
        # Price patterns
        price_patterns = CollusionDetection.detect_price_patterns(df)
        
        suspicious_count = sum(1 for v in price_patterns.values() if v.get('suspicious', False))
        st.subheader("Vendor Price Pattern Analysis")
        st.metric("Suspicious Vendors", suspicious_count)
        
        if suspicious_count > 0:
            st.warning(f"⚠️ {suspicious_count} vendors show suspicious price patterns!")


def perform_financial_analysis():
    """Perform financial analysis and ROI calculations"""
    if st.session_state.data is None:
        st.warning("No data loaded for analysis")
        return
    
    df = st.session_state.data
    
    st.markdown("### <div class='sub-header'>💰 Financial Analysis & ROI</div>",
               unsafe_allow_html=True)
    
    # Financial metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_spend = df['contract_value'].sum()
    avg_contract = df['contract_value'].mean()
    max_contract = df['contract_value'].max()
    min_contract = df['contract_value'].min()
    
    with col1:
        st.metric("Total Procurement Spend", f"INR {total_spend/1e7:.1f} Cr")
    
    with col2:
        st.metric("Average Contract Value", f"INR {avg_contract/1e5:.1f} L")
    
    with col3:
        st.metric("Maximum Contract", f"INR {max_contract/1e5:.1f} L")
    
    with col4:
        st.metric("Minimum Contract", f"INR {min_contract:.0f}")
    
    # Cost efficiency analysis
    st.markdown("#### Cost Efficiency Analysis")
    cost_analysis = CostEfficiencyAnalysis.category_cost_analysis(df)
    st.dataframe(cost_analysis, use_container_width=True)
    
    # Procurement cycle analysis
    cycle_analysis = CostEfficiencyAnalysis.procurement_cycle_cost(df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg Processing Days", f"{cycle_analysis['avg_processing_days']:.0f}")
    
    with col2:
        st.metric("Cost Per Day", f"INR {cycle_analysis['cost_per_day']/1e4:.1f} L")
    
    with col3:
        st.metric("Time-Cost Correlation", f"{cycle_analysis['time_cost_correlation']:.3f}")


def perform_efficiency_analysis():
    """Analyze procurement efficiency"""
    if st.session_state.data is None:
        st.warning("No data loaded for analysis")
        return
    
    df = st.session_state.data
    
    st.markdown("### <div class='sub-header'>⚡ Procurement Efficiency Analysis</div>",
               unsafe_allow_html=True)
    
    # Efficiency scores
    efficiency_scores = EfficiencyAnalysis.efficiency_score(df)
    df['efficiency_score'] = efficiency_scores
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_efficiency = efficiency_scores.mean()
        st.metric("Average Efficiency Score", f"{avg_efficiency:.2f}/1.00")
    
    with col2:
        high_efficiency = (efficiency_scores > 0.7).sum()
        st.metric("Highly Efficient Contracts", high_efficiency)
    
    with col3:
        low_efficiency = (efficiency_scores < 0.3).sum()
        st.metric("Low Efficiency Contracts", low_efficiency)
    
    # Competitive bidding analysis
    st.markdown("#### Competitive Bidding Analysis")
    bidding_analysis = EfficiencyAnalysis.competitive_bidding_analysis(df)
    
    bidding_df = pd.DataFrame(bidding_analysis).T
    st.dataframe(bidding_df, use_container_width=True)
    
    # Time series visualization
    fig_ts = TimeSeriesVisualization.create_efficiency_timeseries(df)
    st.plotly_chart(fig_ts, use_container_width=True)


def display_home_summary():
    """Display enhanced home page with data summary"""
    # The global page header is rendered in `main()`; show a concise subtitle here to avoid duplication
    st.markdown('### 📊 Home — Data Summary', unsafe_allow_html=True)
    st.markdown("---")
    
    if st.session_state.data is not None:
        df = st.session_state.data
        
        # Overview Section
        st.markdown('<div class="sub-header">📈 Data Summary Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Contracts", f"{len(df):,}", help="Total number of procurement records")
        
        with col2:
            total_value = df['contract_value'].sum() / 1e7
            st.metric("Total Value", f"₹{total_value:.1f} Cr", help="Total procurement value in Crores")
        
        with col3:
            avg_contract = df['contract_value'].mean() / 1e5
            st.metric("Avg Contract", f"₹{avg_contract:.1f} L", help="Average contract value in Lakhs")
        
        with col4:
            unique_vendors = df['vendor_name'].nunique()
            st.metric("Unique Vendors", unique_vendors, help="Number of unique vendors")
        
        # Key Insights
        st.markdown('<div class="sub-header">🔍 Key Insights</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            hhi = VendorAnalysis.herfindahl_index(df)
            if hhi > 2500:
                status = "🔴 Highly Concentrated"
                color = "danger"
            elif hhi > 1500:
                status = "🟡 Moderately Concentrated"
                color = "warning"
            else:
                status = "🟢 Competitive"
                color = "success"
            
            st.info(f"**Market Concentration (HHI):** {hhi:.0f}\n\n{status}")
        
        with col2:
            cr4 = VendorAnalysis.vendor_concentration_ratio(df, top_n=4)
            st.info(f"**Top 4 Vendors (CR-4):** {cr4:.1f}%\n\nMarket share of leading vendors")
        
        with col3:
            diversity = VendorAnalysis.vendor_diversity_index(df)
            st.info(f"**Vendor Diversity:** {diversity:.3f}\n\nHigher values = more diverse")
        
        # Processing Efficiency
        st.markdown('<div class="sub-header">⏱️ Procurement Cycle Efficiency</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_processing = df['processing_days'].mean()
            st.metric("Avg Processing Time", f"{avg_processing:.0f} days")
        
        with col2:
            min_processing = df['processing_days'].min()
            max_processing = df['processing_days'].max()
            st.metric("Processing Range", f"{min_processing:.0f} - {max_processing:.0f} days")
        
        with col3:
            fast_tracked = (df['processing_days'] < 30).sum()
            st.metric("Fast-Tracked (<30 days)", f"{fast_tracked} ({fast_tracked/len(df)*100:.1f}%)")
        
        # Data Quality & Anomalies
        st.markdown('<div class="sub-header">✅ Data Quality & Anomalies</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            is_valid, errors = DataValidator.validate_contract_data(df)
            if is_valid:
                st.success("✓ Data Quality: PASSED")
            else:
                st.warning(f"⚠️ Issues found: {len(errors)}")
        
        with col2:
            iso_anomalies = AnomalyDetection.isolation_forest_detection(
                df, ['contract_value', 'processing_days'], contamination=0.1
            )
            anomaly_pct = (iso_anomalies.sum() / len(df) * 100)
            st.metric("Anomalies Detected", f"{iso_anomalies.sum()} ({anomaly_pct:.1f}%)")
        
        with col3:
            report = DataValidator.get_data_quality_report(df)
            completeness = ((len(df) - sum(report['missing_values'].values())) / len(df) * 100)
            st.metric("Data Completeness", f"{completeness:.1f}%")
        
        # Financial Summary
        st.markdown('<div class="sub-header">💰 Financial Summary</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            min_value = df['contract_value'].min() / 1e5
            st.metric("Min Contract", f"₹{min_value:.1f} L")
        
        with col2:
            max_value = df['contract_value'].max() / 1e7
            st.metric("Max Contract", f"₹{max_value:.1f} Cr")
        
        with col3:
            median_value = df['contract_value'].median() / 1e5
            st.metric("Median Contract", f"₹{median_value:.1f} L")
        
        with col4:
            std_value = df['contract_value'].std() / 1e5
            st.metric("Std Deviation", f"₹{std_value:.1f} L")
        
        # Category & Ministry Analysis
        st.markdown('<div class="sub-header">📊 Category & Ministry Distribution</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'category' in df.columns:
                categories = df['category'].value_counts()
                st.write("**Top Categories:**")
                for cat, count in categories.head(5).items():
                    st.write(f"  • {cat}: {count} contracts")
        
        with col2:
            if 'ministry' in df.columns:
                ministries = df['ministry'].value_counts()
                st.write("**Top Ministries:**")
                for min, count in ministries.head(5).items():
                    st.write(f"  • {min}: {count} contracts")
        
        # Top Vendors
        st.markdown('<div class="sub-header">🏢 Top Vendors</div>', unsafe_allow_html=True)
        
        top_vendors = df.groupby('vendor_name')['contract_value'].agg(['sum', 'count']).sort_values('sum', ascending=False).head(10)
        top_vendors.columns = ['Total Value (₹)', 'Contract Count']
        top_vendors['Total Value (₹)'] = top_vendors['Total Value (₹)'].apply(lambda x: f"₹{x/1e7:.1f} Cr")
        
        st.dataframe(top_vendors, use_container_width=True)
        
        # Quick Actions
        st.markdown('<div class="sub-header">⚡ Quick Actions</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔍 Run Anomaly Detection", use_container_width=True):
                st.session_state.page = "Anomaly Detection"
                st.rerun()
        
        with col2:
            if st.button("🏢 Vendor Analysis", use_container_width=True):
                st.session_state.page = "Vendor Analysis"
                st.rerun()
        
        with col3:
            if st.button("📊 Executive Dashboard", use_container_width=True):
                st.session_state.page = "Executive Dashboard"
                st.rerun()
        
        # Download Options
        st.markdown("---")
        st.markdown('<div class="sub-header">📥 Download Options</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Full Dataset (CSV)",
                csv,
                "procurement_data.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            json_str = df.to_json(orient='records', date_format='iso')
            st.download_button(
                "📥 Download Data (JSON)",
                json_str,
                "procurement_data.json",
                "application/json",
                use_container_width=True
            )
    
    else:
        st.info("📥 Load sample data to get started! Use the 'Load Data' button in the sidebar.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🚀 Getting Started
            
            1. Click **'Load Data'** in the sidebar (Sample Data is recommended)
            2. Explore the data summary above
            3. Use navigation to run different analyses
            4. Download reports as needed
            """)
        
        with col2:
            st.markdown("""
            ### 📖 Available Features
            
            - 🔍 Anomaly Detection
            - 🏢 Vendor Analysis
            - 🚨 Collusion Detection
            - 💰 Financial Analysis
            - ⚡ Efficiency Analysis
            - 📊 Executive Dashboard
            - 📈 Statistics
            """)


def display_executive_dashboard():
    """Display comprehensive executive dashboard"""
    if st.session_state.data is None:
        st.warning("No data loaded for dashboard")
        return
    
    df = st.session_state.data
    
    st.markdown("### <div class='sub-header'>📊 Executive Dashboard</div>",
               unsafe_allow_html=True)
    
    fig_dashboard = MultiPanelDashboard.create_executive_dashboard(df)
    st.plotly_chart(fig_dashboard, use_container_width=True)


def main():
    """Main application"""
    st.markdown('<div class="main-header">TransparAI</div>', unsafe_allow_html=True)
    st.markdown("**Advanced Procurement Analytics & Transparency Platform**")
    st.markdown("Powered by SFLC.in for Defending Digital Rights")
    st.markdown("---")
    
    # Sidebar with improved layout
    with st.sidebar:
        st.markdown("### 🎯 Navigation")
        
        page = st.radio(
            "Select Analysis:",
            ["Home", "Data Management", "Anomaly Detection", "Vendor Analysis",
             "Collusion Detection", "Financial Analysis", "Efficiency Analysis",
             "Executive Dashboard", "Statistics", "Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 📥 Data Source")
        
        current_source = st.session_state.get("data_source", "Sample Data")
        data_source = st.selectbox(
            "Select data source:",
            ["Sample Data", "CSV File", "Government APIs"],
            index=["Sample Data", "CSV File", "Government APIs"].index(current_source) if current_source in ["Sample Data", "CSV File", "Government APIs"] else 0,
            label_visibility="collapsed"
        )
        
        if st.button("🔄 Load Data", use_container_width=True):
            with st.spinner("Loading data..."):
                load_data(data_source)
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        
        if st.session_state.data is not None:
            df = st.session_state.data
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Records", f"{len(df):,}")
            with col2:
                st.metric("Columns", len(df.columns))
            
            st.write(f"**Source:** {st.session_state.get('data_source', 'Unknown')}")
            st.write(f"**Memory:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        else:
            st.warning("No data loaded")
    
    # Main content
    if page == "Home":
        display_home_summary()
    
    elif page == "Data Management":
        st.markdown("### <div class='sub-header'>📁 Data Management</div>",
                   unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            display_data_quality_metrics()
            
            st.markdown("#### Data Preview")
            st.dataframe(st.session_state.data.head(20), use_container_width=True)
            
            st.markdown("#### Download Data")
            csv = st.session_state.data.to_csv(index=False)
            st.download_button(
                "Download as CSV",
                csv,
                "procurement_data.csv",
                "text/csv"
            )
        else:
            st.info("Please load data from the sidebar first")
    
    elif page == "Anomaly Detection":
        perform_anomaly_analysis()
    
    elif page == "Vendor Analysis":
        perform_vendor_analysis()
    
    elif page == "Collusion Detection":
        perform_collusion_analysis()
    
    elif page == "Financial Analysis":
        perform_financial_analysis()
    
    elif page == "Efficiency Analysis":
        perform_efficiency_analysis()
    
    elif page == "Executive Dashboard":
        display_executive_dashboard()
    
    elif page == "Statistics":
        st.markdown("### <div class='sub-header'>📈 Statistical Summary</div>",
                   unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            st.dataframe(st.session_state.data.describe(), use_container_width=True)
            
            st.markdown("#### Correlation Matrix")
            numeric_df = st.session_state.data.select_dtypes(include=[np.number])
            corr_matrix = StatisticalAnalysis.correlation_matrix(numeric_df)
            st.dataframe(corr_matrix, use_container_width=True)
        else:
            st.info("Please load data from the sidebar first")
    
    elif page == "Settings":
        st.markdown("### <div class='sub-header'>⚙️ Settings</div>",
                   unsafe_allow_html=True)
        
        st.markdown("**Anomaly Detection Sensitivity**")
        sensitivity = st.slider("Z-score threshold:", 1.0, 5.0, 3.0, 0.1)
        
        st.markdown("**Visualization Options**")
        show_grid = st.checkbox("Show grid lines", value=True)
        show_legend = st.checkbox("Show legend", value=True)
        
        st.markdown("**Data Validation**")
        if st.button("Validate Current Dataset"):
            if st.session_state.data is not None:
                is_valid, errors = DataValidator.validate_contract_data(st.session_state.data)
                
                if is_valid:
                    st.success("✓ Data validation passed!")
                else:
                    st.error("Data validation issues found:")
                    for error in errors:
                        st.write(f"  • {error}")
            else:
                st.warning("No data loaded")
    
    # Footer
    st.markdown("""
    ---
    <div class='footer'>
    <p>TransparAI - Open Source Initiative by SFLC.in</p>
    <p>Defending Digital Rights through Open Source Innovation</p>
    <p style='font-size: 0.8rem; margin-top: 1rem;'>
    © 2025 Software Freedom Law Center. All rights reserved.
    </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

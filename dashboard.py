"""
TransparAI - Enhanced Procurement Analytics Dashboard
Main application integrating all analytics, financial, API, and visualization modules
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)

# Import all custom modules
from src.analytics import (
    AnomalyDetection, StatisticalAnalysis, VendorAnalysis,
    CollusionDetection, EfficiencyAnalysis, ClusteringAnalysis
)
from src.financial_analysis import (
    FinancialValuation, ROIAnalysis, CostEfficiencyAnalysis, BudgetForecasting
)
from src.api_integration import DataValidator
from src.configuration import load_settings, configured_aggregator
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

* {
    font-family: sans-serif;
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

SETTINGS = load_settings()


def initialize_state():
    if 'data' not in st.session_state:
        st.session_state.data = generate_sample_data(n_records=500)
        st.session_state.data_source = 'Sample Data'
        st.session_state.loaded_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
        st.session_state.live_loaded = False
    st.session_state.setdefault('analysis_settings', {
        'zscore_threshold': SETTINGS['anomaly_detection']['zscore_threshold'],
        'contamination': SETTINGS['anomaly_detection']['contamination'],
        **SETTINGS['visualization']})
    st.session_state.setdefault('navigation', 'Home')
    st.session_state.setdefault('load_messages', [])


def deployment_aggregator():
    try:
        secrets = dict(st.secrets)
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        secrets = {}
    return configured_aggregator(SETTINGS, secrets)


def load_data(data_source: str, uploaded_file=None):
    """Commit a new dataset only after normalization and validation succeed."""
    messages = []
    try:
        if data_source == 'Sample Data':
            df, source = generate_sample_data(n_records=500), 'Sample Data'
        elif data_source == 'CSV File':
            if uploaded_file is None:
                raise ValueError('Select a CSV file before loading data')
            if uploaded_file.size > SETTINGS['limits']['max_upload_mb'] * 1024**2:
                raise ValueError('CSV file exceeds the 10 MB upload limit')
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, nrows=SETTINGS['limits']['max_records'] + 1)
            source = 'Uploaded CSV'
        elif data_source == 'Government APIs' and SETTINGS.get('live_enabled', False):
            aggregator = deployment_aggregator()
            df, source, messages = aggregator.fetch_and_aggregate_data(max_records=500)
            if df.empty:
                raise ValueError('; '.join(messages))
        else:
            raise ValueError('Live sources are disabled; use CSV or sample data')
        df = DataValidator.prepare_contract_data(df)
    except (ValueError, TypeError, OSError, pd.errors.ParserError) as exc:
        st.session_state.load_messages = [('error', str(exc)),
            ('info', 'The previous dataset is still active.')]
        return False
    st.session_state.data = df
    st.session_state.data_source = source
    st.session_state.live_loaded = data_source == 'Government APIs'
    st.session_state.loaded_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
    st.session_state.load_messages = [('warning', message) for message in messages]
    st.session_state.load_messages.append(('success', f'Loaded {len(df):,} records from {source}'))
    return True


@st.cache_data(ttl=300, max_entries=32, show_spinner=False)
def review_flags(df, contamination, random_state):
    return AnomalyDetection.isolation_forest_detection(
        df, ['contract_value', 'processing_days'], contamination=contamination, random_state=random_state)


def current_flags(df):
    return review_flags(df, st.session_state.analysis_settings['contamination'],
                        SETTINGS['anomaly_detection']['random_state'])


def show_chart(fig, **kwargs):
    options = st.session_state.analysis_settings
    fig.update_layout(showlegend=options['show_legend'])
    fig.update_xaxes(showgrid=options['show_grid'])
    fig.update_yaxes(showgrid=options['show_grid'])
    st.plotly_chart(fig, width='stretch')


def navigate(page):
    st.session_state.navigation = page


def save_setting(name):
    st.session_state.analysis_settings[name] = st.session_state['setting_' + name]


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
        completeness = report['completeness_pct']
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

    st.caption('Statistical outliers are review signals. They do not establish fraud or bias. '
               'Isolation Forest uses the configured expected outlier share.')
    methods = st.multiselect('Select anomaly detection methods:',
        ['Z-Score', 'IQR', 'Isolation Forest'], default=['Z-Score', 'Isolation Forest'])
    masks = []
    if 'Z-Score' in methods:
        flags = AnomalyDetection.zscore_detection(df.contract_value,
            threshold=st.session_state.analysis_settings['zscore_threshold'])
        masks.append(flags)
        st.write(f'**Z-Score Anomalies:** {flags.sum()} detected')
    if 'IQR' in methods:
        flags = AnomalyDetection.iqr_detection(df.contract_value).to_numpy()
        masks.append(flags)
        st.write(f'**IQR Anomalies:** {flags.sum()} detected')
    if 'Isolation Forest' in methods:
        flags = current_flags(df)
        masks.append(flags)
        st.write(f'**Isolation Forest Anomalies:** {flags.sum()} detected')
    if not masks:
        st.info('Select a method to calculate review flags.')
        return
    flags = np.logical_or.reduce(masks)
    st.metric('Contracts flagged for review', int(flags.sum()))
    show_chart(RiskVisualization.create_anomaly_detection_plot(df, flags))
    st.dataframe(df.loc[flags], width='stretch')


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
            status = "🔴 Higher concentration in this dataset"
        elif hhi > 1500:
            status = "🟡 Moderate concentration in this dataset"
        else:
            status = "🟢 Lower concentration in this dataset"

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
        show_chart(fig_pie, width='stretch')

    with col2:
        fig_heatmap = RiskVisualization.create_risk_heatmap(df, {'review_flags': current_flags(df)})
        show_chart(fig_heatmap, width='stretch')

    # Vendor details
    st.markdown("#### Top Vendors Analysis")
    vendor_stats = VendorAnalysis.vendor_repeat_analysis(df).head(10)
    st.dataframe(vendor_stats, width='stretch')


def perform_collusion_analysis():
    """Detect potential collusion patterns"""
    if st.session_state.data is None:
        st.warning("No data loaded for analysis")
        return

    df = st.session_state.data

    st.markdown("### <div class='sub-header'>🔎 Bid and Price Review Signals</div>",
               unsafe_allow_html=True)

    st.caption('Price similarity alone does not establish collusion. Comparisons need '
               'competing bids for the same tender, item, quantity, and price basis.')
    result = CollusionDetection.detect_bid_clustering(df)
    if not result['available']:
        st.info(result['reason'])
    else:
        st.metric('Similar bid pairs for review', result['similar_bids_count'])
        st.dataframe(pd.DataFrame(result['similar_bids']), width='stretch')
        if result['truncated']:
            st.caption('Showing the first 100 pairs; the count includes all matching pairs.')
    patterns = CollusionDetection.detect_price_patterns(df)
    st.subheader('Contract price variation by vendor and category')
    st.dataframe(pd.DataFrame(patterns).T, width='stretch')


def perform_financial_analysis():
    """Perform financial analysis and ROI calculations"""
    if st.session_state.data is None:
        st.warning("No data loaded for analysis")
        return

    df = st.session_state.data

    st.markdown("### <div class='sub-header'>💰 Financial Analysis & Scenarios</div>",
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
    st.dataframe(cost_analysis, width='stretch')

    # Procurement cycle analysis
    cycle_analysis = CostEfficiencyAnalysis.procurement_cycle_cost(df)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Avg Processing Days", f"{cycle_analysis['avg_processing_days']:.0f}")

    with col2:
        st.metric("Cost Per Day", f"INR {cycle_analysis['cost_per_day']/1e4:.1f} L")

    with col3:
        st.metric("Time-Cost Correlation", f"{cycle_analysis['time_cost_correlation']:.3f}")

    st.markdown('#### Investment scenario')
    st.caption('Enter assumed annual net cash flows. These are scenario inputs, not measured procurement returns.')
    investment = st.number_input('Initial investment (INR)', min_value=1.0, value=1000000.0)
    discount = st.number_input('Annual discount rate (%)', min_value=0.0, max_value=100.0, value=10.0)
    text = st.text_input('Annual net cash flows (INR, comma separated)', '400000,400000,400000')
    try:
        flows = [float(value.strip()) for value in text.split(',')]
        if not flows or len(flows) > 100 or not np.isfinite(flows).all() or any(v < 0 for v in flows):
            raise ValueError('Enter 1 to 100 finite, nonnegative annual cash flows')
        npv = FinancialValuation.calculate_npv(flows, discount / 100, investment)
        st.metric('Scenario NPV (INR)', f'{npv:,.0f}')
        st.metric('Scenario ROI (%)', f'{ROIAnalysis.calculate_roi(sum(flows)-investment, investment):.1f}')
        payback = FinancialValuation.calculate_payback_period(flows, investment)
        st.metric('Payback (years)', f'{payback:.2f}' if np.isfinite(payback) else 'Not reached')
        if any(flows):
            st.metric('Scenario IRR (%)', f'{FinancialValuation.calculate_irr(flows, investment)*100:.2f}')
    except (ValueError, OverflowError, ZeroDivisionError):
        st.warning('Use finite nonnegative cash flows. The scenario could not be calculated.')


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
    st.dataframe(bidding_df, width='stretch')

    # Time series visualization
    fig_ts = TimeSeriesVisualization.create_efficiency_timeseries(df)
    show_chart(fig_ts, width='stretch')


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
                status = "🔴 Higher concentration in this dataset"
                color = "danger"
            elif hhi > 1500:
                status = "🟡 Moderate concentration in this dataset"
                color = "warning"
            else:
                status = "🟢 Lower concentration in this dataset"
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
            delayed = (df.processing_days > SETTINGS['thresholds']['processing_delay']).sum()
            st.caption(f"{delayed} contracts exceed the configured processing-delay threshold.")

        # Data Quality & Review Flags
        st.markdown('<div class="sub-header">✅ Data Quality & Review Flags</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            is_valid, errors = DataValidator.validate_contract_data(df)
            if is_valid:
                st.success("✓ Data Quality: PASSED")
            else:
                st.warning(f"⚠️ Issues found: {len(errors)}")

        with col2:
            iso_anomalies = current_flags(df)
            anomaly_pct = (iso_anomalies.sum() / len(df) * 100)
            st.metric("Contracts flagged for review", f"{iso_anomalies.sum()} ({anomaly_pct:.1f}%)")

        with col3:
            report = DataValidator.get_data_quality_report(df)
            completeness = report['completeness_pct']
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

        st.dataframe(top_vendors, width='stretch')

        # Quick Actions
        st.markdown('<div class="sub-header">⚡ Quick Actions</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.button('🔍 Run Anomaly Detection', on_click=navigate, args=('Anomaly Detection',), width='stretch')
        with col2:
            st.button('🏢 Vendor Analysis', on_click=navigate, args=('Vendor Analysis',), width='stretch')
        with col3:
            st.button('📊 Executive Dashboard', on_click=navigate, args=('Executive Dashboard',), width='stretch')

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
                width='stretch'
            )

        with col2:
            json_str = df.to_json(orient='records', date_format='iso')
            st.download_button(
                "📥 Download Data (JSON)",
                json_str,
                "procurement_data.json",
                "application/json",
                width='stretch'
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
    show_chart(fig_dashboard, width='stretch')


def main():
    """Main application"""
    initialize_state()
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
             "Executive Dashboard", "Statistics", "Clustering", "Forecasting", "Settings"],
            key="navigation",
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### 📥 Data Source")

        sources = ['Sample Data', 'CSV File']
        if SETTINGS.get('live_enabled', False):
            sources.append('Government APIs')
        data_source = st.selectbox('Select data source:', sources, key='source_selection')
        uploaded = st.file_uploader('Upload CSV file', type=['csv'], key='csv_upload',
            max_upload_size=SETTINGS['limits']['max_upload_mb']) if data_source == 'CSV File' else None
        if st.button('🔄 Load Data', width='stretch'):
            load_data(data_source, uploaded)
        auto_refresh = False
        if SETTINGS.get('live_enabled', False):
            auto_refresh = st.checkbox('Refresh live data every 5 minutes', value=False)
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

    for level, message in st.session_state.load_messages:
        getattr(st, level)(message)
    source = st.session_state.get('data_source', 'None')
    st.info(f'Active dataset: {source}. Loaded at {st.session_state.get("loaded_at", "unknown")} (UTC).')
    if not st.session_state.get('live_loaded'):
        st.caption('This is not a live government feed. Sample Data contains generated fictional records.')

    @st.fragment(run_every=SETTINGS['refresh_seconds'] if auto_refresh else None)
    def refresh_live_data():
        if auto_refresh and st.session_state.get('live_loaded'):
            loaded = datetime.fromisoformat(st.session_state.loaded_at)
            attempted = st.session_state.get('refresh_attempt_at', loaded)
            if (datetime.now(timezone.utc) - max(loaded, attempted)).total_seconds() >= SETTINGS['refresh_seconds']:
                st.session_state.refresh_attempt_at = datetime.now(timezone.utc)
                load_data('Government APIs')
                st.rerun()
    refresh_live_data()

    # Main content
    if page == "Home":
        display_home_summary()

    elif page == "Data Management":
        st.markdown("### <div class='sub-header'>📁 Data Management</div>",
                   unsafe_allow_html=True)

        if st.session_state.data is not None:
            display_data_quality_metrics()

            st.markdown("#### Data Preview")
            st.dataframe(st.session_state.data.head(20), width='stretch')

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
            st.dataframe(st.session_state.data.describe(), width='stretch')

            st.markdown("#### Correlation Matrix")
            numeric_df = st.session_state.data.select_dtypes(include=[np.number])
            corr_matrix = StatisticalAnalysis.correlation_matrix(numeric_df)
            st.dataframe(corr_matrix, width='stretch')
        else:
            st.info("Please load data from the sidebar first")

    elif page == 'Clustering':
        df = st.session_state.data
        st.subheader('Contract pattern clusters')
        if len(df) > 2000:
            st.info('Clustering uses the first 2,000 records to limit memory use.')
            df = df.head(2000)
        labels = ClusteringAnalysis.dbscan_clustering(df, ['contract_value', 'processing_days'])
        st.caption('Clusters show similar values and durations. Noise points use label -1.')
        show_chart(ClusterVisualization.create_cluster_plot(df, labels))
        show_chart(TimeSeriesVisualization.create_vendor_performance_timeseries(df))

    elif page == 'Forecasting':
        st.subheader('Monthly spending projection')
        df = st.session_state.data
        monthly = df.set_index('award_date').contract_value.resample('MS').sum().reset_index()
        periods = st.slider('Months to project', 1, 24, 6)
        if len(monthly) < 3:
            st.info('At least three months of observations are needed.')
        else:
            forecast = BudgetForecasting.linear_trend_forecast(monthly, 'contract_value', forecast_periods=periods)
            future = pd.date_range(monthly.award_date.max() + pd.offsets.MonthBegin(1), periods=periods, freq='MS')
            st.caption('A linear extrapolation of monthly totals. This is not a validated budget forecast.')
            st.dataframe(pd.DataFrame({'month': future, 'projected_value': forecast['forecast']}), width='stretch')

    elif page == "Settings":
        st.markdown("### <div class='sub-header'>⚙️ Settings</div>",
                   unsafe_allow_html=True)

        st.markdown("**Anomaly Detection Sensitivity**")
        options = st.session_state.analysis_settings
        st.slider('Z-score threshold:', 1.0, 5.0, options['zscore_threshold'], 0.1,
            key='setting_zscore_threshold', on_change=save_setting, args=('zscore_threshold',))
        st.slider('Expected outlier share:', 0.01, 0.30, options['contamination'], 0.01,
            key='setting_contamination', on_change=save_setting, args=('contamination',))
        st.markdown('**Visualization Options**')
        st.checkbox('Show grid lines', value=options['show_grid'], key='setting_show_grid',
            on_change=save_setting, args=('show_grid',))
        st.checkbox('Show legend', value=options['show_legend'], key='setting_show_legend',
            on_change=save_setting, args=('show_legend',))

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

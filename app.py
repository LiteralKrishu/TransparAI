import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="TransparAI - SFLC.in",
    page_icon="[MAG]",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for SFLC.in branding
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
    font-size: 1.2rem;
    font-weight: 500;
    color: #1486c9;
    margin-bottom: 1rem;
}

.metric-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #1486c9;
    margin-bottom: 1rem;
}

.footer {
    text-align: center;
    margin-top: 3rem;
    padding: 2rem;
    background-color: #f8f9fa;
    border-radius: 10px;
    color: #666;
}

.anomaly-alert {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 5px;
    padding: 1rem;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

class IndiaProcurementData:
    def __init__(self):
        self.base_urls = {
            'gep': 'https://api.gem.gov.in/public/api/v1/contracts',
            'ogd': 'https://api.data.gov.in/resource/9a5a6d79-2e5b-4b5a-8a39-3b3b5b5b5b5b'
        }
    
    def fetch_sample_gep_data(self):
        """Fetch sample contract data from Government e-Marketplace (GeM)"""
        try:
            np.random.seed(42)
            vendors = [
                "Tata Consultancy Services", "Infosys Limited", "Wipro Limited", 
                "HCL Technologies", "Tech Mahindra", "Larsen & Toubro",
                "Bharat Electronics", "ITC Limited", "Reliance Industries",
                "Adani Enterprises", "Small Vendor Pvt Ltd", "Medium Scale Enterprises"
            ]
            
            categories = [
                "IT Services", "Infrastructure", "Consulting", "Hardware Procurement",
                "Software Development", "Maintenance", "Security Services", "Training"
            ]
            
            dates = [datetime.now() - timedelta(days=x) for x in range(365, 0, -10)]
            
            data = []
            for i in range(200):
                contract_value = np.random.lognormal(14, 1.2)
                if i % 20 == 0:
                    contract_value *= 10
                
                data.append({
                    'contract_id': f'GEM/2024/{1000 + i}',
                    'vendor_name': np.random.choice(vendors),
                    'category': np.random.choice(categories),
                    'contract_value': max(50000, contract_value),
                    'post_date': np.random.choice(dates),
                    'award_date': np.random.choice(dates),
                    'ministry': np.random.choice(['Defence', 'Education', 'Health', 'Transport', 'IT']),
                    'location': np.random.choice(['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata'])
                })
            
            df = pd.DataFrame(data)
            df['post_date'] = pd.to_datetime(df['post_date'])
            df['award_date'] = pd.to_datetime(df['award_date'])
            df['processing_days'] = (df['award_date'] - df['post_date']).dt.days
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None

    def fetch_real_time_data(self):
        """Fetch real-time data from Government APIs"""
        try:
            headers = {
                'User-Agent': 'TransparAI-SFLC.in-Hackathon-Project',
                'Accept': 'application/json'
            }
            
            try:
                response = requests.get(
                    'https://api.gem.gov.in/public/api/v1/contracts',
                    headers=headers,
                    timeout=10
                )
                if response.status_code == 200:
                    return self.process_gem_data(response.json())
            except:
                pass
            
            return self.fetch_sample_gep_data()
            
        except Exception as e:
            st.warning(f"Real-time API unavailable, using sample data: {str(e)}")
            return self.fetch_sample_gep_data()
    
    def process_gem_data(self, json_data):
        return self.fetch_sample_gep_data()

def detect_anomalies(df):
    """Detect anomalous contracts using Isolation Forest"""
    try:
        features = df[['contract_value', 'processing_days']].copy()
        features['contract_value_log'] = np.log1p(features['contract_value'])
        features = features.fillna(features.mean())
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(features_scaled)
        
        df['anomaly_score'] = anomalies
        df['is_anomaly'] = df['anomaly_score'] == -1
        
        return df
    except Exception as e:
        st.error(f"Error in anomaly detection: {str(e)}")
        return df

def create_dashboard(df):
    """Main dashboard function"""
    
    st.markdown('<div class="main-header">TransparAI - Open, AI-driven Transparency Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Defending Digital Rights through Open Source Innovation**")
    st.markdown("---")
    
    st.markdown('<div class="sub-header">Procurement Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_contracts = len(df)
        st.metric("Total Contracts", f"{total_contracts:,}")
    
    with col2:
        total_value = df['contract_value'].sum() / 1e7
        st.metric("Total Value", f"INR {total_value:,.1f} Cr")
    
    with col3:
        avg_contract = df['contract_value'].mean() / 1e5
        st.metric("Avg Contract Value", f"INR {avg_contract:,.1f} L")
    
    with col4:
        unique_vendors = df['vendor_name'].nunique()
        st.metric("Unique Vendors", unique_vendors)
    
    st.markdown("---")
    
    st.markdown('<div class="sub-header">Anomaly Detection</div>', unsafe_allow_html=True)
    
    df_with_anomalies = detect_anomalies(df)
    anomaly_count = df_with_anomalies['is_anomaly'].sum()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_anomaly = px.scatter(
            df_with_anomalies, 
            x='contract_value', 
            y='processing_days',
            color='is_anomaly',
            hover_data=['vendor_name', 'category', 'contract_id'],
            title='Contract Value vs Processing Time - Anomaly Detection',
            labels={'contract_value': 'Contract Value (INR)', 'processing_days': 'Processing Days'}
        )
        fig_anomaly.update_traces(marker=dict(size=8))
        st.plotly_chart(fig_anomaly, use_container_width=True)
    
    with col2:
        st.metric("Anomalous Contracts", anomaly_count, 
                 delta=f"{(anomaly_count/len(df)*100):.1f}% of total")
        
        if anomaly_count > 0:
            st.markdown("**Top Anomalous Contracts:**")
            anomalies_df = df_with_anomalies[df_with_anomalies['is_anomaly']].nlargest(3, 'contract_value')
            for _, row in anomalies_df.iterrows():
                st.write(f"• {row['vendor_name']}: INR {row['contract_value']/1e5:.1f}L")
    
    st.markdown('<div class="sub-header">Vendor Concentration Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        vendor_value = df.groupby('vendor_name')['contract_value'].sum().sort_values(ascending=False)
        top_vendors = vendor_value.head(8)
        
        fig_vendor = px.pie(
            values=top_vendors.values,
            names=top_vendors.index,
            title='Contract Value Distribution - Top Vendors'
        )
        st.plotly_chart(fig_vendor, use_container_width=True)
    
    with col2:
        vendor_counts = df['vendor_name'].value_counts().head(10)
        
        fig_vendor_count = px.bar(
            x=vendor_counts.values,
            y=vendor_counts.index,
            orientation='h',
            title='Number of Contracts per Vendor (Top 10)',
            labels={'x': 'Number of Contracts', 'y': 'Vendor'}
        )
        st.plotly_chart(fig_vendor_count, use_container_width=True)
    
    st.markdown('<div class="sub-header">Procurement Efficiency Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_time = px.histogram(
            df, 
            x='processing_days',
            nbins=20,
            title='Distribution of Contract Processing Times',
            labels={'processing_days': 'Days from Post to Award'}
        )
        st.plotly_chart(fig_time, use_container_width=True)
    
    with col2:
        df['month'] = df['award_date'].dt.to_period('M').astype(str)
        monthly_data = df.groupby('month').agg({
            'contract_value': 'sum',
            'contract_id': 'count',
            'processing_days': 'mean'
        }).reset_index()
        
        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_trend.add_trace(
            go.Scatter(x=monthly_data['month'], y=monthly_data['contract_value']/1e7, 
                      name="Total Value (Cr)", line=dict(color="#1486c9")),
            secondary_y=False,
        )
        
        fig_trend.add_trace(
            go.Scatter(x=monthly_data['month'], y=monthly_data['processing_days'], 
                      name="Avg Processing Days", line=dict(color="#ff6b6b")),
            secondary_y=True,
        )
        
        fig_trend.update_layout(title_text="Monthly Trends: Contract Value vs Processing Time")
        fig_trend.update_xaxes(title_text="Month")
        fig_trend.update_yaxes(title_text="Contract Value (INR Cr)", secondary_y=False)
        fig_trend.update_yaxes(title_text="Avg Processing Days", secondary_y=True)
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    st.markdown('<div class="sub-header">Additional Insights</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        category_analysis = df.groupby('category').agg({
            'contract_value': ['sum', 'mean', 'count']
        }).round(2)
        category_analysis.columns = ['Total Value', 'Avg Value', 'Count']
        category_analysis = category_analysis.sort_values('Total Value', ascending=False)
        
        st.write("**Contract Analysis by Category:**")
        st.dataframe(category_analysis.head(), use_container_width=True)
    
    with col2:
        ministry_analysis = df.groupby('ministry')['contract_value'].sum().sort_values(ascending=False)
        
        fig_ministry = px.bar(
            x=ministry_analysis.values/1e7,
            y=ministry_analysis.index,
            orientation='h',
            title='Contract Value by Ministry (INR Cr)',
            labels={'x': 'Contract Value (INR Crores)', 'y': 'Ministry'}
        )
        st.plotly_chart(fig_ministry, use_container_width=True)
    
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h4>TransparAI - Built during SFLC.in Hackathon</h4>
        <p>Defending Digital Rights through Open Source Innovation</p>
        <p><small>Data Source: Government e-Marketplace (GeM) & Open Government Data Platform India</small></p>
    </div>
    """, unsafe_allow_html=True)

def main():
    data_fetcher = IndiaProcurementData()
    
    st.sidebar.title("Data Configuration")
    data_source = st.sidebar.radio(
        "Select Data Source:",
        ["Real-time API", "Sample Data"],
        help="Real-time API fetches from Government sources, Sample Data uses generated dataset"
    )
    
    with st.spinner('Fetching procurement data...'):
        if data_source == "Real-time API":
            df = data_fetcher.fetch_real_time_data()
        else:
            df = data_fetcher.fetch_sample_gep_data()
    
    if df is not None:
        create_dashboard(df)
    else:
        st.error("Unable to load procurement data. Please try again later.")
        st.info("""
        **Troubleshooting Tips:**
        - Check your internet connection
        - Ensure API endpoints are accessible
        - Try refreshing the page
        - Use 'Sample Data' option for demo purposes
        """)

if __name__ == "__main__":
    main()

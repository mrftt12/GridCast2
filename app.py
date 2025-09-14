import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="GridCast Utility IRP",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'feeder_config' not in st.session_state:
    st.session_state.feeder_config = {
        'name': 'Feeder_001',
        'base_voltage': 12.47,
        'rated_capacity': 15.0,
        'length_km': 5.2,
        'customers': 850,
        'base_load_mw': 8.5,
        'dg_capacity_mw': 2.1,
        'ev_penetration': 0.15,
        'temp_sensitivity': 0.02
    }

if 'load_profile' not in st.session_state:
    # Initialize with realistic base load profile
    hours = np.arange(8760)
    base_profile = np.array([
        0.4 + 0.3 * np.sin(2 * np.pi * (h % 24 - 6) / 24) + 
        0.2 * np.sin(2 * np.pi * h / (24 * 7)) +
        0.1 * np.sin(2 * np.pi * h / (24 * 365.25)) +
        np.random.normal(0, 0.05)
        for h in hours
    ])
    base_profile = np.clip(base_profile, 0.2, 1.0)
    st.session_state.load_profile = base_profile

if 'forecast_scenarios' not in st.session_state:
    st.session_state.forecast_scenarios = {
        'Low EV Adoption': {'ev_growth_rate': 0.05, 'load_growth_rate': 0.015},
        'Medium EV Adoption': {'ev_growth_rate': 0.15, 'load_growth_rate': 0.025},
        'High EV Adoption': {'ev_growth_rate': 0.30, 'load_growth_rate': 0.035}
    }

if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None

if 'network_model' not in st.session_state:
    st.session_state.network_model = None

if 'loadflow_results' not in st.session_state:
    st.session_state.loadflow_results = None

# Main page content
st.title("⚡ GridCast Utility IRP")
st.markdown("---")

# Introduction and overview
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🏗️ Feeder Configuration")
    st.write("Define feeder parameters, base load characteristics, distributed generation, and EV charging patterns.")
    if st.button("Configure Feeder", use_container_width=True):
        st.switch_page("pages/1_Feeder_Configuration.py")

with col2:
    st.subheader("📈 Load Forecasting")
    st.write("Generate 10-year load forecasts based on different EV adoption and load growth scenarios.")
    if st.button("Create Forecast", use_container_width=True):
        st.switch_page("pages/2_Load_Forecasting.py")

with col3:
    st.subheader("📊 Load Profile Editor")
    st.write("Visualize and edit 8760 hourly load profiles with interactive charts and analysis.")
    if st.button("Edit Profiles", use_container_width=True):
        st.switch_page("pages/3_Load_Profile_Editor.py")

st.markdown("---")

col4, col5, col6 = st.columns(3)

with col4:
    st.subheader("🔌 Load Flow Analysis")
    st.write("Run time series load flow analysis using PandaPower for detailed network studies.")
    if st.button("Analyze Load Flow", use_container_width=True):
        st.switch_page("pages/4_Load_Flow_Analysis.py")

with col5:
    st.subheader("📋 Utility Dashboard")
    st.write("Professional dashboard with load growth projections, peak demand analysis, and capacity planning.")
    if st.button("View Dashboard", use_container_width=True):
        st.switch_page("pages/5_Dashboard.py")

with col6:
    st.subheader("💾 Export Data")
    st.write("Export forecast results and load flow analysis data in standard utility formats.")
    st.download_button(
        "Export Results",
        data="Export functionality available in Dashboard",
        file_name="utility_analysis.txt",
        use_container_width=True,
        disabled=True
    )

# Current system status
st.markdown("---")
st.subheader("Current System Status")

status_col1, status_col2, status_col3, status_col4 = st.columns(4)

with status_col1:
    st.metric("Active Feeder", st.session_state.feeder_config['name'])

with status_col2:
    st.metric("Base Load (MW)", f"{st.session_state.feeder_config['base_load_mw']:.1f}")

with status_col3:
    forecast_status = "Complete" if st.session_state.forecast_results is not None else "Pending"
    st.metric("Forecast Status", forecast_status)

with status_col4:
    loadflow_status = "Complete" if st.session_state.loadflow_results is not None else "Pending"
    st.metric("Load Flow Status", loadflow_status)

# Quick system overview
if st.session_state.forecast_results is not None:
    st.markdown("### Recent Forecast Summary")
    
    # Display key forecast metrics
    forecast_data = st.session_state.forecast_results
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        peak_demand = forecast_data['peak_demands'][-1] if 'peak_demands' in forecast_data else 0
        st.metric("10-Year Peak Demand (MW)", f"{peak_demand:.2f}")
    
    with summary_col2:
        growth_rate = forecast_data['scenario'].get('load_growth_rate', 0) * 100
        st.metric("Annual Growth Rate (%)", f"{growth_rate:.1f}")
    
    with summary_col3:
        ev_rate = forecast_data['scenario'].get('ev_growth_rate', 0) * 100
        st.metric("EV Growth Rate (%)", f"{ev_rate:.1f}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Electric Utility Integrated Resource Planning System<br>
        Powered by Streamlit, PandaPower, and Advanced Analytics
    </div>
    """,
    unsafe_allow_html=True
)

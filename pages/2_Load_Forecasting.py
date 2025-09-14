import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.load_forecasting import LoadForecaster
from utils.load_profiles import LoadProfileGenerator
import datetime

st.set_page_config(page_title="Load Forecasting", page_icon="📈", layout="wide")

st.title("📈 Load Forecasting")
st.markdown("Generate 10-year load forecasts based on different EV adoption and load growth scenarios.")

# Initialize forecaster
forecaster = LoadForecaster()
profile_generator = LoadProfileGenerator()

# Scenario Configuration
st.header("Forecast Scenario Configuration")

scenario_col1, scenario_col2 = st.columns(2)

with scenario_col1:
    st.subheader("Load Growth Parameters")
    base_load_growth = st.slider("Base Load Growth Rate (%/year)", 0.0, 8.0, 2.5, step=0.1) / 100
    peak_growth_multiplier = st.slider("Peak Growth Multiplier", 1.0, 2.0, 1.3, step=0.1)
    economic_factor = st.slider("Economic Growth Factor", 0.5, 2.0, 1.0, step=0.1)
    
    st.subheader("Temperature Impact")
    climate_change_factor = st.slider("Climate Change Impact (%/year)", 0.0, 2.0, 0.3, step=0.1) / 100
    cooling_load_sensitivity = st.slider("Cooling Load Sensitivity", 0.5, 3.0, 1.2, step=0.1)

with scenario_col2:
    st.subheader("EV Adoption Scenarios")
    
    # Predefined scenarios with custom option
    scenario_type = st.selectbox(
        "Select Scenario Type",
        ["Low EV Adoption", "Medium EV Adoption", "High EV Adoption", "Custom"]
    )
    
    if scenario_type == "Custom":
        ev_growth_rate = st.slider("EV Growth Rate (%/year)", 0.0, 50.0, 15.0, step=1.0) / 100
        max_ev_penetration = st.slider("Maximum EV Penetration (%)", 10.0, 100.0, 60.0, step=5.0) / 100
        smart_charging_adoption = st.slider("Smart Charging Adoption (%)", 0.0, 100.0, 30.0, step=5.0) / 100
    else:
        # Use predefined scenarios
        scenario_params = st.session_state.forecast_scenarios[scenario_type]
        ev_growth_rate = scenario_params['ev_growth_rate']
        max_ev_penetration = {
            'Low EV Adoption': 0.25,
            'Medium EV Adoption': 0.60,
            'High EV Adoption': 0.85
        }[scenario_type]
        smart_charging_adoption = {
            'Low EV Adoption': 0.20,
            'Medium EV Adoption': 0.40,
            'High EV Adoption': 0.70
        }[scenario_type]
    
    st.metric("EV Growth Rate", f"{ev_growth_rate*100:.1f}%/year")
    st.metric("Max EV Penetration", f"{max_ev_penetration*100:.0f}%")
    st.metric("Smart Charging", f"{smart_charging_adoption*100:.0f}%")

st.markdown("---")

# Advanced Forecasting Parameters
st.header("Advanced Forecasting Parameters")

advanced_col1, advanced_col2, advanced_col3 = st.columns(3)

with advanced_col1:
    st.subheader("Distributed Generation")
    solar_growth_rate = st.slider("Solar Growth Rate (%/year)", 0.0, 30.0, 12.0, step=1.0) / 100
    storage_adoption_rate = st.slider("Storage Adoption Rate (%/year)", 0.0, 20.0, 5.0, step=1.0) / 100
    net_metering_impact = st.slider("Net Metering Impact", 0.7, 1.0, 0.85, step=0.05)

with advanced_col2:
    st.subheader("Energy Efficiency")
    efficiency_improvement = st.slider("Efficiency Improvement (%/year)", 0.0, 5.0, 1.5, step=0.1) / 100
    appliance_saturation = st.slider("Appliance Saturation Factor", 0.8, 1.2, 0.95, step=0.05)
    building_code_impact = st.slider("Building Code Impact", 0.0, 10.0, 2.0, step=0.5) / 100

with advanced_col3:
    st.subheader("Load Shape Changes")
    electrification_rate = st.slider("Electrification Rate (%/year)", 0.0, 10.0, 3.0, step=0.5) / 100
    demand_response_penetration = st.slider("Demand Response (%)", 0.0, 50.0, 15.0, step=5.0) / 100
    time_of_use_adoption = st.slider("Time-of-Use Adoption (%)", 0.0, 100.0, 25.0, step=5.0) / 100

# Generate Forecast Button
st.markdown("---")
forecast_col1, forecast_col2 = st.columns([3, 1])

with forecast_col1:
    st.subheader("Generate 10-Year Forecast")
    st.write("Click the button below to generate a detailed 10-year load forecast based on your selected parameters.")

with forecast_col2:
    if st.button("Generate Forecast", type="primary", use_container_width=True):
        with st.spinner("Generating 10-year forecast..."):
            # Compile forecast parameters
            forecast_params = {
                'scenario_name': scenario_type,
                'base_load_growth': base_load_growth,
                'ev_growth_rate': ev_growth_rate,
                'max_ev_penetration': max_ev_penetration,
                'smart_charging_adoption': smart_charging_adoption,
                'solar_growth_rate': solar_growth_rate,
                'efficiency_improvement': efficiency_improvement,
                'electrification_rate': electrification_rate,
                'climate_change_factor': climate_change_factor,
                'peak_growth_multiplier': peak_growth_multiplier,
                'economic_factor': economic_factor,
                'storage_adoption_rate': storage_adoption_rate,
                'demand_response_penetration': demand_response_penetration
            }
            
            # Generate forecast
            forecast_results = forecaster.generate_forecast(
                st.session_state.feeder_config,
                st.session_state.load_profile,
                forecast_params
            )
            
            # Store results in session state
            st.session_state.forecast_results = forecast_results
            
            st.success("Forecast generated successfully!")
            st.rerun()

# Display Forecast Results
if st.session_state.forecast_results is not None:
    st.markdown("---")
    st.header("Forecast Results")
    
    results = st.session_state.forecast_results
    years = results['years']
    
    # Key Metrics
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        current_peak = results['peak_demands'][0]
        future_peak = results['peak_demands'][-1]
        peak_growth = ((future_peak / current_peak) - 1) * 100
        st.metric("Peak Demand Growth", f"{peak_growth:.1f}%", delta=f"{future_peak:.1f} MW")
    
    with metrics_col2:
        current_energy = results['annual_energy'][0]
        future_energy = results['annual_energy'][-1]
        energy_growth = ((future_energy / current_energy) - 1) * 100
        st.metric("Energy Growth", f"{energy_growth:.1f}%", delta=f"{future_energy:.0f} MWh")
    
    with metrics_col3:
        final_ev_penetration = results['ev_penetration'][-1] * 100
        st.metric("Final EV Penetration", f"{final_ev_penetration:.1f}%")
    
    with metrics_col4:
        capacity_utilization = (future_peak / st.session_state.feeder_config['rated_capacity']) * 100
        st.metric("Capacity Utilization", f"{capacity_utilization:.1f}%")
    
    # Forecast Charts
    st.subheader("Load Growth Projections")
    
    # Create comprehensive forecast visualization
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Peak Demand Forecast',
            'Annual Energy Forecast', 
            'EV Adoption Curve',
            'Load Factor Trends'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Peak Demand
    fig.add_trace(
        go.Scatter(
            x=years, 
            y=results['peak_demands'],
            mode='lines+markers',
            name='Peak Demand',
            line=dict(color='red', width=3)
        ),
        row=1, col=1
    )
    
    # Add capacity line
    capacity_line = [st.session_state.feeder_config['rated_capacity']] * len(years)
    fig.add_trace(
        go.Scatter(
            x=years,
            y=capacity_line,
            mode='lines',
            name='Rated Capacity',
            line=dict(color='orange', dash='dash', width=2)
        ),
        row=1, col=1
    )
    
    # Annual Energy
    fig.add_trace(
        go.Scatter(
            x=years,
            y=results['annual_energy'],
            mode='lines+markers',
            name='Annual Energy',
            line=dict(color='blue', width=3)
        ),
        row=1, col=2
    )
    
    # EV Adoption
    fig.add_trace(
        go.Scatter(
            x=years,
            y=[pct * 100 for pct in results['ev_penetration']],
            mode='lines+markers',
            name='EV Penetration',
            line=dict(color='green', width=3)
        ),
        row=2, col=1
    )
    
    # Load Factor
    load_factors = [energy/(peak*8760) for energy, peak in zip(results['annual_energy'], results['peak_demands'])]
    fig.add_trace(
        go.Scatter(
            x=years,
            y=[lf * 100 for lf in load_factors],
            mode='lines+markers',
            name='Load Factor',
            line=dict(color='purple', width=3)
        ),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=1, col=2)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=2)
    
    fig.update_yaxes(title_text="MW", row=1, col=1)
    fig.update_yaxes(title_text="MWh", row=1, col=2)
    fig.update_yaxes(title_text="%", row=2, col=1)
    fig.update_yaxes(title_text="%", row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Annual Breakdown
    st.subheader("Annual Forecast Breakdown")
    
    # Create detailed results table
    forecast_df = pd.DataFrame({
        'Year': years,
        'Peak Demand (MW)': [f"{p:.2f}" for p in results['peak_demands']],
        'Annual Energy (MWh)': [f"{e:.0f}" for e in results['annual_energy']],
        'EV Penetration (%)': [f"{p*100:.1f}" for p in results['ev_penetration']],
        'Solar Capacity (MW)': [f"{s:.2f}" for s in results.get('solar_capacity', [0]*len(years))],
        'Load Factor (%)': [f"{lf*100:.1f}" for lf in load_factors],
        'Capacity Utilization (%)': [f"{(p/st.session_state.feeder_config['rated_capacity'])*100:.1f}" for p in results['peak_demands']]
    })
    
    st.dataframe(forecast_df, use_container_width=True)
    
    # Capacity Planning Insights
    st.subheader("Capacity Planning Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("**Key Findings:**")
        
        # Check if capacity will be exceeded
        max_utilization = max([(p/st.session_state.feeder_config['rated_capacity']) for p in results['peak_demands']])
        if max_utilization > 0.8:
            st.error(f"⚠️ Capacity concern: Peak utilization reaches {max_utilization*100:.1f}%")
            years_to_80_pct = next((y for y, p in zip(years, results['peak_demands']) 
                                  if p/st.session_state.feeder_config['rated_capacity'] > 0.8), None)
            if years_to_80_pct:
                st.warning(f"Capacity upgrade needed by {years_to_80_pct}")
        else:
            st.success("✅ Adequate capacity for forecast period")
        
        # EV impact analysis
        ev_load_impact = (results['peak_demands'][-1] - results['peak_demands'][0]) * 0.6  # Assume 60% of growth from EVs
        st.info(f"📊 EV contribution to peak growth: ~{ev_load_impact:.1f} MW")
        
        # Load factor trend
        initial_lf = load_factors[0]
        final_lf = load_factors[-1]
        lf_change = ((final_lf / initial_lf) - 1) * 100
        if lf_change > 0:
            st.success(f"📈 Load factor improvement: +{lf_change:.1f}%")
        else:
            st.warning(f"📉 Load factor degradation: {lf_change:.1f}%")
    
    with insight_col2:
        st.markdown("**Recommendations:**")
        
        # Generate recommendations based on results
        recommendations = []
        
        if max_utilization > 0.9:
            recommendations.append("🔧 Immediate capacity upgrade planning required")
        elif max_utilization > 0.8:
            recommendations.append("📋 Monitor loading and plan capacity expansion")
        
        if results['ev_penetration'][-1] > 0.3:
            recommendations.append("🚗 Consider EV-specific infrastructure upgrades")
            recommendations.append("⚡ Implement smart charging programs")
        
        if final_lf < initial_lf:
            recommendations.append("📊 Develop demand response programs")
            recommendations.append("💰 Consider time-of-use rate structures")
        
        if len(recommendations) == 0:
            recommendations.append("✅ Current infrastructure adequate")
            recommendations.append("📈 Continue monitoring growth trends")
        
        for rec in recommendations:
            st.write(rec)

# Navigation
st.markdown("---")
nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    if st.button("← Back to Configuration", use_container_width=True):
        st.switch_page("pages/1_Feeder_Configuration.py")

with nav_col2:
    if st.button("Edit Load Profiles →", use_container_width=True):
        st.switch_page("pages/3_Load_Profile_Editor.py")

with nav_col3:
    if st.button("Run Load Flow Analysis →", use_container_width=True):
        st.switch_page("pages/4_Load_Flow_Analysis.py")

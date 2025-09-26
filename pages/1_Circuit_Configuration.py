import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Circuit Configuration", page_icon="🏗️", layout="wide")

st.title("🏗️ Circuit Configuration")
st.markdown("Configure circuit parameters, base load characteristics, distributed generation, and EV charging patterns.")

# Circuit Basic Parameters
st.header("Basic Circuit Parameters")
col1, col2 = st.columns(2)

with col1:
    circuit_name = st.text_input(
        "Circuit Name", 
        value=st.session_state.feeder_config['name']
    )
    base_voltage = st.number_input(
        "Base Voltage (kV)", 
        min_value=1.0, 
        max_value=50.0, 
        value=st.session_state.feeder_config['base_voltage'],
        step=0.1
    )
    rated_capacity = st.number_input(
        "Rated Capacity (MVA)", 
        min_value=1.0, 
        max_value=100.0, 
        value=st.session_state.feeder_config['rated_capacity'],
        step=0.1
    )

with col2:
    length_km = st.number_input(
        "Feeder Length (km)", 
        min_value=0.1, 
        max_value=50.0, 
        value=st.session_state.feeder_config['length_km'],
        step=0.1
    )
    customers = st.number_input(
        "Number of Customers", 
        min_value=1, 
        max_value=5000, 
        value=st.session_state.feeder_config['customers']
    )
    base_load_mw = st.number_input(
        "Base Load (MW)", 
        min_value=0.1, 
        max_value=50.0, 
        value=st.session_state.feeder_config['base_load_mw'],
        step=0.1
    )

st.markdown("---")

# Load Characteristics
st.header("Load Characteristics")
load_col1, load_col2 = st.columns(2)

with load_col1:
    st.subheader("Load Profile Parameters")
    residential_pct = st.slider("Residential Load (%)", 0, 100, 65)
    commercial_pct = st.slider("Commercial Load (%)", 0, 100, 25)
    industrial_pct = st.slider("Industrial Load (%)", 0, 100, 10)
    
    # Ensure percentages add up to 100
    total_pct = residential_pct + commercial_pct + industrial_pct
    if total_pct != 100:
        st.warning(f"Load percentages total {total_pct}%. Adjusting to 100%.")
        if total_pct > 0:
            residential_pct = int(residential_pct * 100 / total_pct)
            commercial_pct = int(commercial_pct * 100 / total_pct)
            industrial_pct = 100 - residential_pct - commercial_pct

with load_col2:
    st.subheader("Temperature Sensitivity")
    temp_sensitivity = st.slider(
        "Temperature Sensitivity (MW/°C)", 
        0.0, 0.1, 
        st.session_state.feeder_config['temp_sensitivity'],
        step=0.001,
        format="%.3f"
    )
    
    heating_threshold = st.number_input("Heating Threshold (°C)", value=15.0)
    cooling_threshold = st.number_input("Cooling Threshold (°C)", value=22.0)
    
    st.subheader("Peak Load Factors")
    summer_peak_factor = st.slider("Summer Peak Factor", 0.8, 1.5, 1.2)
    winter_peak_factor = st.slider("Winter Peak Factor", 0.8, 1.5, 0.9)

st.markdown("---")

# Distributed Generation Configuration
st.header("Distributed Generation Configuration")
dg_col1, dg_col2 = st.columns(2)

with dg_col1:
    st.subheader("Solar PV Configuration")
    solar_capacity = st.number_input(
        "Solar PV Capacity (MW)", 
        min_value=0.0, 
        max_value=10.0, 
        value=st.session_state.feeder_config['dg_capacity_mw'],
        step=0.1
    )
    solar_penetration = st.slider("Solar Penetration (%)", 0, 100, 25)
    tilt_angle = st.slider("Panel Tilt Angle (degrees)", 0, 90, 30)
    azimuth_angle = st.slider("Azimuth Angle (degrees)", 0, 360, 180)

with dg_col2:
    st.subheader("Other DG Sources")
    wind_capacity = st.number_input("Wind Capacity (MW)", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
    battery_capacity = st.number_input("Battery Storage (MWh)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
    battery_power = st.number_input("Battery Power (MW)", min_value=0.0, max_value=5.0, value=0.0, step=0.1)

# Generate sample DG profile visualization
if solar_capacity > 0:
    st.subheader("Sample Solar Generation Profile")
    
    # Generate typical solar profile for a day
    hours = np.arange(24)
    solar_profile = np.array([
        max(0, solar_capacity * np.sin(np.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0 
        for h in hours
    ])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, 
        y=solar_profile, 
        mode='lines+markers',
        name='Solar Generation',
        line=dict(color='orange', width=3)
    ))
    fig.update_layout(
        title='Typical Daily Solar Generation Profile',
        xaxis_title='Hour of Day',
        yaxis_title='Generation (MW)',
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# EV Charging Configuration
st.header("Electric Vehicle Charging Configuration")
ev_col1, ev_col2 = st.columns(2)

with ev_col1:
    st.subheader("EV Adoption Parameters")
    current_ev_penetration = st.slider(
        "Current EV Penetration (%)", 
        0.0, 50.0, 
        st.session_state.feeder_config['ev_penetration'] * 100,
        step=0.5
    ) / 100
    
    ev_growth_rate = st.slider("Annual EV Growth Rate (%)", 0.0, 50.0, 15.0) / 100
    avg_ev_power = st.number_input("Average EV Charging Power (kW)", value=7.2)
    charging_efficiency = st.slider("Charging Efficiency (%)", 80, 100, 90) / 100

with ev_col2:
    st.subheader("Charging Patterns")
    home_charging_pct = st.slider("Home Charging (%)", 0, 100, 80)
    workplace_charging_pct = st.slider("Workplace Charging (%)", 0, 100, 15)
    public_charging_pct = st.slider("Public Charging (%)", 0, 100, 5)
    
    # Peak charging hours
    home_peak_start = st.selectbox("Home Charging Peak Start", range(24), index=18)
    home_peak_end = st.selectbox("Home Charging Peak End", range(24), index=22)

# Generate EV charging profile visualization
st.subheader("EV Charging Load Profile")

# Calculate EV load for each hour
hours = np.arange(24)
ev_load = np.zeros(24)

# Home charging pattern (evening peak)
for h in hours:
    if home_peak_start <= h <= home_peak_end:
        ev_load[h] += current_ev_penetration * customers * avg_ev_power / 1000 * home_charging_pct / 100 * 0.8
    elif 6 <= h <= 8:  # Morning charging
        ev_load[h] += current_ev_penetration * customers * avg_ev_power / 1000 * home_charging_pct / 100 * 0.3
    else:
        ev_load[h] += current_ev_penetration * customers * avg_ev_power / 1000 * home_charging_pct / 100 * 0.1

# Workplace charging (daytime)
for h in range(9, 17):
    ev_load[h] += current_ev_penetration * customers * avg_ev_power / 1000 * workplace_charging_pct / 100 * 0.6

# Public charging (distributed)
for h in hours:
    ev_load[h] += current_ev_penetration * customers * avg_ev_power / 1000 * public_charging_pct / 100 * 0.2

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hours, 
    y=ev_load, 
    mode='lines+markers',
    name='EV Charging Load',
    fill='tonexty',
    line=dict(color='green', width=3)
))
fig.update_layout(
    title='Daily EV Charging Load Profile',
    xaxis_title='Hour of Day',
    yaxis_title='Load (MW)',
    height=300
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Save Configuration
st.header("Save Configuration")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Save Configuration", type="primary", use_container_width=True):
        # Update session state with new configuration
        st.session_state.feeder_config.update({
            'name': feeder_name,
            'base_voltage': base_voltage,
            'rated_capacity': rated_capacity,
            'length_km': length_km,
            'customers': customers,
            'base_load_mw': base_load_mw,
            'dg_capacity_mw': solar_capacity,
            'ev_penetration': current_ev_penetration,
            'temp_sensitivity': temp_sensitivity,
            'load_mix': {
                'residential': residential_pct,
                'commercial': commercial_pct,
                'industrial': industrial_pct
            },
            'dg_config': {
                'solar_capacity': solar_capacity,
                'wind_capacity': wind_capacity,
                'battery_capacity': battery_capacity,
                'battery_power': battery_power,
                'solar_penetration': solar_penetration
            },
            'ev_config': {
                'penetration': current_ev_penetration,
                'growth_rate': ev_growth_rate,
                'avg_power': avg_ev_power,
                'charging_efficiency': charging_efficiency,
                'home_charging_pct': home_charging_pct,
                'workplace_charging_pct': workplace_charging_pct,
                'public_charging_pct': public_charging_pct
            }
        })
        st.success("Configuration saved successfully!")
        st.rerun()

with col2:
    if st.button("Reset to Defaults", use_container_width=True):
        # Reset to default configuration
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
        st.info("Configuration reset to defaults.")
        st.rerun()

with col3:
    if st.button("Go to Forecasting", use_container_width=True):
        st.switch_page("pages/2_Load_Forecasting.py")

# Configuration Summary
st.header("Configuration Summary")
config_summary = pd.DataFrame([
    {"Parameter": "Circuit Name", "Value": circuit_name},
    {"Parameter": "Base Voltage (kV)", "Value": f"{base_voltage:.1f}"},
    {"Parameter": "Rated Capacity (MVA)", "Value": f"{rated_capacity:.1f}"},
    {"Parameter": "Base Load (MW)", "Value": f"{base_load_mw:.1f}"},
    {"Parameter": "DG Capacity (MW)", "Value": f"{solar_capacity:.1f}"},
    {"Parameter": "EV Penetration (%)", "Value": f"{current_ev_penetration*100:.1f}"},
    {"Parameter": "Temperature Sensitivity (MW/°C)", "Value": f"{temp_sensitivity:.3f}"}
])

st.dataframe(config_summary, use_container_width=True)

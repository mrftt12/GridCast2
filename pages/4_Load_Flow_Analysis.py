import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.pandapower_integration import PowerFlowAnalyzer
import datetime

st.set_page_config(page_title="Load Flow Analysis", page_icon="🔌", layout="wide")

st.title("🔌 Load Flow Analysis")
st.markdown("Run time series load flow analysis using PandaPower for detailed network studies.")

# Initialize power flow analyzer
try:
    analyzer = PowerFlowAnalyzer()
except Exception as e:
    st.error(f"Error initializing PandaPower analyzer: {str(e)}")
    st.stop()

# Network Configuration
st.header("Network Model Configuration")

config_col1, config_col2 = st.columns(2)

with config_col1:
    st.subheader("Feeder Topology")
    
    network_type = st.selectbox(
        "Network Type",
        ["Radial Feeder", "Loop Feeder", "Networked System", "Custom Topology"]
    )
    
    num_buses = st.number_input("Number of Buses", min_value=3, max_value=50, value=12)
    num_lines = st.number_input("Number of Lines", min_value=2, max_value=100, value=11)
    
    # Transformer configuration
    transformer_config = st.checkbox("Include Substation Transformer", value=True)
    if transformer_config:
        hv_voltage = st.number_input("HV Side Voltage (kV)", value=69.0)
        lv_voltage = st.number_input("LV Side Voltage (kV)", value=st.session_state.feeder_config['base_voltage'])
        trafo_rating = st.number_input("Transformer Rating (MVA)", value=25.0)

with config_col2:
    st.subheader("Load Distribution")
    
    # Load allocation method
    load_allocation = st.selectbox(
        "Load Allocation Method",
        ["Uniform Distribution", "Distance-Based", "Customer-Based", "Manual Input"]
    )
    
    # DG placement
    dg_buses = st.multiselect(
        "DG Connection Buses",
        options=list(range(1, num_buses + 1)),
        default=[5, 8] if num_buses >= 8 else [2]
    )
    
    # Voltage regulation
    voltage_control = st.checkbox("Enable Voltage Regulation", value=True)
    if voltage_control:
        tap_changer = st.checkbox("Transformer Tap Changer", value=True)
        capacitor_banks = st.checkbox("Capacitor Banks", value=True)

# Network Creation and Validation
if st.button("Create Network Model", type="primary"):
    with st.spinner("Creating network model..."):
        try:
            print("Creating network model with the following parameters:")
            print(f"Network Type: {network_type}")
            print(f"Number of Buses: {num_buses}")
            print(f"Number of Lines: {num_lines}")
            print(f"Transformer Configuration: {transformer_config}")
            print(f"Load Allocation: {load_allocation}")
            print(f"Distributed Generation Buses: {dg_buses}")

            # Create the network based on configuration
            network = analyzer.create_network(
                network_type=network_type,
                num_buses=num_buses,
                num_lines=num_lines,
                feeder_config=st.session_state.feeder_config,
                transformer_config={
                    'enabled': transformer_config,
                    'hv_kv': hv_voltage if transformer_config else None,
                    'lv_kv': lv_voltage if transformer_config else None,
                    'rating_mva': trafo_rating if transformer_config else None
                },
                load_allocation=load_allocation,
                dg_buses=dg_buses
            )#FIXME:Error creating network model: Error creating network: Unknown standard trafo type 25 MVA 69/12.47 kV
            
            st.session_state.network_model = network
            st.success("Network model created successfully!")
            st.rerun()
            
        except Exception as e:
            print(f"Error creating network model: {str(e)}")
            st.error(f"Error creating network model: {str(e)}")

# Display Network Information
if st.session_state.network_model is not None:
    st.markdown("---")
    st.header("Network Model Overview")
    
    network = st.session_state.network_model
    
    # Network statistics
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        st.metric("Buses", len(network.bus))
    
    with stats_col2:
        st.metric("Lines", len(network.line))
    
    with stats_col3:
        st.metric("Loads", len(network.load))
    
    with stats_col4:
        st.metric("DG Units", len(network.sgen) if hasattr(network, 'sgen') and network.sgen is not None else 0)
    
    # Network topology visualization
    st.subheader("Network Topology")
    
    # Create a simple network diagram
    fig = go.Figure()
    
    # Add bus locations (simplified linear layout for radial feeder)
    bus_positions = {}
    for i, bus_idx in enumerate(network.bus.index):
        x = i * 2  # Simplified linear spacing
        y = 0 if i == 0 else np.random.uniform(-1, 1)  # Small vertical variation
        bus_positions[bus_idx] = (x, y)
        
        # Determine bus type for coloring
        if bus_idx == network.ext_grid.bus.iloc[0]:  # External grid connection
            color = 'red'
            symbol = 'square'
            size = 12
            name = f'Substation Bus {bus_idx}'
        elif bus_idx in dg_buses:  # DG buses
            color = 'green'
            symbol = 'diamond'
            size = 10
            name = f'DG Bus {bus_idx}'
        else:  # Load buses
            color = 'blue'
            symbol = 'circle'
            size = 8
            name = f'Load Bus {bus_idx}'
        
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(color=color, symbol=symbol, size=size),
            text=[str(bus_idx)],
            textposition='middle center',
            name=name,
            showlegend=False
        ))
    
    # Add lines
    for _, line in network.line.iterrows():
        from_bus = line['from_bus']
        to_bus = line['to_bus']
        
        if from_bus in bus_positions and to_bus in bus_positions:
            x_coords = [bus_positions[from_bus][0], bus_positions[to_bus][0]]
            y_coords = [bus_positions[from_bus][1], bus_positions[to_bus][1]]
            
            fig.add_trace(go.Scatter(
                x=x_coords, y=y_coords,
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        title='Network Topology Diagram',
        xaxis_title='Distance (arbitrary units)',
        yaxis_title='Lateral Position',
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Network component tables
    component_col1, component_col2 = st.columns(2)
    
    with component_col1:
        st.subheader("Bus Information")
        bus_info = network.bus.copy()
        bus_info['Type'] = bus_info.index.map(lambda x: 
            'Substation' if x == network.ext_grid.bus.iloc[0] else
            'DG Bus' if x in dg_buses else 'Load Bus'
        )
        st.dataframe(bus_info[['vn_kv', 'Type']], use_container_width=True)
    
    with component_col2:
        st.subheader("Load Information")
        if not network.load.empty:
            load_info = network.load[['bus', 'p_mw', 'q_mvar']].copy()
            load_info['Load Factor'] = load_info['p_mw'] / load_info['p_mw'].sum()
            st.dataframe(load_info, use_container_width=True)

st.markdown("---")

# Time Series Load Flow Configuration
st.header("Time Series Load Flow Configuration")

ts_col1, ts_col2 = st.columns(2)

with ts_col1:
    st.subheader("Analysis Settings")
    
    analysis_period = st.selectbox(
        "Analysis Period",
        ["Full Year (8760 hours)", "Peak Month", "Peak Week", "Peak Day", "Custom Range"]
    )
    
    if analysis_period == "Custom Range":
        start_hour = st.number_input("Start Hour", min_value=0, max_value=8759, value=0)
        end_hour = st.number_input("End Hour", min_value=1, max_value=8760, value=24)
        if end_hour <= start_hour:
            st.error("End hour must be greater than start hour")
    
    # Time series data source
    ts_data_source = st.selectbox(
        "Load Profile Source",
        ["Current Session Profile", "Forecast Results", "Upload Custom Data"]
    )
    
    # Analysis options
    include_voltage_analysis = st.checkbox("Include Voltage Analysis", value=True)
    include_thermal_analysis = st.checkbox("Include Thermal Analysis", value=True)
    include_losses_analysis = st.checkbox("Include Losses Analysis", value=True)

with ts_col2:
    st.subheader("Output Options")
    
    # Results storage
    store_all_results = st.checkbox("Store All Hourly Results", value=False)
    if not store_all_results:
        st.info("Only summary statistics will be stored to save memory")
    
    # Output variables
    output_variables = st.multiselect(
        "Output Variables",
        ["Bus Voltages", "Line Loadings", "Power Flows", "Losses", "DG Output"],
        default=["Bus Voltages", "Line Loadings"]
    )
    
    # Violation thresholds
    voltage_min = st.slider("Minimum Voltage (p.u.)", 0.90, 0.98, 0.95)
    voltage_max = st.slider("Maximum Voltage (p.u.)", 1.02, 1.10, 1.05)
    thermal_limit = st.slider("Thermal Loading Limit (%)", 80, 100, 90)

# Run Time Series Analysis
if st.session_state.network_model is not None:
    if st.button("Run Time Series Load Flow", type="primary"):
        with st.spinner("Running time series load flow analysis..."):
            try:
                # Determine analysis period
                if analysis_period == "Full Year (8760 hours)":
                    hours_to_analyze = list(range(8760))
                elif analysis_period == "Peak Month":
                    # Find peak month (month with highest average load)
                    profile = st.session_state.load_profile
                    monthly_avg = []
                    days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                    
                    for month in range(12):
                        start_day = sum(days_in_months[:month])
                        end_day = start_day + days_in_months[month]
                        start_hour = start_day * 24
                        end_hour = end_day * 24
                        monthly_avg.append(np.mean(profile[start_hour:end_hour]))
                    
                    peak_month = np.argmax(monthly_avg)
                    start_day = sum(days_in_months[:peak_month])
                    end_day = start_day + days_in_months[peak_month]
                    hours_to_analyze = list(range(start_day * 24, end_day * 24))
                
                elif analysis_period == "Peak Week":
                    # Find peak week
                    profile = st.session_state.load_profile
                    weekly_avg = []
                    for week in range(52):
                        start_hour = week * 168
                        end_hour = min(start_hour + 168, 8760)
                        weekly_avg.append(np.mean(profile[start_hour:end_hour]))
                    
                    peak_week = np.argmax(weekly_avg)
                    start_hour = peak_week * 168
                    end_hour = min(start_hour + 168, 8760)
                    hours_to_analyze = list(range(start_hour, end_hour))
                
                elif analysis_period == "Peak Day":
                    # Find peak day
                    profile = st.session_state.load_profile
                    daily_avg = []
                    for day in range(365):
                        start_hour = day * 24
                        end_hour = min(start_hour + 24, 8760)
                        daily_avg.append(np.mean(profile[start_hour:end_hour]))
                    
                    peak_day = np.argmax(daily_avg)
                    start_hour = peak_day * 24
                    end_hour = min(start_hour + 24, 8760)
                    hours_to_analyze = list(range(start_hour, end_hour))
                
                else:  # Custom Range
                    hours_to_analyze = list(range(start_hour, end_hour))
                
                # Get load profile data
                if ts_data_source == "Current Session Profile":
                    load_data = st.session_state.load_profile
                elif ts_data_source == "Forecast Results" and st.session_state.forecast_results is not None:
                    # Use the first year of forecast results
                    load_data = st.session_state.forecast_results.get('hourly_profiles', [st.session_state.load_profile])[0]
                else:
                    load_data = st.session_state.load_profile
                
                # Run time series analysis
                results = analyzer.run_time_series_analysis(
                    network=st.session_state.network_model,
                    load_profile=load_data,
                    hours_to_analyze=hours_to_analyze,
                    analysis_options={
                        'voltage_analysis': include_voltage_analysis,
                        'thermal_analysis': include_thermal_analysis,
                        'losses_analysis': include_losses_analysis,
                        'store_all_results': store_all_results,
                        'output_variables': output_variables,
                        'voltage_limits': (voltage_min, voltage_max),
                        'thermal_limit': thermal_limit / 100
                    }
                )
                
                st.session_state.loadflow_results = results
                st.success(f"Time series analysis completed! Analyzed {len(hours_to_analyze)} hours.")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error running time series analysis: {str(e)}")
                st.write("Debug info:", str(e))

# Display Results
if st.session_state.loadflow_results is not None:
    st.markdown("---")
    st.header("Load Flow Analysis Results")
    
    results = st.session_state.loadflow_results
    
    # Results summary
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        st.metric("Analysis Period", f"{len(results['hours_analyzed'])} hours")
    
    with summary_col2:
        if 'voltage_violations' in results:
            st.metric("Voltage Violations", results['voltage_violations'])
    
    with summary_col3:
        if 'thermal_violations' in results:
            st.metric("Thermal Violations", results['thermal_violations'])
    
    with summary_col4:
        if 'total_losses_mwh' in results:
            st.metric("Total Losses (MWh)", f"{results['total_losses_mwh']:.1f}")
    
    # Voltage Analysis Results
    if include_voltage_analysis and 'voltage_results' in results:
        st.subheader("Voltage Analysis")
        
        voltage_data = results['voltage_results']
        
        # Voltage profile chart
        fig = go.Figure()
        
        # Plot voltage for each bus
        for bus_id in voltage_data.keys():
            voltages = voltage_data[bus_id]
            hours = results['hours_analyzed'][:len(voltages)]
            
            fig.add_trace(go.Scatter(
                x=hours,
                y=voltages,
                mode='lines',
                name=f'Bus {bus_id}',
                line=dict(width=1)
            ))
        
        # Add voltage limit lines
        fig.add_hline(y=voltage_max, line_dash="dash", line_color="red", annotation_text="Max Limit")
        fig.add_hline(y=voltage_min, line_dash="dash", line_color="red", annotation_text="Min Limit")
        
        fig.update_layout(
            title='Bus Voltages Over Time',
            xaxis_title='Hour',
            yaxis_title='Voltage (p.u.)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Voltage statistics table
        voltage_stats = []
        for bus_id in voltage_data.keys():
            voltages = voltage_data[bus_id]
            voltage_stats.append({
                'Bus': bus_id,
                'Min Voltage (p.u.)': f"{np.min(voltages):.4f}",
                'Max Voltage (p.u.)': f"{np.max(voltages):.4f}",
                'Avg Voltage (p.u.)': f"{np.mean(voltages):.4f}",
                'Violations': np.sum((np.array(voltages) < voltage_min) | (np.array(voltages) > voltage_max))
            })
        
        voltage_df = pd.DataFrame(voltage_stats)
        st.dataframe(voltage_df, use_container_width=True)
    
    # Thermal Analysis Results
    if include_thermal_analysis and 'thermal_results' in results:
        st.subheader("Thermal Analysis")
        
        thermal_data = results['thermal_results']
        
        # Line loading chart
        fig = go.Figure()
        
        for line_id in thermal_data.keys():
            loadings = [l * 100 for l in thermal_data[line_id]]  # Convert to percentage
            hours = results['hours_analyzed'][:len(loadings)]
            
            fig.add_trace(go.Scatter(
                x=hours,
                y=loadings,
                mode='lines',
                name=f'Line {line_id}',
                line=dict(width=1)
            ))
        
        # Add thermal limit line
        fig.add_hline(y=thermal_limit, line_dash="dash", line_color="red", annotation_text="Thermal Limit")
        
        fig.update_layout(
            title='Line Thermal Loading Over Time',
            xaxis_title='Hour',
            yaxis_title='Loading (%)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Thermal statistics table
        thermal_stats = []
        for line_id in thermal_data.keys():
            loadings = thermal_data[line_id]
            thermal_stats.append({
                'Line': line_id,
                'Max Loading (%)': f"{np.max(loadings)*100:.1f}",
                'Avg Loading (%)': f"{np.mean(loadings)*100:.1f}",
                'Hours > 80%': np.sum(np.array(loadings) > 0.80),
                'Hours > 90%': np.sum(np.array(loadings) > 0.90),
                'Violations': np.sum(np.array(loadings) > thermal_limit/100)
            })
        
        thermal_df = pd.DataFrame(thermal_stats)
        st.dataframe(thermal_df, use_container_width=True)
    
    # Losses Analysis Results
    if include_losses_analysis and 'losses_results' in results:
        st.subheader("System Losses Analysis")
        
        losses_data = results['losses_results']
        hours = results['hours_analyzed']
        
        # Losses over time
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=hours,
            y=losses_data,
            mode='lines',
            name='System Losses',
            fill='tonexty',
            line=dict(color='orange', width=2)
        ))
        
        fig.update_layout(
            title='System Losses Over Time',
            xaxis_title='Hour',
            yaxis_title='Losses (MW)',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Losses statistics
        losses_col1, losses_col2, losses_col3 = st.columns(3)
        
        with losses_col1:
            st.metric("Total Losses (MWh)", f"{np.sum(losses_data):.1f}")
        
        with losses_col2:
            avg_loss_pct = (np.mean(losses_data) / st.session_state.feeder_config['base_load_mw']) * 100
            st.metric("Avg Loss Rate (%)", f"{avg_loss_pct:.2f}")
        
        with losses_col3:
            peak_losses = np.max(losses_data)
            st.metric("Peak Losses (MW)", f"{peak_losses:.2f}")

# Export Results
if st.session_state.loadflow_results is not None:
    st.markdown("---")
    st.header("Export Analysis Results")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        if st.button("Export Summary Report", use_container_width=True):
            results = st.session_state.loadflow_results
            
            report = f"""
TIME SERIES LOAD FLOW ANALYSIS REPORT
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

NETWORK INFORMATION:
- Network Type: {network_type}
- Number of Buses: {len(st.session_state.network_model.bus)}
- Number of Lines: {len(st.session_state.network_model.line)}
- Analysis Period: {len(results['hours_analyzed'])} hours

ANALYSIS RESULTS SUMMARY:
"""
            
            if 'voltage_violations' in results:
                report += f"- Voltage Violations: {results['voltage_violations']}\n"
            
            if 'thermal_violations' in results:
                report += f"- Thermal Violations: {results['thermal_violations']}\n"
            
            if 'total_losses_mwh' in results:
                report += f"- Total System Losses: {results['total_losses_mwh']:.1f} MWh\n"
            
            report += """
OPERATIONAL RECOMMENDATIONS:
- Review buses with voltage violations for voltage regulation equipment
- Monitor thermally loaded lines for capacity upgrades
- Consider distributed generation placement to reduce losses
- Implement demand response programs during peak loading periods
            """
            
            st.download_button(
                label="Download Report",
                data=report,
                file_name="loadflow_analysis_report.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    with export_col2:
        if st.button("Export Detailed Results", use_container_width=True):
            # Prepare detailed results for export
            detailed_results = {}
            results = st.session_state.loadflow_results
            
            # Add voltage results
            if 'voltage_results' in results:
                voltage_df_list = []
                for bus_id, voltages in results['voltage_results'].items():
                    for hour, voltage in zip(results['hours_analyzed'], voltages):
                        voltage_df_list.append({
                            'Hour': hour,
                            'Bus': bus_id,
                            'Voltage_pu': voltage
                        })
                
                voltage_df = pd.DataFrame(voltage_df_list)
                detailed_results['voltage_results'] = voltage_df.to_csv(index=False)
            
            # Combine all results
            if detailed_results:
                combined_csv = "\n\n".join([f"# {key}\n{data}" for key, data in detailed_results.items()])
                
                st.download_button(
                    label="Download CSV Data",
                    data=combined_csv,
                    file_name="detailed_loadflow_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    with export_col3:
        if st.button("Generate IEEE Format", use_container_width=True):
            st.info("IEEE format export functionality would be implemented here")
            st.write("This would export results in standard IEEE formats for utility analysis tools")

# Navigation
st.markdown("---")
nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    if st.button("← Back to Load Profiles", use_container_width=True):
        st.switch_page("pages/3_Load_Profile_Editor.py")

with nav_col2:
    if st.button("View Dashboard →", use_container_width=True):
        st.switch_page("pages/5_Dashboard.py")

with nav_col3:
    if st.button("← Back to Home", use_container_width=True):
        st.switch_page("app.py")

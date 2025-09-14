import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from utils.data_export import DataExporter
import json

st.set_page_config(page_title="Utility Dashboard", page_icon="📋", layout="wide")

st.title("📋 Utility Dashboard")
st.markdown("Professional dashboard with load growth projections, peak demand analysis, and capacity planning metrics.")

# Initialize data exporter
exporter = DataExporter()

# Check if we have data to display
has_forecast = st.session_state.forecast_results is not None
has_loadflow = st.session_state.loadflow_results is not None

if not has_forecast and not has_loadflow:
    st.warning("⚠️ No analysis results available. Please run forecasting or load flow analysis first.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Go to Forecasting", use_container_width=True):
            st.switch_page("pages/2_Load_Forecasting.py")
    with col2:
        if st.button("Configure Feeder", use_container_width=True):
            st.switch_page("pages/1_Feeder_Configuration.py")
    with col3:
        if st.button("Load Flow Analysis", use_container_width=True):
            st.switch_page("pages/4_Load_Flow_Analysis.py")
    
    st.stop()

# Executive Summary
st.header("Executive Summary")

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

feeder_config = st.session_state.feeder_config

with summary_col1:
    st.metric(
        "Feeder", 
        feeder_config['name'],
        help="Current feeder under analysis"
    )

with summary_col2:
    current_capacity_util = (feeder_config['base_load_mw'] / feeder_config['rated_capacity']) * 100
    st.metric(
        "Current Utilization", 
        f"{current_capacity_util:.1f}%",
        help="Current capacity utilization"
    )

with summary_col3:
    if has_forecast:
        future_peak = st.session_state.forecast_results['peak_demands'][-1]
        future_util = (future_peak / feeder_config['rated_capacity']) * 100
        util_change = future_util - current_capacity_util
        st.metric(
            "10-Year Utilization", 
            f"{future_util:.1f}%",
            delta=f"{util_change:+.1f}%",
            help="Projected capacity utilization in 10 years"
        )
    else:
        st.metric("10-Year Utilization", "N/A", help="Run forecast analysis to see projection")

with summary_col4:
    if has_forecast:
        ev_penetration = st.session_state.forecast_results['ev_penetration'][-1] * 100
        st.metric(
            "Final EV Penetration", 
            f"{ev_penetration:.0f}%",
            help="EV penetration after 10 years"
        )
    else:
        current_ev = feeder_config.get('ev_penetration', 0) * 100
        st.metric("Current EV Penetration", f"{current_ev:.1f}%")

st.markdown("---")

# Main Dashboard Content
if has_forecast:
    # Load Growth Analysis
    st.header("Load Growth Analysis")
    
    forecast_results = st.session_state.forecast_results
    years = forecast_results['years']
    
    # Create comprehensive dashboard charts
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Peak Demand Projection',
            'Capacity Utilization Trend',
            'EV Adoption Curve',
            'Annual Energy Growth',
            'Load Factor Evolution',
            'Infrastructure Requirements'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Peak Demand Projection
    fig.add_trace(
        go.Scatter(
            x=years,
            y=forecast_results['peak_demands'],
            mode='lines+markers',
            name='Peak Demand',
            line=dict(color='red', width=3),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    # Add capacity limit line
    capacity_line = [feeder_config['rated_capacity']] * len(years)
    fig.add_trace(
        go.Scatter(
            x=years,
            y=capacity_line,
            mode='lines',
            name='Rated Capacity',
            line=dict(color='orange', dash='dash', width=2),
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Capacity Utilization
    utilization = [(p / feeder_config['rated_capacity']) * 100 for p in forecast_results['peak_demands']]
    fig.add_trace(
        go.Scatter(
            x=years,
            y=utilization,
            mode='lines+markers',
            name='Utilization',
            line=dict(color='blue', width=3),
            fill='tonexty'
        ),
        row=1, col=2
    )
    
    # Add utilization thresholds
    fig.add_hline(y=80, line_dash="dash", line_color="yellow", row=1, col=2)
    fig.add_hline(y=90, line_dash="dash", line_color="red", row=1, col=2)
    
    # EV Adoption
    ev_penetration_pct = [p * 100 for p in forecast_results['ev_penetration']]
    fig.add_trace(
        go.Scatter(
            x=years,
            y=ev_penetration_pct,
            mode='lines+markers',
            name='EV Penetration',
            line=dict(color='green', width=3)
        ),
        row=1, col=3
    )
    
    # Annual Energy Growth
    fig.add_trace(
        go.Scatter(
            x=years,
            y=forecast_results['annual_energy'],
            mode='lines+markers',
            name='Annual Energy',
            line=dict(color='purple', width=3)
        ),
        row=2, col=1
    )
    
    # Load Factor Evolution
    load_factors = [energy/(peak*8760)*100 for energy, peak in zip(forecast_results['annual_energy'], forecast_results['peak_demands'])]
    fig.add_trace(
        go.Scatter(
            x=years,
            y=load_factors,
            mode='lines+markers',
            name='Load Factor',
            line=dict(color='teal', width=3)
        ),
        row=2, col=2
    )
    
    # Infrastructure Requirements (Investment Timeline)
    investment_timeline = []
    for i, (year, util) in enumerate(zip(years, utilization)):
        if util > 80:
            investment_timeline.append(util - 80)
        else:
            investment_timeline.append(0)
    
    fig.add_trace(
        go.Bar(
            x=years,
            y=investment_timeline,
            name='Investment Need',
            marker_color='orange'
        ),
        row=2, col=3
    )
    
    # Update layout
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=1, col=2)
    fig.update_xaxes(title_text="Year", row=1, col=3)
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=2)
    fig.update_xaxes(title_text="Year", row=2, col=3)
    
    fig.update_yaxes(title_text="MW", row=1, col=1)
    fig.update_yaxes(title_text="Utilization (%)", row=1, col=2)
    fig.update_yaxes(title_text="EV Penetration (%)", row=1, col=3)
    fig.update_yaxes(title_text="MWh", row=2, col=1)
    fig.update_yaxes(title_text="Load Factor (%)", row=2, col=2)
    fig.update_yaxes(title_text="Investment Priority", row=2, col=3)
    
    fig.update_layout(height=700, showlegend=False, title_text="Comprehensive Load Growth Dashboard")
    st.plotly_chart(fig, use_container_width=True)
    
    # Capacity Planning Insights
    st.header("Capacity Planning Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.subheader("Critical Planning Milestones")
        
        # Identify key planning milestones
        milestones = []
        
        # When capacity reaches 80%
        year_80_pct = None
        for year, util in zip(years, utilization):
            if util >= 80 and year_80_pct is None:
                year_80_pct = year
                milestones.append(f"🟡 {year}: Capacity reaches 80% - Begin expansion planning")
                break
        
        # When capacity reaches 90%
        year_90_pct = None
        for year, util in zip(years, utilization):
            if util >= 90 and year_90_pct is None:
                year_90_pct = year
                milestones.append(f"🟠 {year}: Capacity reaches 90% - Initiate upgrade projects")
                break
        
        # When capacity is exceeded
        year_100_pct = None
        for year, util in zip(years, utilization):
            if util >= 100 and year_100_pct is None:
                year_100_pct = year
                milestones.append(f"🔴 {year}: Capacity exceeded - Critical upgrade required")
                break
        
        # EV milestones
        for year, ev_pct in zip(years, ev_penetration_pct):
            if ev_pct >= 50 and year not in [m.split(':')[0].split(' ')[1] for m in milestones]:
                milestones.append(f"⚡ {year}: EV penetration reaches 50% - Infrastructure assessment needed")
                break
        
        if milestones:
            for milestone in milestones:
                st.write(milestone)
        else:
            st.success("✅ No critical capacity issues identified in forecast period")
        
        # Investment timeline
        st.subheader("Recommended Actions")
        
        max_util = max(utilization)
        if max_util > 100:
            st.error("🚨 Immediate Action Required: Capacity will be exceeded")
            st.write("• Accelerate capacity expansion projects")
            st.write("• Implement demand response programs")
            st.write("• Consider load transfer options")
        elif max_util > 90:
            st.warning("⚠️ Near-Term Action Required: High capacity utilization")
            st.write("• Begin detailed engineering studies")
            st.write("• Secure funding for capacity expansion")
            st.write("• Evaluate alternative solutions")
        elif max_util > 80:
            st.info("📋 Planning Action Required: Capacity planning needed")
            st.write("• Initiate capacity expansion planning")
            st.write("• Monitor load growth trends")
            st.write("• Assess distributed generation opportunities")
        else:
            st.success("✅ No immediate capacity concerns")
            st.write("• Continue routine monitoring")
            st.write("• Periodic forecast updates recommended")
    
    with insight_col2:
        st.subheader("Financial Impact Analysis")
        
        # Estimate investment requirements
        excess_capacity_needed = max(0, max(utilization) - 100) / 100 * feeder_config['rated_capacity']
        
        if excess_capacity_needed > 0:
            # Rough investment estimates ($/kW for distribution infrastructure)
            unit_cost_per_kw = 2000  # Typical range $1500-3000/kW
            estimated_investment = excess_capacity_needed * 1000 * unit_cost_per_kw
            
            st.metric(
                "Additional Capacity Needed", 
                f"{excess_capacity_needed:.1f} MW",
                help="Capacity addition required to meet projected demand"
            )
            st.metric(
                "Estimated Investment", 
                f"${estimated_investment/1e6:.1f}M",
                help=f"Rough estimate at ${unit_cost_per_kw}/kW"
            )
        
        # Calculate revenue impact from load growth
        base_energy = forecast_results['annual_energy'][0]
        final_energy = forecast_results['annual_energy'][-1]
        energy_growth = final_energy - base_energy
        
        # Typical utility revenue per MWh
        revenue_per_mwh = 80  # $80/MWh typical range
        additional_revenue = energy_growth * revenue_per_mwh
        
        st.metric(
            "Additional Annual Energy", 
            f"{energy_growth:.0f} MWh",
            help="Projected energy growth over 10 years"
        )
        st.metric(
            "Additional Annual Revenue", 
            f"${additional_revenue/1e6:.1f}M",
            help=f"Estimated at ${revenue_per_mwh}/MWh"
        )
        
        # EV impact analysis
        ev_load_contribution = (forecast_results['peak_demands'][-1] - forecast_results['peak_demands'][0]) * 0.4  # Assume 40% from EV
        st.metric(
            "EV Load Contribution", 
            f"{ev_load_contribution:.1f} MW",
            help="Estimated peak demand increase from EV adoption"
        )

# Load Flow Analysis Results (if available)
if has_loadflow:
    st.markdown("---")
    st.header("Network Performance Analysis")
    
    loadflow_results = st.session_state.loadflow_results
    
    # Network health indicators
    health_col1, health_col2, health_col3, health_col4 = st.columns(4)
    
    with health_col1:
        voltage_violations = loadflow_results.get('voltage_violations', 0)
        if voltage_violations == 0:
            st.metric("Voltage Violations", "0", delta="Good", delta_color="normal")
        else:
            st.metric("Voltage Violations", str(voltage_violations), delta="Poor", delta_color="inverse")
    
    with health_col2:
        thermal_violations = loadflow_results.get('thermal_violations', 0)
        if thermal_violations == 0:
            st.metric("Thermal Violations", "0", delta="Good", delta_color="normal")
        else:
            st.metric("Thermal Violations", str(thermal_violations), delta="Poor", delta_color="inverse")
    
    with health_col3:
        total_losses = loadflow_results.get('total_losses_mwh', 0)
        loss_percentage = (total_losses / (feeder_config['base_load_mw'] * len(loadflow_results['hours_analyzed']))) * 100
        st.metric("System Losses", f"{loss_percentage:.1f}%", help="Percentage of total energy lost")
    
    with health_col4:
        hours_analyzed = len(loadflow_results['hours_analyzed'])
        st.metric("Analysis Period", f"{hours_analyzed} hrs", help="Hours included in load flow analysis")
    
    # Network performance charts
    if 'voltage_results' in loadflow_results or 'thermal_results' in loadflow_results:
        performance_fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Voltage Profile Summary', 'Thermal Loading Summary')
        )
        
        # Voltage analysis
        if 'voltage_results' in loadflow_results:
            voltage_data = loadflow_results['voltage_results']
            bus_ids = list(voltage_data.keys())
            min_voltages = [np.min(voltage_data[bus]) for bus in bus_ids]
            max_voltages = [np.max(voltage_data[bus]) for bus in bus_ids]
            avg_voltages = [np.mean(voltage_data[bus]) for bus in bus_ids]
            
            performance_fig.add_trace(
                go.Scatter(
                    x=bus_ids,
                    y=min_voltages,
                    mode='markers',
                    name='Min Voltage',
                    marker=dict(color='red', symbol='triangle-down')
                ),
                row=1, col=1
            )
            
            performance_fig.add_trace(
                go.Scatter(
                    x=bus_ids,
                    y=max_voltages,
                    mode='markers',
                    name='Max Voltage',
                    marker=dict(color='blue', symbol='triangle-up')
                ),
                row=1, col=1
            )
            
            performance_fig.add_trace(
                go.Scatter(
                    x=bus_ids,
                    y=avg_voltages,
                    mode='lines+markers',
                    name='Avg Voltage',
                    line=dict(color='green', width=2)
                ),
                row=1, col=1
            )
            
            # Add voltage limits
            performance_fig.add_hline(y=0.95, line_dash="dash", line_color="red", row=1, col=1)
            performance_fig.add_hline(y=1.05, line_dash="dash", line_color="red", row=1, col=1)
        
        # Thermal analysis
        if 'thermal_results' in loadflow_results:
            thermal_data = loadflow_results['thermal_results']
            line_ids = list(thermal_data.keys())
            max_loadings = [np.max(thermal_data[line]) * 100 for line in line_ids]
            avg_loadings = [np.mean(thermal_data[line]) * 100 for line in line_ids]
            
            performance_fig.add_trace(
                go.Bar(
                    x=line_ids,
                    y=max_loadings,
                    name='Max Loading',
                    marker_color='orange',
                    opacity=0.7
                ),
                row=1, col=2
            )
            
            performance_fig.add_trace(
                go.Scatter(
                    x=line_ids,
                    y=avg_loadings,
                    mode='lines+markers',
                    name='Avg Loading',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=2
            )
            
            # Add thermal limit
            performance_fig.add_hline(y=90, line_dash="dash", line_color="red", row=1, col=2)
        
        performance_fig.update_xaxes(title_text="Bus ID", row=1, col=1)
        performance_fig.update_xaxes(title_text="Line ID", row=1, col=2)
        performance_fig.update_yaxes(title_text="Voltage (p.u.)", row=1, col=1)
        performance_fig.update_yaxes(title_text="Loading (%)", row=1, col=2)
        
        performance_fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(performance_fig, use_container_width=True)

# Operational Recommendations
st.markdown("---")
st.header("Operational Recommendations")

rec_col1, rec_col2 = st.columns(2)

with rec_col1:
    st.subheader("Short-Term Actions (1-2 years)")
    
    short_term_actions = []
    
    if has_forecast:
        current_util = (feeder_config['base_load_mw'] / feeder_config['rated_capacity']) * 100
        if current_util > 75:
            short_term_actions.append("🔍 Implement enhanced monitoring systems")
            short_term_actions.append("📊 Develop demand response programs")
        
        ev_current = feeder_config.get('ev_penetration', 0) * 100
        if ev_current > 20:
            short_term_actions.append("⚡ Install EV charging management systems")
            short_term_actions.append("🕒 Implement time-of-use rate structures")
    
    if has_loadflow:
        if loadflow_results.get('voltage_violations', 0) > 0:
            short_term_actions.append("🔧 Install voltage regulation equipment")
        
        if loadflow_results.get('thermal_violations', 0) > 0:
            short_term_actions.append("🌡️ Monitor thermally loaded equipment")
    
    if not short_term_actions:
        short_term_actions.append("✅ Continue routine monitoring and maintenance")
        short_term_actions.append("📈 Update load forecasts annually")
    
    for action in short_term_actions:
        st.write(action)

with rec_col2:
    st.subheader("Long-Term Planning (3-10 years)")
    
    long_term_actions = []
    
    if has_forecast:
        max_util = max(utilization) if has_forecast else 0
        
        if max_util > 90:
            long_term_actions.append("🏗️ Plan major capacity expansion projects")
            long_term_actions.append("💰 Secure capital funding for infrastructure")
        elif max_util > 80:
            long_term_actions.append("📋 Develop capacity expansion options")
            long_term_actions.append("🔍 Evaluate distributed generation integration")
        
        final_ev = forecast_results['ev_penetration'][-1] * 100
        if final_ev > 50:
            long_term_actions.append("🚗 Plan EV-specific infrastructure upgrades")
            long_term_actions.append("🔋 Consider grid-scale energy storage")
    
    if not long_term_actions:
        long_term_actions.append("🔄 Periodic infrastructure assessments")
        long_term_actions.append("🌱 Evaluate emerging technology integration")
        long_term_actions.append("📊 Continue advanced analytics development")
    
    for action in long_term_actions:
        st.write(action)

# Data Export Section
st.markdown("---")
st.header("Export Dashboard Data")

export_col1, export_col2, export_col3 = st.columns(3)

with export_col1:
    if st.button("📊 Export Executive Summary", use_container_width=True):
        if has_forecast or has_loadflow:
            summary_data = exporter.create_executive_summary(
                feeder_config,
                st.session_state.forecast_results,
                st.session_state.loadflow_results
            )
            
            st.download_button(
                label="Download Summary Report",
                data=summary_data,
                file_name=f"executive_summary_{feeder_config['name']}_{datetime.datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.error("No analysis data available for export")

with export_col2:
    if st.button("📈 Export Forecast Data", use_container_width=True):
        if has_forecast:
            forecast_data = exporter.export_forecast_data(st.session_state.forecast_results)
            
            st.download_button(
                label="Download Forecast CSV",
                data=forecast_data,
                file_name=f"forecast_data_{feeder_config['name']}_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.error("No forecast data available for export")

with export_col3:
    if st.button("🔌 Export Load Flow Results", use_container_width=True):
        if has_loadflow:
            loadflow_data = exporter.export_loadflow_data(st.session_state.loadflow_results)
            
            st.download_button(
                label="Download Load Flow CSV",
                data=loadflow_data,
                file_name=f"loadflow_results_{feeder_config['name']}_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.error("No load flow data available for export")

# System Health Summary
st.markdown("---")
st.header("System Health Summary")

health_status = "🟢 Good"
health_issues = []

if has_forecast:
    max_util = max(utilization)
    if max_util > 100:
        health_status = "🔴 Critical"
        health_issues.append("Capacity will be exceeded within forecast period")
    elif max_util > 90:
        health_status = "🟠 Warning"
        health_issues.append("High capacity utilization projected")

if has_loadflow:
    if loadflow_results.get('voltage_violations', 0) > 0:
        if health_status == "🟢 Good":
            health_status = "🟡 Caution"
        health_issues.append(f"{loadflow_results['voltage_violations']} voltage violations detected")
    
    if loadflow_results.get('thermal_violations', 0) > 0:
        if health_status == "🟢 Good":
            health_status = "🟡 Caution"
        health_issues.append(f"{loadflow_results['thermal_violations']} thermal violations detected")

st.subheader(f"Overall System Health: {health_status}")

if health_issues:
    st.warning("Issues Identified:")
    for issue in health_issues:
        st.write(f"• {issue}")
else:
    st.success("✅ No significant issues identified in current analysis")

# Navigation
st.markdown("---")
nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    if st.button("← Back to Load Flow", use_container_width=True):
        st.switch_page("pages/4_Load_Flow_Analysis.py")

with nav_col2:
    if st.button("← Back to Home", use_container_width=True):
        st.switch_page("app.py")

with nav_col3:
    if st.button("Start New Analysis", use_container_width=True):
        # Clear all analysis results
        st.session_state.forecast_results = None
        st.session_state.loadflow_results = None
        st.success("Analysis results cleared. Ready for new analysis.")
        st.rerun()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.load_profiles import LoadProfileGenerator
import datetime

st.set_page_config(page_title="Load Profile Editor", page_icon="📊", layout="wide")

st.title("📊 Load Profile Editor")
st.markdown("Visualize and edit 8760 hourly load profiles with interactive charts and analysis.")

# Initialize profile generator
profile_gen = LoadProfileGenerator()

# Profile Selection and Loading
st.header("Load Profile Management")

profile_col1, profile_col2 = st.columns([2, 1])

with profile_col1:
    st.subheader("Current Profile Options")
    
    profile_source = st.radio(
        "Select Profile Source:",
        ["Current Session Profile", "Generate New Profile", "Upload Profile", "Template Profiles"],
        horizontal=True
    )

with profile_col2:
    st.subheader("Profile Statistics")
    current_profile = st.session_state.load_profile
    
    st.metric("Peak Load (p.u.)", f"{np.max(current_profile):.3f}")
    st.metric("Min Load (p.u.)", f"{np.min(current_profile):.3f}")
    st.metric("Average Load (p.u.)", f"{np.mean(current_profile):.3f}")
    st.metric("Load Factor", f"{np.mean(current_profile)/np.max(current_profile):.3f}")

# Profile Generation/Loading based on selection
if profile_source == "Generate New Profile":
    st.subheader("Generate New Load Profile")
    
    gen_col1, gen_col2 = st.columns(2)
    
    with gen_col1:
        profile_type = st.selectbox(
            "Profile Type",
            ["Residential", "Commercial", "Industrial", "Mixed", "Custom"]
        )
        
        seasonality = st.slider("Seasonality Strength", 0.0, 1.0, 0.3, step=0.1)
        daily_variation = st.slider("Daily Variation", 0.0, 1.0, 0.4, step=0.1)
        weekly_pattern = st.checkbox("Include Weekly Pattern", value=True)
        
    with gen_col2:
        noise_level = st.slider("Noise Level", 0.0, 0.2, 0.05, step=0.01)
        peak_hour = st.slider("Peak Hour", 0, 23, 19)
        min_load_ratio = st.slider("Minimum Load Ratio", 0.1, 0.8, 0.3, step=0.05)
    
    if st.button("Generate Profile", type="primary"):
        with st.spinner("Generating load profile..."):
            new_profile = profile_gen.generate_profile(
                profile_type=profile_type,
                seasonality=seasonality,
                daily_variation=daily_variation,
                weekly_pattern=weekly_pattern,
                noise_level=noise_level,
                peak_hour=peak_hour,
                min_load_ratio=min_load_ratio
            )
            st.session_state.load_profile = new_profile
            st.success("New profile generated!")
            st.rerun()

elif profile_source == "Upload Profile":
    st.subheader("Upload Load Profile")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file with 8760 hourly values",
        type=['csv'],
        help="CSV should contain a single column with 8760 hourly load values"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if len(df) == 8760:
                # Assume the first column contains load values
                load_values = df.iloc[:, 0].values
                
                # Normalize to per unit if needed
                if np.max(load_values) > 10:  # Assume it's in MW or kW, normalize to p.u.
                    load_values = load_values / np.max(load_values)
                
                st.session_state.load_profile = load_values
                st.success("Profile uploaded successfully!")
                st.rerun()
            else:
                st.error(f"File must contain exactly 8760 rows. Found {len(df)} rows.")
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

elif profile_source == "Template Profiles":
    st.subheader("Template Load Profiles")
    
    template_col1, template_col2 = st.columns(2)
    
    with template_col1:
        template_type = st.selectbox(
            "Select Template",
            [
                "Typical Residential Feeder",
                "Commercial District",
                "Industrial Complex",
                "Rural Agricultural",
                "Urban Mixed-Use",
                "University Campus"
            ]
        )
        
    with template_col2:
        if st.button("Load Template", type="primary"):
            template_profile = profile_gen.get_template_profile(template_type)
            st.session_state.load_profile = template_profile
            st.success(f"Loaded {template_type} template!")
            st.rerun()

st.markdown("---")

# Profile Visualization
st.header("Load Profile Visualization")

viz_col1, viz_col2 = st.columns([3, 1])

with viz_col2:
    st.subheader("Visualization Options")
    
    view_type = st.selectbox(
        "View Type",
        ["Full Year", "Monthly View", "Weekly View", "Daily Patterns", "Duration Curve"]
    )
    
    if view_type in ["Monthly View", "Weekly View"]:
        if view_type == "Monthly View":
            month_options = [
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
            ]
            selected_month = st.selectbox("Select Month", month_options)
            month_num = month_options.index(selected_month) + 1
        
        elif view_type == "Weekly View":
            week_num = st.slider("Select Week", 1, 52, 26)
    
    show_statistics = st.checkbox("Show Statistics", value=True)
    show_peaks = st.checkbox("Highlight Peaks", value=True)

with viz_col1:
    current_profile = st.session_state.load_profile
    
    # Create visualization based on selected view type
    fig = go.Figure()
    
    if view_type == "Full Year":
        hours = np.arange(8760)
        days = hours / 24
        
        fig.add_trace(go.Scatter(
            x=days,
            y=current_profile,
            mode='lines',
            name='Load Profile',
            line=dict(color='blue', width=1)
        ))
        
        # Add peak markers if requested
        if show_peaks:
            peak_indices = np.where(current_profile >= np.percentile(current_profile, 95))[0]
            fig.add_trace(go.Scatter(
                x=peak_indices / 24,
                y=current_profile[peak_indices],
                mode='markers',
                name='Peak Hours (>95%)',
                marker=dict(color='red', size=3)
            ))
        
        fig.update_layout(
            title='Annual Load Profile (8760 Hours)',
            xaxis_title='Day of Year',
            yaxis_title='Load (p.u.)',
            height=500
        )
    
    elif view_type == "Monthly View":
        # Calculate start and end hours for the selected month
        days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]  # Non-leap year
        start_day = sum(days_in_months[:month_num-1])
        end_day = start_day + days_in_months[month_num-1]
        
        start_hour = start_day * 24
        end_hour = end_day * 24
        
        month_profile = current_profile[start_hour:end_hour]
        hours_in_month = np.arange(len(month_profile))
        days_in_month = hours_in_month / 24
        
        fig.add_trace(go.Scatter(
            x=days_in_month,
            y=month_profile,
            mode='lines',
            name=f'{selected_month} Load Profile',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title=f'{selected_month} Load Profile',
            xaxis_title=f'Day of {selected_month}',
            yaxis_title='Load (p.u.)',
            height=500
        )
    
    elif view_type == "Weekly View":
        start_hour = (week_num - 1) * 24 * 7
        end_hour = start_hour + 24 * 7
        
        week_profile = current_profile[start_hour:end_hour]
        hours_in_week = np.arange(len(week_profile))
        
        # Create day labels
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_labels = []
        for day in range(7):
            for hour in range(24):
                if hour == 12:  # Label at noon
                    day_labels.append(day_names[day])
                else:
                    day_labels.append('')
        
        fig.add_trace(go.Scatter(
            x=hours_in_week,
            y=week_profile,
            mode='lines+markers',
            name=f'Week {week_num} Profile',
            line=dict(color='purple', width=2)
        ))
        
        # Add vertical lines for day boundaries
        for day in range(1, 7):
            fig.add_vline(x=day*24, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title=f'Week {week_num} Load Profile',
            xaxis_title='Hour of Week',
            yaxis_title='Load (p.u.)',
            height=500
        )
    
    elif view_type == "Daily Patterns":
        # Show average daily patterns for different day types
        weekday_pattern = np.zeros(24)
        weekend_pattern = np.zeros(24)
        
        # Calculate average patterns
        for hour in range(8760):
            day_of_year = hour // 24
            hour_of_day = hour % 24
            day_of_week = day_of_year % 7
            
            if day_of_week < 5:  # Weekday
                weekday_pattern[hour_of_day] += current_profile[hour]
            else:  # Weekend
                weekend_pattern[hour_of_day] += current_profile[hour]
        
        # Average over the number of days
        weekday_pattern /= (365 * 5 / 7)  # Approximate number of weekdays
        weekend_pattern /= (365 * 2 / 7)   # Approximate number of weekend days
        
        hours_of_day = np.arange(24)
        
        fig.add_trace(go.Scatter(
            x=hours_of_day,
            y=weekday_pattern,
            mode='lines+markers',
            name='Weekday Pattern',
            line=dict(color='blue', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=hours_of_day,
            y=weekend_pattern,
            mode='lines+markers',
            name='Weekend Pattern',
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            title='Average Daily Load Patterns',
            xaxis_title='Hour of Day',
            yaxis_title='Load (p.u.)',
            height=500
        )
    
    elif view_type == "Duration Curve":
        sorted_loads = np.sort(current_profile)[::-1]  # Sort in descending order
        percentiles = np.arange(len(sorted_loads)) / len(sorted_loads) * 100
        
        fig.add_trace(go.Scatter(
            x=percentiles,
            y=sorted_loads,
            mode='lines',
            name='Load Duration Curve',
            line=dict(color='orange', width=2)
        ))
        
        # Add percentile markers
        key_percentiles = [5, 10, 25, 50, 75, 90, 95]
        for pct in key_percentiles:
            idx = int(pct * len(sorted_loads) / 100)
            fig.add_trace(go.Scatter(
                x=[pct],
                y=[sorted_loads[idx]],
                mode='markers+text',
                name=f'{pct}%ile',
                text=[f'{pct}%'],
                textposition='top center',
                marker=dict(size=8, color='red'),
                showlegend=False
            ))
        
        fig.update_layout(
            title='Load Duration Curve',
            xaxis_title='Percentage of Time (%)',
            yaxis_title='Load (p.u.)',
            height=500
        )
    
    st.plotly_chart(fig, use_container_width=True)

# Profile Statistics and Analysis
if show_statistics:
    st.subheader("Load Profile Statistics")
    
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    
    with stats_col1:
        st.markdown("**Basic Statistics**")
        stats_df = pd.DataFrame({
            'Statistic': ['Maximum', 'Minimum', 'Mean', 'Median', 'Std Dev'],
            'Value (p.u.)': [
                f"{np.max(current_profile):.4f}",
                f"{np.min(current_profile):.4f}",
                f"{np.mean(current_profile):.4f}",
                f"{np.median(current_profile):.4f}",
                f"{np.std(current_profile):.4f}"
            ]
        })
        st.dataframe(stats_df, hide_index=True)
    
    with stats_col2:
        st.markdown("**Load Factors**")
        annual_lf = np.mean(current_profile) / np.max(current_profile)
        monthly_lfs = []
        
        for month in range(12):
            days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            start_day = sum(days_in_months[:month])
            end_day = start_day + days_in_months[month]
            start_hour = start_day * 24
            end_hour = end_day * 24
            
            month_profile = current_profile[start_hour:end_hour]
            month_lf = np.mean(month_profile) / np.max(month_profile)
            monthly_lfs.append(month_lf)
        
        lf_df = pd.DataFrame({
            'Period': ['Annual', 'Highest Monthly', 'Lowest Monthly', 'Average Monthly'],
            'Load Factor': [
                f"{annual_lf:.3f}",
                f"{max(monthly_lfs):.3f}",
                f"{min(monthly_lfs):.3f}",
                f"{np.mean(monthly_lfs):.3f}"
            ]
        })
        st.dataframe(lf_df, hide_index=True)
    
    with stats_col3:
        st.markdown("**Peak Analysis**")
        top_1_pct = np.percentile(current_profile, 99)
        top_5_pct = np.percentile(current_profile, 95)
        top_10_pct = np.percentile(current_profile, 90)
        
        # Count hours above thresholds
        hours_above_99 = np.sum(current_profile >= top_1_pct)
        hours_above_95 = np.sum(current_profile >= top_5_pct)
        hours_above_90 = np.sum(current_profile >= top_10_pct)
        
        peak_df = pd.DataFrame({
            'Percentile': ['99th %ile', '95th %ile', '90th %ile'],
            'Load (p.u.)': [f"{top_1_pct:.3f}", f"{top_5_pct:.3f}", f"{top_10_pct:.3f}"],
            'Hours Above': [f"{hours_above_99}", f"{hours_above_95}", f"{hours_above_90}"]
        })
        st.dataframe(peak_df, hide_index=True)

st.markdown("---")

# Profile Editing Tools
st.header("Profile Editing Tools")

edit_col1, edit_col2 = st.columns(2)

with edit_col1:
    st.subheader("Mathematical Adjustments")
    
    # Scaling operations
    scale_factor = st.slider("Scale Factor", 0.5, 2.0, 1.0, step=0.05)
    offset_value = st.slider("Offset Value", -0.2, 0.2, 0.0, step=0.01)
    
    # Peak shaving
    enable_peak_shaving = st.checkbox("Enable Peak Shaving")
    if enable_peak_shaving:
        peak_shaving_threshold = st.slider("Peak Shaving Threshold (p.u.)", 0.7, 1.0, 0.9, step=0.01)
        shaving_factor = st.slider("Shaving Factor", 0.1, 1.0, 0.8, step=0.05)

with edit_col2:
    st.subheader("Seasonal Adjustments")
    
    # Seasonal multipliers
    summer_multiplier = st.slider("Summer Multiplier", 0.8, 1.5, 1.0, step=0.05)
    winter_multiplier = st.slider("Winter Multiplier", 0.8, 1.5, 1.0, step=0.05)
    
    # Add noise or smooth
    operation_type = st.selectbox("Operation", ["None", "Add Noise", "Smooth Profile"])
    
    if operation_type == "Add Noise":
        noise_strength = st.slider("Noise Strength", 0.0, 0.1, 0.02, step=0.01)
    elif operation_type == "Smooth Profile":
        smoothing_window = st.slider("Smoothing Window (hours)", 1, 24, 3)

# Apply modifications
if st.button("Apply Modifications", type="primary"):
    with st.spinner("Applying modifications..."):
        modified_profile = current_profile.copy()
        
        # Apply scaling and offset
        modified_profile = modified_profile * scale_factor + offset_value
        
        # Apply seasonal adjustments
        for hour in range(8760):
            day_of_year = hour // 24
            
            # Summer: June-August (days 152-243)
            if 152 <= day_of_year <= 243:
                modified_profile[hour] *= summer_multiplier
            # Winter: December-February (days 0-59, 334-365)
            elif day_of_year <= 59 or day_of_year >= 334:
                modified_profile[hour] *= winter_multiplier
        
        # Apply peak shaving if enabled
        if enable_peak_shaving:
            peak_mask = modified_profile > peak_shaving_threshold
            modified_profile[peak_mask] = (
                peak_shaving_threshold + 
                (modified_profile[peak_mask] - peak_shaving_threshold) * shaving_factor
            )
        
        # Apply additional operations
        if operation_type == "Add Noise":
            noise = np.random.normal(0, noise_strength, size=8760)
            modified_profile += noise
        elif operation_type == "Smooth Profile":
            # Simple moving average smoothing
            padded_profile = np.pad(modified_profile, smoothing_window//2, mode='edge')
            for i in range(8760):
                start_idx = i
                end_idx = i + smoothing_window
                modified_profile[i] = np.mean(padded_profile[start_idx:end_idx])
        
        # Ensure non-negative values
        modified_profile = np.maximum(modified_profile, 0.1)
        
        # Update session state
        st.session_state.load_profile = modified_profile
        
        st.success("Profile modifications applied successfully!")
        st.rerun()

st.markdown("---")

# Export Options
st.header("Export Options")

export_col1, export_col2, export_col3 = st.columns(3)

with export_col1:
    if st.button("Export as CSV", use_container_width=True):
        # Create DataFrame for export
        hours = np.arange(8760)
        timestamps = pd.date_range(
            start='2024-01-01 00:00:00',
            periods=8760,
            freq='H'
        )
        
        export_df = pd.DataFrame({
            'Timestamp': timestamps,
            'Hour': hours,
            'Load_pu': current_profile
        })
        
        csv_data = export_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="load_profile_8760.csv",
            mime="text/csv",
            use_container_width=True
        )

with export_col2:
    if st.button("Export Statistics", use_container_width=True):
        # Prepare statistics for export
        stats_data = {
            'Annual Statistics': {
                'Peak Load (p.u.)': np.max(current_profile),
                'Minimum Load (p.u.)': np.min(current_profile),
                'Average Load (p.u.)': np.mean(current_profile),
                'Load Factor': np.mean(current_profile) / np.max(current_profile),
                'Standard Deviation': np.std(current_profile)
            }
        }
        
        # Add monthly statistics
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for i, month in enumerate(months):
            days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            start_day = sum(days_in_months[:i])
            end_day = start_day + days_in_months[i]
            start_hour = start_day * 24
            end_hour = end_day * 24
            
            month_profile = current_profile[start_hour:end_hour]
            stats_data[f'{month} Statistics'] = {
                'Peak (p.u.)': np.max(month_profile),
                'Average (p.u.)': np.mean(month_profile),
                'Load Factor': np.mean(month_profile) / np.max(month_profile)
            }
        
        # Convert to JSON string for download
        import json
        json_data = json.dumps(stats_data, indent=2)
        
        st.download_button(
            label="Download Statistics",
            data=json_data,
            file_name="load_profile_statistics.json",
            mime="application/json",
            use_container_width=True
        )

with export_col3:
    if st.button("Generate Report", use_container_width=True):
        # Create a comprehensive report
        report = f"""
LOAD PROFILE ANALYSIS REPORT
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

FEEDER INFORMATION:
- Feeder Name: {st.session_state.feeder_config['name']}
- Base Voltage: {st.session_state.feeder_config['base_voltage']:.1f} kV
- Rated Capacity: {st.session_state.feeder_config['rated_capacity']:.1f} MVA
- Base Load: {st.session_state.feeder_config['base_load_mw']:.1f} MW

ANNUAL PROFILE STATISTICS:
- Peak Load: {np.max(current_profile):.4f} p.u.
- Minimum Load: {np.min(current_profile):.4f} p.u.
- Average Load: {np.mean(current_profile):.4f} p.u.
- Load Factor: {np.mean(current_profile)/np.max(current_profile):.3f}
- Standard Deviation: {np.std(current_profile):.4f}

LOAD CHARACTERISTICS:
- Hours above 95th percentile: {np.sum(current_profile >= np.percentile(current_profile, 95))} hours
- Hours below 25th percentile: {np.sum(current_profile <= np.percentile(current_profile, 25))} hours
- Peak-to-Average Ratio: {np.max(current_profile)/np.mean(current_profile):.2f}
- Coefficient of Variation: {np.std(current_profile)/np.mean(current_profile):.3f}

OPERATIONAL INSIGHTS:
- Annual Energy (if base load = {st.session_state.feeder_config['base_load_mw']:.1f} MW): {np.sum(current_profile) * st.session_state.feeder_config['base_load_mw']:.0f} MWh
- Capacity Utilization: {np.max(current_profile) * st.session_state.feeder_config['base_load_mw'] / st.session_state.feeder_config['rated_capacity'] * 100:.1f}%
- Load Diversity: {1/np.max(current_profile):.3f}

This report provides a comprehensive analysis of the 8760-hour load profile
for integrated resource planning and capacity assessments.
        """
        
        st.download_button(
            label="Download Report",
            data=report,
            file_name="load_profile_report.txt",
            mime="text/plain",
            use_container_width=True
        )

# Navigation
st.markdown("---")
nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    if st.button("← Back to Forecasting", use_container_width=True):
        st.switch_page("pages/2_Load_Forecasting.py")

with nav_col2:
    if st.button("Run Load Flow Analysis →", use_container_width=True):
        st.switch_page("pages/4_Load_Flow_Analysis.py")

with nav_col3:
    if st.button("View Dashboard →", use_container_width=True):
        st.switch_page("pages/5_Dashboard.py")

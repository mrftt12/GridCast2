import pandas as pd
import numpy as np
import json
from datetime import datetime
import io

class DataExporter:
    """
    Comprehensive data export utilities for electric utility planning applications.
    Provides export functionality for forecasts, load flow results, and analysis reports
    in various standard utility formats.
    """
    
    def __init__(self):
        self.export_timestamp = datetime.now()
    
    def create_executive_summary(self, feeder_config, forecast_results, loadflow_results):
        """Create comprehensive executive summary report."""
        
        try:
            report = []
            report.append("=" * 80)
            report.append("ELECTRIC UTILITY INTEGRATED RESOURCE PLANNING")
            report.append("EXECUTIVE SUMMARY REPORT")
            report.append("=" * 80)
            report.append(f"Generated: {self.export_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            # Feeder Information
            report.append("FEEDER INFORMATION")
            report.append("-" * 40)
            report.append(f"Feeder Name: {feeder_config.get('name', 'N/A')}")
            report.append(f"Base Voltage: {feeder_config.get('base_voltage', 0):.1f} kV")
            report.append(f"Rated Capacity: {feeder_config.get('rated_capacity', 0):.1f} MVA")
            report.append(f"Length: {feeder_config.get('length_km', 0):.1f} km")
            report.append(f"Number of Customers: {feeder_config.get('customers', 0):,}")
            report.append(f"Base Load: {feeder_config.get('base_load_mw', 0):.1f} MW")
            report.append(f"Current EV Penetration: {feeder_config.get('ev_penetration', 0)*100:.1f}%")
            report.append(f"DG Capacity: {feeder_config.get('dg_capacity_mw', 0):.1f} MW")
            report.append("")
            
            # Current System Status
            current_utilization = (feeder_config.get('base_load_mw', 0) / 
                                 feeder_config.get('rated_capacity', 1)) * 100
            report.append("CURRENT SYSTEM STATUS")
            report.append("-" * 40)
            report.append(f"Current Capacity Utilization: {current_utilization:.1f}%")
            
            if current_utilization > 90:
                report.append("⚠️  HIGH UTILIZATION - Immediate attention required")
            elif current_utilization > 80:
                report.append("⚠️  ELEVATED UTILIZATION - Monitor closely")
            else:
                report.append("✅ NORMAL UTILIZATION - Routine monitoring")
            report.append("")
            
            # Forecast Results Summary
            if forecast_results:
                report.append("10-YEAR FORECAST SUMMARY")
                report.append("-" * 40)
                report.append(f"Scenario: {forecast_results.get('scenario', {}).get('name', 'Unknown')}")
                
                initial_peak = forecast_results['peak_demands'][0]
                final_peak = forecast_results['peak_demands'][-1]
                peak_growth = ((final_peak / initial_peak) - 1) * 100
                
                initial_energy = forecast_results['annual_energy'][0]
                final_energy = forecast_results['annual_energy'][-1]
                energy_growth = ((final_energy / initial_energy) - 1) * 100
                
                final_ev_penetration = forecast_results['ev_penetration'][-1] * 100
                final_utilization = (final_peak / feeder_config.get('rated_capacity', 1)) * 100
                
                report.append(f"Peak Demand Growth: {peak_growth:.1f}% ({initial_peak:.1f} → {final_peak:.1f} MW)")
                report.append(f"Energy Growth: {energy_growth:.1f}% ({initial_energy:.0f} → {final_energy:.0f} MWh)")
                report.append(f"Final EV Penetration: {final_ev_penetration:.1f}%")
                report.append(f"10-Year Capacity Utilization: {final_utilization:.1f}%")
                
                # Capacity planning insights
                report.append("")
                report.append("CAPACITY PLANNING INSIGHTS")
                report.append("-" * 40)
                
                if final_utilization > 100:
                    report.append("🔴 CRITICAL: Capacity will be exceeded - Immediate expansion required")
                elif final_utilization > 90:
                    report.append("🟠 WARNING: High utilization projected - Plan capacity expansion")
                elif final_utilization > 80:
                    report.append("🟡 CAUTION: Elevated utilization - Begin expansion planning")
                else:
                    report.append("✅ ADEQUATE: Capacity sufficient for forecast period")
                
                # Find critical years
                years = forecast_results['years']
                utilizations = [(p / feeder_config.get('rated_capacity', 1)) * 100 
                              for p in forecast_results['peak_demands']]
                
                year_80_pct = None
                year_90_pct = None
                year_100_pct = None
                
                for year, util in zip(years, utilizations):
                    if util >= 80 and year_80_pct is None:
                        year_80_pct = year
                    if util >= 90 and year_90_pct is None:
                        year_90_pct = year
                    if util >= 100 and year_100_pct is None:
                        year_100_pct = year
                
                if year_80_pct:
                    report.append(f"80% Capacity Reached: {year_80_pct}")
                if year_90_pct:
                    report.append(f"90% Capacity Reached: {year_90_pct}")
                if year_100_pct:
                    report.append(f"Capacity Exceeded: {year_100_pct}")
                
                report.append("")
            
            # Load Flow Analysis Summary
            if loadflow_results:
                report.append("NETWORK PERFORMANCE ANALYSIS")
                report.append("-" * 40)
                report.append(f"Analysis Period: {len(loadflow_results.get('hours_analyzed', []))} hours")
                
                voltage_violations = loadflow_results.get('voltage_violations', 0)
                thermal_violations = loadflow_results.get('thermal_violations', 0)
                total_losses = loadflow_results.get('total_losses_mwh', 0)
                
                report.append(f"Voltage Violations: {voltage_violations}")
                report.append(f"Thermal Violations: {thermal_violations}")
                report.append(f"System Losses: {total_losses:.1f} MWh")
                
                if voltage_violations == 0 and thermal_violations == 0:
                    report.append("✅ NETWORK STATUS: No violations detected")
                else:
                    report.append("⚠️  NETWORK STATUS: Violations detected - Review required")
                
                # Loss analysis
                if total_losses > 0:
                    hours_analyzed = len(loadflow_results.get('hours_analyzed', []))
                    avg_loss_rate = (total_losses / (feeder_config.get('base_load_mw', 1) * hours_analyzed)) * 100
                    report.append(f"Average Loss Rate: {avg_loss_rate:.2f}%")
                    
                    if avg_loss_rate > 8:
                        report.append("⚠️  HIGH LOSSES - Consider system improvements")
                    elif avg_loss_rate > 5:
                        report.append("⚠️  ELEVATED LOSSES - Monitor for optimization opportunities")
                    else:
                        report.append("✅ ACCEPTABLE LOSSES - Within normal range")
                
                report.append("")
            
            # Key Recommendations
            report.append("KEY RECOMMENDATIONS")
            report.append("-" * 40)
            
            recommendations = []
            
            # Capacity recommendations
            if forecast_results:
                final_util = (forecast_results['peak_demands'][-1] / 
                             feeder_config.get('rated_capacity', 1)) * 100
                
                if final_util > 100:
                    recommendations.append("🔧 IMMEDIATE: Accelerate capacity expansion projects")
                    recommendations.append("📊 IMMEDIATE: Implement aggressive demand response programs")
                elif final_util > 90:
                    recommendations.append("📋 SHORT-TERM: Begin detailed engineering studies for expansion")
                    recommendations.append("💰 SHORT-TERM: Secure funding for capacity upgrades")
                elif final_util > 80:
                    recommendations.append("📅 MEDIUM-TERM: Initiate capacity expansion planning")
                
                # EV recommendations
                final_ev = forecast_results['ev_penetration'][-1] * 100
                if final_ev > 50:
                    recommendations.append("⚡ PLANNING: Prepare for high EV adoption impacts")
                    recommendations.append("🔌 PLANNING: Consider EV-specific infrastructure")
                
                if final_ev > 30:
                    recommendations.append("🕒 OPERATIONAL: Implement smart charging programs")
            
            # Network recommendations
            if loadflow_results:
                if loadflow_results.get('voltage_violations', 0) > 0:
                    recommendations.append("🔧 TECHNICAL: Install voltage regulation equipment")
                
                if loadflow_results.get('thermal_violations', 0) > 0:
                    recommendations.append("🌡️  TECHNICAL: Address thermally overloaded equipment")
                
                total_losses = loadflow_results.get('total_losses_mwh', 0)
                if total_losses > 0:
                    hours = len(loadflow_results.get('hours_analyzed', []))
                    loss_rate = (total_losses / (feeder_config.get('base_load_mw', 1) * hours)) * 100
                    if loss_rate > 5:
                        recommendations.append("📈 EFFICIENCY: Investigate loss reduction opportunities")
            
            # General recommendations
            if not recommendations:
                recommendations.append("✅ MONITORING: Continue routine monitoring and analysis")
                recommendations.append("📊 PLANNING: Update forecasts annually")
                recommendations.append("🔍 ASSESSMENT: Periodic infrastructure assessments")
            
            for rec in recommendations:
                report.append(rec)
            
            report.append("")
            report.append("=" * 80)
            report.append("END OF EXECUTIVE SUMMARY")
            report.append("=" * 80)
            
            return "\n".join(report)
            
        except Exception as e:
            return f"Error generating executive summary: {str(e)}"
    
    def export_forecast_data(self, forecast_results):
        """Export forecast results to CSV format."""
        
        try:
            if not forecast_results:
                return "No forecast data available"
            
            # Create comprehensive forecast DataFrame
            years = forecast_results['years']
            
            forecast_df = pd.DataFrame({
                'Year': years,
                'Peak_Demand_MW': forecast_results['peak_demands'],
                'Annual_Energy_MWh': forecast_results['annual_energy'],
                'EV_Penetration_Percent': [p * 100 for p in forecast_results['ev_penetration']],
                'Load_Factor': forecast_results['load_factors'],
            })
            
            # Add solar capacity if available
            if 'solar_capacity' in forecast_results:
                forecast_df['Solar_Capacity_MW'] = forecast_results['solar_capacity']
            
            # Calculate growth rates
            if len(years) > 1:
                peak_growth = []
                energy_growth = []
                
                for i in range(len(years)):
                    if i == 0:
                        peak_growth.append(0)
                        energy_growth.append(0)
                    else:
                        pg = ((forecast_results['peak_demands'][i] / 
                              forecast_results['peak_demands'][i-1]) - 1) * 100
                        eg = ((forecast_results['annual_energy'][i] / 
                              forecast_results['annual_energy'][i-1]) - 1) * 100
                        peak_growth.append(pg)
                        energy_growth.append(eg)
                
                forecast_df['Peak_Growth_Percent'] = peak_growth
                forecast_df['Energy_Growth_Percent'] = energy_growth
            
            # Add scenario information as header
            scenario_info = forecast_results.get('scenario', {})
            header_lines = [
                f"# Electric Utility Load Forecast Results",
                f"# Generated: {self.export_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                f"# Scenario: {scenario_info.get('name', 'Unknown')}",
                f"# Base Load Growth Rate: {scenario_info.get('base_load_growth', 0)*100:.1f}%",
                f"# EV Growth Rate: {scenario_info.get('ev_growth_rate', 0)*100:.1f}%",
                f"# Solar Growth Rate: {scenario_info.get('solar_growth_rate', 0)*100:.1f}%",
                ""
            ]
            
            # Convert to CSV
            csv_buffer = io.StringIO()
            
            # Write header
            for line in header_lines:
                csv_buffer.write(line + "\n")
            
            # Write data
            forecast_df.to_csv(csv_buffer, index=False)
            
            return csv_buffer.getvalue()
            
        except Exception as e:
            return f"Error exporting forecast data: {str(e)}"
    
    def export_loadflow_data(self, loadflow_results):
        """Export load flow results to CSV format."""
        
        try:
            if not loadflow_results:
                return "No load flow data available"
            
            csv_sections = []
            
            # Header information
            header_lines = [
                f"# Time Series Load Flow Analysis Results",
                f"# Generated: {self.export_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                f"# Analysis Period: {len(loadflow_results.get('hours_analyzed', []))} hours",
                f"# Voltage Violations: {loadflow_results.get('voltage_violations', 0)}",
                f"# Thermal Violations: {loadflow_results.get('thermal_violations', 0)}",
                f"# Total Losses: {loadflow_results.get('total_losses_mwh', 0):.1f} MWh",
                ""
            ]
            
            csv_sections.extend(header_lines)
            
            # Voltage results
            if 'voltage_results' in loadflow_results:
                csv_sections.append("# VOLTAGE RESULTS (p.u.)")
                
                voltage_data = loadflow_results['voltage_results']
                hours = loadflow_results['hours_analyzed']
                
                # Create voltage DataFrame
                voltage_df_data = {'Hour': hours}
                for bus_id in sorted(voltage_data.keys()):
                    voltage_df_data[f'Bus_{bus_id}_Voltage'] = voltage_data[bus_id][:len(hours)]
                
                voltage_df = pd.DataFrame(voltage_df_data)
                
                csv_buffer = io.StringIO()
                voltage_df.to_csv(csv_buffer, index=False)
                csv_sections.append(csv_buffer.getvalue())
                csv_sections.append("")
            
            # Thermal results
            if 'thermal_results' in loadflow_results:
                csv_sections.append("# THERMAL LOADING RESULTS (%)")
                
                thermal_data = loadflow_results['thermal_results']
                hours = loadflow_results['hours_analyzed']
                
                # Create thermal DataFrame
                thermal_df_data = {'Hour': hours}
                for line_id in sorted(thermal_data.keys()):
                    thermal_df_data[f'Line_{line_id}_Loading'] = [l * 100 for l in thermal_data[line_id][:len(hours)]]
                
                thermal_df = pd.DataFrame(thermal_df_data)
                
                csv_buffer = io.StringIO()
                thermal_df.to_csv(csv_buffer, index=False)
                csv_sections.append(csv_buffer.getvalue())
                csv_sections.append("")
            
            # Losses results
            if 'losses_results' in loadflow_results:
                csv_sections.append("# SYSTEM LOSSES (MW)")
                
                losses_data = loadflow_results['losses_results']
                hours = loadflow_results['hours_analyzed']
                
                losses_df = pd.DataFrame({
                    'Hour': hours,
                    'System_Losses_MW': losses_data[:len(hours)]
                })
                
                csv_buffer = io.StringIO()
                losses_df.to_csv(csv_buffer, index=False)
                csv_sections.append(csv_buffer.getvalue())
            
            return "\n".join(csv_sections)
            
        except Exception as e:
            return f"Error exporting load flow data: {str(e)}"
    
    def export_load_profile(self, load_profile, profile_name="Load_Profile"):
        """Export load profile to CSV with timestamps."""
        
        try:
            # Create timestamps for 8760 hours
            timestamps = pd.date_range(
                start='2024-01-01 00:00:00',
                periods=len(load_profile),
                freq='H'
            )
            
            profile_df = pd.DataFrame({
                'Timestamp': timestamps,
                'Hour_of_Year': range(len(load_profile)),
                'Load_pu': load_profile,
                'Month': timestamps.month,
                'Day': timestamps.day,
                'Hour': timestamps.hour,
                'Day_of_Week': timestamps.dayofweek
            })
            
            # Add header
            header_lines = [
                f"# Load Profile Export - {profile_name}",
                f"# Generated: {self.export_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                f"# Total Hours: {len(load_profile)}",
                f"# Peak Load: {np.max(load_profile):.4f} p.u.",
                f"# Minimum Load: {np.min(load_profile):.4f} p.u.",
                f"# Average Load: {np.mean(load_profile):.4f} p.u.",
                f"# Load Factor: {np.mean(load_profile)/np.max(load_profile):.3f}",
                ""
            ]
            
            csv_buffer = io.StringIO()
            
            # Write header
            for line in header_lines:
                csv_buffer.write(line + "\n")
            
            # Write data
            profile_df.to_csv(csv_buffer, index=False)
            
            return csv_buffer.getvalue()
            
        except Exception as e:
            return f"Error exporting load profile: {str(e)}"
    
    def export_network_model(self, network_model):
        """Export PandaPower network model to JSON format."""
        
        try:
            if network_model is None:
                return "No network model available"
            
            # Extract network data
            network_data = {
                'metadata': {
                    'name': network_model.name,
                    'export_timestamp': self.export_timestamp.isoformat(),
                    'network_type': 'Distribution Feeder',
                    'base_frequency': getattr(network_model, 'f_hz', 60.0)
                },
                'buses': network_model.bus.to_dict('records'),
                'lines': network_model.line.to_dict('records'),
                'loads': network_model.load.to_dict('records'),
                'external_grid': network_model.ext_grid.to_dict('records'),
                'transformers': network_model.trafo.to_dict('records') if len(network_model.trafo) > 0 else [],
                'generators': network_model.sgen.to_dict('records') if len(network_model.sgen) > 0 else [],
                'shunts': network_model.shunt.to_dict('records') if len(network_model.shunt) > 0 else []
            }
            
            # Convert to JSON string
            return json.dumps(network_data, indent=2, default=str)
            
        except Exception as e:
            return f"Error exporting network model: {str(e)}"
    
    def create_ieee_report(self, feeder_config, forecast_results, loadflow_results):
        """Create IEEE-style technical report."""
        
        try:
            report = []
            
            # IEEE Paper Header
            report.append("TIME SERIES LOAD FLOW ANALYSIS FOR DISTRIBUTION FEEDER")
            report.append("INTEGRATED RESOURCE PLANNING")
            report.append("")
            report.append("ABSTRACT")
            report.append("-" * 20)
            report.append("This report presents a comprehensive time series load flow analysis")
            report.append("for distribution feeder integrated resource planning. The analysis")
            report.append("includes 10-year load forecasting with EV adoption scenarios and")
            report.append("detailed network performance evaluation using PandaPower.")
            report.append("")
            
            # System Description
            report.append("I. SYSTEM DESCRIPTION")
            report.append("-" * 30)
            feeder_name = feeder_config.get('name', 'Unknown')
            report.append(f"The {feeder_name} distribution feeder operates at {feeder_config.get('base_voltage', 0):.1f} kV")
            report.append(f"with a rated capacity of {feeder_config.get('rated_capacity', 0):.1f} MVA.")
            report.append(f"The feeder serves {feeder_config.get('customers', 0):,} customers over")
            report.append(f"{feeder_config.get('length_km', 0):.1f} km with a base load of {feeder_config.get('base_load_mw', 0):.1f} MW.")
            report.append("")
            
            # Load Forecasting Results
            if forecast_results:
                report.append("II. LOAD FORECASTING RESULTS")
                report.append("-" * 35)
                
                scenario = forecast_results.get('scenario', {})
                report.append(f"Scenario Analysis: {scenario.get('name', 'Unknown')}")
                report.append(f"Base Load Growth Rate: {scenario.get('base_load_growth', 0)*100:.1f}% per year")
                report.append(f"EV Growth Rate: {scenario.get('ev_growth_rate', 0)*100:.1f}% per year")
                
                initial_peak = forecast_results['peak_demands'][0]
                final_peak = forecast_results['peak_demands'][-1]
                peak_cagr = forecast_results.get('growth_rates', {}).get('peak_demand_cagr', 0) * 100
                
                report.append(f"Peak Demand Growth: {initial_peak:.1f} MW → {final_peak:.1f} MW")
                report.append(f"Compound Annual Growth Rate: {peak_cagr:.2f}%")
                
                final_ev = forecast_results['ev_penetration'][-1] * 100
                report.append(f"Final EV Penetration: {final_ev:.1f}%")
                report.append("")
            
            # Network Analysis Results
            if loadflow_results:
                report.append("III. NETWORK PERFORMANCE ANALYSIS")
                report.append("-" * 40)
                
                hours_analyzed = len(loadflow_results.get('hours_analyzed', []))
                report.append(f"Time Series Analysis Period: {hours_analyzed} hours")
                
                voltage_violations = loadflow_results.get('voltage_violations', 0)
                thermal_violations = loadflow_results.get('thermal_violations', 0)
                
                report.append(f"Voltage Violations: {voltage_violations}")
                report.append(f"Thermal Violations: {thermal_violations}")
                
                if 'voltage_results' in loadflow_results:
                    voltage_data = loadflow_results['voltage_results']
                    all_voltages = []
                    for bus_voltages in voltage_data.values():
                        all_voltages.extend(bus_voltages)
                    
                    min_voltage = min(all_voltages)
                    max_voltage = max(all_voltages)
                    avg_voltage = sum(all_voltages) / len(all_voltages)
                    
                    report.append(f"Voltage Statistics:")
                    report.append(f"  Minimum: {min_voltage:.4f} p.u.")
                    report.append(f"  Maximum: {max_voltage:.4f} p.u.")
                    report.append(f"  Average: {avg_voltage:.4f} p.u.")
                
                total_losses = loadflow_results.get('total_losses_mwh', 0)
                if total_losses > 0:
                    loss_rate = (total_losses / (feeder_config.get('base_load_mw', 1) * hours_analyzed)) * 100
                    report.append(f"System Loss Rate: {loss_rate:.2f}%")
                
                report.append("")
            
            # Conclusions and Recommendations
            report.append("IV. CONCLUSIONS AND RECOMMENDATIONS")
            report.append("-" * 45)
            
            # Generate technical conclusions
            conclusions = []
            
            if forecast_results:
                final_util = (forecast_results['peak_demands'][-1] / 
                             feeder_config.get('rated_capacity', 1)) * 100
                
                if final_util > 100:
                    conclusions.append("The analysis indicates capacity constraints within the forecast period.")
                    conclusions.append("Immediate capacity expansion planning is recommended.")
                elif final_util > 80:
                    conclusions.append("The feeder will experience elevated loading within 10 years.")
                    conclusions.append("Proactive capacity planning is recommended.")
                
                final_ev = forecast_results['ev_penetration'][-1] * 100
                if final_ev > 50:
                    conclusions.append("High EV adoption will significantly impact load patterns.")
                    conclusions.append("EV-specific infrastructure upgrades should be considered.")
            
            if loadflow_results:
                if loadflow_results.get('voltage_violations', 0) > 0:
                    conclusions.append("Voltage violations indicate need for voltage regulation equipment.")
                
                if loadflow_results.get('thermal_violations', 0) > 0:
                    conclusions.append("Thermal violations require attention to prevent equipment damage.")
            
            if not conclusions:
                conclusions.append("The analysis indicates satisfactory system performance.")
                conclusions.append("Continued monitoring and periodic updates are recommended.")
            
            for conclusion in conclusions:
                report.append(conclusion)
            
            report.append("")
            report.append("V. REFERENCES")
            report.append("-" * 20)
            report.append("[1] PandaPower Documentation, https://pandapower.readthedocs.io/")
            report.append("[2] IEEE Standard 1547, Interconnection and Interoperability of")
            report.append("    Distributed Energy Resources")
            report.append("[3] Electric Power Research Institute (EPRI), Distribution System")
            report.append("    Planning Guidelines")
            
            return "\n".join(report)
            
        except Exception as e:
            return f"Error creating IEEE report: {str(e)}"
    
    def export_all_data(self, feeder_config, forecast_results, loadflow_results, load_profile):
        """Export all data in a comprehensive package."""
        
        try:
            export_package = {}
            
            # Executive Summary
            export_package['executive_summary'] = self.create_executive_summary(
                feeder_config, forecast_results, loadflow_results
            )
            
            # Technical Report
            export_package['technical_report'] = self.create_ieee_report(
                feeder_config, forecast_results, loadflow_results
            )
            
            # Data exports
            if forecast_results:
                export_package['forecast_data_csv'] = self.export_forecast_data(forecast_results)
            
            if loadflow_results:
                export_package['loadflow_data_csv'] = self.export_loadflow_data(loadflow_results)
            
            if load_profile is not None:
                export_package['load_profile_csv'] = self.export_load_profile(
                    load_profile, f"Profile_{feeder_config.get('name', 'Unknown')}"
                )
            
            # Metadata
            export_package['metadata'] = {
                'export_timestamp': self.export_timestamp.isoformat(),
                'feeder_name': feeder_config.get('name', 'Unknown'),
                'analysis_type': 'Integrated Resource Planning',
                'software_version': 'Electric Utility Planning System v1.0'
            }
            
            return export_package
            
        except Exception as e:
            return {'error': f"Error creating export package: {str(e)}"}

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

class LoadForecaster:
    """
    Advanced load forecasting engine for electric utility planning.
    Provides 10-year load forecasts with multiple scenarios including
    EV adoption, distributed generation, and climate impacts.
    """
    
    def __init__(self):
        self.base_profile = None
        self.forecast_years = 10
        
    def generate_forecast(self, feeder_config, base_load_profile, forecast_params):
        """
        Generate comprehensive 10-year load forecast.
        
        Args:
            feeder_config: Dictionary with feeder configuration
            base_load_profile: 8760-hour base load profile (p.u.)
            forecast_params: Dictionary with forecasting parameters
            
        Returns:
            Dictionary with forecast results including yearly projections
        """
        try:
            self.base_profile = base_load_profile
            scenario_name = forecast_params.get('scenario_name', 'Custom')
            
            # Initialize forecast years
            current_year = datetime.now().year
            forecast_years = list(range(current_year, current_year + self.forecast_years + 1))
            
            # Extract forecasting parameters
            base_load_growth = forecast_params.get('base_load_growth', 0.025)
            ev_growth_rate = forecast_params.get('ev_growth_rate', 0.15)
            max_ev_penetration = forecast_params.get('max_ev_penetration', 0.60)
            solar_growth_rate = forecast_params.get('solar_growth_rate', 0.12)
            efficiency_improvement = forecast_params.get('efficiency_improvement', 0.015)
            electrification_rate = forecast_params.get('electrification_rate', 0.03)
            climate_change_factor = forecast_params.get('climate_change_factor', 0.003)
            peak_growth_multiplier = forecast_params.get('peak_growth_multiplier', 1.3)
            economic_factor = forecast_params.get('economic_factor', 1.0)
            
            # Initialize tracking arrays
            peak_demands = []
            annual_energy = []
            ev_penetration = []
            solar_capacity = []
            hourly_profiles = []
            load_factors = []
            
            # Get initial values
            initial_ev_penetration = feeder_config.get('ev_penetration', 0.15)
            initial_solar_capacity = feeder_config.get('dg_capacity_mw', 2.1)
            base_load_mw = feeder_config.get('base_load_mw', 8.5)
            
            # Generate yearly forecasts
            for year_idx, year in enumerate(forecast_years):
                # Calculate growth factors for this year
                years_elapsed = year_idx
                
                # Base load growth with economic factors
                base_growth_factor = (1 + base_load_growth * economic_factor) ** years_elapsed
                
                # EV penetration growth (S-curve adoption)
                ev_penetration_year = self._calculate_ev_penetration(
                    initial_ev_penetration,
                    ev_growth_rate,
                    max_ev_penetration,
                    years_elapsed
                )
                ev_penetration.append(ev_penetration_year)
                
                # Solar capacity growth
                solar_capacity_year = initial_solar_capacity * (1 + solar_growth_rate) ** years_elapsed
                solar_capacity.append(solar_capacity_year)
                
                # Energy efficiency improvements (reduces base load)
                efficiency_factor = (1 - efficiency_improvement) ** years_elapsed
                
                # Electrification effects (increases load)
                electrification_factor = (1 + electrification_rate) ** years_elapsed
                
                # Climate change impacts (primarily cooling load)
                climate_factor = (1 + climate_change_factor) ** years_elapsed
                
                # Generate load profile for this year
                yearly_profile = self._generate_yearly_profile(
                    base_load_profile,
                    base_growth_factor,
                    ev_penetration_year,
                    efficiency_factor,
                    electrification_factor,
                    climate_factor,
                    feeder_config,
                    forecast_params
                )
                
                # Apply distributed generation impacts
                yearly_profile = self._apply_dg_impacts(
                    yearly_profile,
                    solar_capacity_year,
                    base_load_mw,
                    year_idx
                )
                
                hourly_profiles.append(yearly_profile)
                
                # Calculate key metrics
                peak_demand_year = np.max(yearly_profile) * base_load_mw
                
                # Apply peak growth multiplier for infrastructure and coincidence factors
                if year_idx > 0:
                    peak_demand_year *= (1 + (peak_growth_multiplier - 1) * 
                                       (peak_demand_year / peak_demands[0] - 1))
                
                peak_demands.append(peak_demand_year)
                
                # Annual energy calculation
                annual_energy_year = np.sum(yearly_profile) * base_load_mw
                annual_energy.append(annual_energy_year)
                
                # Load factor
                load_factor = annual_energy_year / (peak_demand_year * 8760)
                load_factors.append(load_factor)
            
            # Compile comprehensive results
            forecast_results = {
                'scenario': {
                    'name': scenario_name,
                    'base_load_growth': base_load_growth,
                    'ev_growth_rate': ev_growth_rate,
                    'max_ev_penetration': max_ev_penetration,
                    'solar_growth_rate': solar_growth_rate,
                    'efficiency_improvement': efficiency_improvement,
                    'electrification_rate': electrification_rate
                },
                'years': forecast_years,
                'peak_demands': peak_demands,
                'annual_energy': annual_energy,
                'ev_penetration': ev_penetration,
                'solar_capacity': solar_capacity,
                'load_factors': load_factors,
                'hourly_profiles': hourly_profiles,
                'base_case': {
                    'peak_demand': peak_demands[0],
                    'annual_energy': annual_energy[0],
                    'load_factor': load_factors[0]
                },
                'ten_year_case': {
                    'peak_demand': peak_demands[-1],
                    'annual_energy': annual_energy[-1],
                    'load_factor': load_factors[-1]
                },
                'growth_rates': {
                    'peak_demand_cagr': self._calculate_cagr(peak_demands[0], peak_demands[-1], self.forecast_years),
                    'energy_cagr': self._calculate_cagr(annual_energy[0], annual_energy[-1], self.forecast_years)
                },
                'metadata': {
                    'generated_date': datetime.now().isoformat(),
                    'feeder_name': feeder_config.get('name', 'Unknown'),
                    'forecast_horizon': self.forecast_years
                }
            }
            
            return forecast_results
            
        except Exception as e:
            raise Exception(f"Error generating forecast: {str(e)}")
    
    def _calculate_ev_penetration(self, initial_penetration, growth_rate, max_penetration, years_elapsed):
        """Calculate EV penetration using S-curve adoption model."""
        if years_elapsed == 0:
            return initial_penetration
        
        # S-curve parameters
        k = growth_rate * 10  # Steepness factor
        midpoint = 5  # Year at which adoption is halfway to maximum
        
        # S-curve formula
        penetration = max_penetration / (1 + np.exp(-k * (years_elapsed - midpoint)))
        
        # Ensure we don't go below initial penetration
        return max(initial_penetration, penetration)
    
    def _generate_yearly_profile(self, base_profile, base_growth_factor, ev_penetration, 
                                efficiency_factor, electrification_factor, climate_factor,
                                feeder_config, forecast_params):
        """Generate load profile for a specific year with all growth factors applied."""
        
        # Start with base profile scaled by base growth
        yearly_profile = base_profile * base_growth_factor
        
        # Apply efficiency improvements (reduces load)
        yearly_profile *= efficiency_factor
        
        # Apply electrification effects (increases load)
        yearly_profile *= electrification_factor
        
        # Add EV charging load profile
        ev_profile = self._generate_ev_profile(ev_penetration, feeder_config, forecast_params)
        yearly_profile += ev_profile
        
        # Apply climate change impacts (primarily affects cooling season)
        yearly_profile = self._apply_climate_impacts(yearly_profile, climate_factor)
        
        # Apply demand response and smart grid impacts
        yearly_profile = self._apply_smart_grid_impacts(yearly_profile, forecast_params, ev_penetration)
        
        return yearly_profile
    
    def _generate_ev_profile(self, ev_penetration, feeder_config, forecast_params):
        """Generate EV charging load profile based on penetration and charging patterns."""
        
        customers = feeder_config.get('customers', 850)
        ev_config = feeder_config.get('ev_config', {})
        avg_ev_power = ev_config.get('avg_power', 7.2)  # kW
        smart_charging_adoption = forecast_params.get('smart_charging_adoption', 0.3)
        
        # Calculate number of EVs
        num_evs = customers * ev_penetration
        
        # Generate base EV charging pattern (typical unmanaged charging)
        ev_profile = np.zeros(8760)
        
        for hour in range(8760):
            hour_of_day = hour % 24
            day_of_year = hour // 24
            day_of_week = day_of_year % 7
            
            # Base charging probability by time of day
            if 18 <= hour_of_day <= 23:  # Evening peak
                charging_probability = 0.7
            elif 6 <= hour_of_day <= 8:  # Morning charging
                charging_probability = 0.2
            elif 9 <= hour_of_day <= 17 and day_of_week < 5:  # Workplace charging weekdays
                charging_probability = 0.3
            else:
                charging_probability = 0.1
            
            # Seasonal adjustments (higher winter charging)
            if 330 <= day_of_year <= 365 or day_of_year <= 60:  # Winter
                charging_probability *= 1.2
            elif 150 <= day_of_year <= 240:  # Summer
                charging_probability *= 0.9
            
            # Calculate EV load for this hour
            ev_load_hour = num_evs * charging_probability * avg_ev_power / 1000  # Convert to MW
            
            # Apply smart charging reduction (shifts load to off-peak)
            if smart_charging_adoption > 0 and 17 <= hour_of_day <= 21:  # Peak hours
                reduction_factor = 1 - (smart_charging_adoption * 0.6)  # 60% reduction with smart charging
                ev_load_hour *= reduction_factor
            elif smart_charging_adoption > 0 and 1 <= hour_of_day <= 5:  # Off-peak hours
                increase_factor = 1 + (smart_charging_adoption * 0.3)  # 30% increase in off-peak
                ev_load_hour *= increase_factor
            
            ev_profile[hour] = ev_load_hour
        
        # Normalize to per-unit based on base load
        base_load_mw = feeder_config.get('base_load_mw', 8.5)
        ev_profile_pu = ev_profile / base_load_mw
        
        return ev_profile_pu
    
    def _apply_dg_impacts(self, load_profile, solar_capacity_mw, base_load_mw, year_idx):
        """Apply distributed generation impacts to load profile."""
        
        # Generate solar profile
        solar_profile = self._generate_solar_profile(solar_capacity_mw)
        
        # Convert to per-unit
        solar_profile_pu = solar_profile / base_load_mw
        
        # Net load = Gross load - Solar generation
        net_profile = load_profile - solar_profile_pu
        
        # Ensure non-negative (excess generation is exported)
        net_profile = np.maximum(net_profile, 0.1 * np.ones_like(net_profile))
        
        return net_profile
    
    def _generate_solar_profile(self, solar_capacity_mw):
        """Generate annual solar generation profile."""
        
        solar_profile = np.zeros(8760)
        
        for hour in range(8760):
            day_of_year = hour // 24
            hour_of_day = hour % 24
            
            # Solar irradiance model (simplified)
            if 6 <= hour_of_day <= 18:  # Daylight hours
                # Daily solar curve
                solar_hour_factor = np.sin(np.pi * (hour_of_day - 6) / 12)
                
                # Seasonal variation
                seasonal_factor = 0.7 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
                
                # Weather variability (simplified)
                weather_factor = 0.8 + 0.2 * np.random.random()
                
                solar_output = solar_capacity_mw * solar_hour_factor * seasonal_factor * weather_factor
            else:
                solar_output = 0
            
            solar_profile[hour] = max(0, solar_output)
        
        return solar_profile
    
    def _apply_climate_impacts(self, profile, climate_factor):
        """Apply climate change impacts to load profile."""
        
        climate_adjusted_profile = np.zeros_like(profile)
        
        for hour in range(8760):
            day_of_year = hour // 24
            hour_of_day = hour % 24
            
            # Climate change primarily affects cooling load (summer months)
            if 150 <= day_of_year <= 240:  # Summer months
                if 12 <= hour_of_day <= 20:  # Peak cooling hours
                    climate_adjustment = (climate_factor - 1) * 1.5  # 50% higher impact during peak cooling
                else:
                    climate_adjustment = (climate_factor - 1) * 0.8
            else:
                climate_adjustment = (climate_factor - 1) * 0.2  # Minimal winter impact
            
            climate_adjusted_profile[hour] = profile[hour] * (1 + climate_adjustment)
        
        return climate_adjusted_profile
    
    def _apply_smart_grid_impacts(self, profile, forecast_params, ev_penetration):
        """Apply smart grid and demand response impacts."""
        
        demand_response_penetration = forecast_params.get('demand_response_penetration', 0.15) / 100
        time_of_use_adoption = forecast_params.get('time_of_use_adoption', 0.25) / 100
        
        smart_grid_profile = np.zeros_like(profile)
        
        for hour in range(8760):
            hour_of_day = hour % 24
            
            # Demand response impacts (peak shaving)
            if demand_response_penetration > 0 and 16 <= hour_of_day <= 20:  # Peak hours
                dr_reduction = demand_response_penetration * 0.15  # 15% load reduction capability
                dr_factor = 1 - dr_reduction
            else:
                dr_factor = 1
            
            # Time-of-use rate impacts (load shifting)
            if time_of_use_adoption > 0:
                if 17 <= hour_of_day <= 21:  # Peak rate hours
                    tou_reduction = time_of_use_adoption * 0.08  # 8% reduction
                    tou_factor = 1 - tou_reduction
                elif 22 <= hour_of_day <= 6:  # Off-peak hours
                    tou_increase = time_of_use_adoption * 0.05  # 5% increase
                    tou_factor = 1 + tou_increase
                else:
                    tou_factor = 1
            else:
                tou_factor = 1
            
            smart_grid_profile[hour] = profile[hour] * dr_factor * tou_factor
        
        return smart_grid_profile
    
    def _calculate_cagr(self, initial_value, final_value, years):
        """Calculate Compound Annual Growth Rate."""
        if initial_value <= 0 or final_value <= 0 or years <= 0:
            return 0
        return ((final_value / initial_value) ** (1 / years)) - 1
    
    def validate_forecast_parameters(self, params):
        """Validate forecast parameters and provide warnings for unrealistic values."""
        
        warnings = []
        
        # Check growth rates
        if params.get('base_load_growth', 0) > 0.08:
            warnings.append("Base load growth rate >8% is unusually high")
        
        if params.get('ev_growth_rate', 0) > 0.5:
            warnings.append("EV growth rate >50% may be unrealistic")
        
        if params.get('max_ev_penetration', 0) > 1.0:
            warnings.append("EV penetration cannot exceed 100%")
        
        # Check consistency
        if params.get('efficiency_improvement', 0) > params.get('base_load_growth', 0):
            warnings.append("Efficiency improvement exceeds load growth - net decline expected")
        
        return warnings
    
    def generate_scenario_comparison(self, feeder_config, base_profile, scenarios):
        """Generate multiple scenarios for comparison."""
        
        results = {}
        
        for scenario_name, scenario_params in scenarios.items():
            try:
                results[scenario_name] = self.generate_forecast(
                    feeder_config, 
                    base_profile, 
                    scenario_params
                )
            except Exception as e:
                results[scenario_name] = {'error': str(e)}
        
        return results

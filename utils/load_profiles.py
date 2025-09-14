import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class LoadProfileGenerator:
    """
    Comprehensive load profile generator for electric utility applications.
    Generates realistic 8760-hour load profiles for different customer types
    and use cases in distribution system planning.
    """
    
    def __init__(self):
        self.hours_per_year = 8760
        
    def generate_profile(self, profile_type="Residential", seasonality=0.3, 
                        daily_variation=0.4, weekly_pattern=True, 
                        noise_level=0.05, peak_hour=19, min_load_ratio=0.3):
        """
        Generate a comprehensive 8760-hour load profile.
        
        Args:
            profile_type: Type of load profile to generate
            seasonality: Strength of seasonal variation (0-1)
            daily_variation: Strength of daily variation (0-1)
            weekly_pattern: Whether to include weekly patterns
            noise_level: Amount of random variation (0-0.2)
            peak_hour: Hour of daily peak (0-23)
            min_load_ratio: Minimum load as ratio of peak (0-1)
            
        Returns:
            numpy.array: 8760-hour load profile in per-unit
        """
        
        try:
            # Initialize profile array
            profile = np.zeros(self.hours_per_year)
            
            # Generate base patterns
            for hour in range(self.hours_per_year):
                day_of_year = hour // 24
                hour_of_day = hour % 24
                day_of_week = day_of_year % 7
                
                # Base load level
                base_load = 0.6
                
                # Seasonal component
                seasonal_component = self._get_seasonal_component(
                    day_of_year, profile_type, seasonality
                )
                
                # Daily component
                daily_component = self._get_daily_component(
                    hour_of_day, profile_type, daily_variation, peak_hour
                )
                
                # Weekly component
                weekly_component = self._get_weekly_component(
                    day_of_week, hour_of_day, profile_type
                ) if weekly_pattern else 1.0
                
                # Combine components
                load_value = (base_load + seasonal_component + daily_component) * weekly_component
                
                # Add random noise
                if noise_level > 0:
                    noise = np.random.normal(0, noise_level)
                    load_value += noise
                
                # Apply constraints
                load_value = max(min_load_ratio, min(1.0, load_value))
                
                profile[hour] = load_value
            
            # Normalize to ensure peak is 1.0
            profile = profile / np.max(profile)
            
            # Ensure minimum load ratio is respected
            profile = np.maximum(profile, min_load_ratio)
            
            return profile
            
        except Exception as e:
            raise Exception(f"Error generating load profile: {str(e)}")
    
    def _get_seasonal_component(self, day_of_year, profile_type, seasonality):
        """Calculate seasonal load variation."""
        
        if seasonality == 0:
            return 0
        
        # Basic seasonal patterns
        if profile_type in ["Residential", "Mixed"]:
            # Peak in summer (cooling) and winter (heating)
            summer_peak = 0.3 * np.sin(2 * np.pi * (day_of_year - 172) / 365)  # Summer peak around day 172 (June 21)
            winter_peak = 0.2 * np.sin(2 * np.pi * (day_of_year - 355) / 365)  # Winter peak around day 355 (Dec 21)
            seasonal = (summer_peak + winter_peak) * seasonality
            
        elif profile_type == "Commercial":
            # Peak in summer (cooling dominant)
            seasonal = 0.25 * np.sin(2 * np.pi * (day_of_year - 172) / 365) * seasonality
            
        elif profile_type == "Industrial":
            # More stable throughout year, slight summer increase
            seasonal = 0.15 * np.sin(2 * np.pi * (day_of_year - 172) / 365) * seasonality
            
        else:  # Custom or other
            seasonal = 0.2 * np.sin(2 * np.pi * (day_of_year - 172) / 365) * seasonality
        
        return seasonal
    
    def _get_daily_component(self, hour_of_day, profile_type, daily_variation, peak_hour):
        """Calculate daily load variation."""
        
        if daily_variation == 0:
            return 0
        
        if profile_type == "Residential":
            # Residential: morning and evening peaks
            if 6 <= hour_of_day <= 9:  # Morning peak
                daily = 0.2 * np.sin(np.pi * (hour_of_day - 6) / 3)
            elif 17 <= hour_of_day <= 22:  # Evening peak
                daily = 0.4 * np.sin(np.pi * (hour_of_day - 17) / 5)
            elif 0 <= hour_of_day <= 5:  # Night minimum
                daily = -0.3
            else:  # Daytime base
                daily = -0.1
                
        elif profile_type == "Commercial":
            # Commercial: single daytime peak
            if 8 <= hour_of_day <= 18:  # Business hours
                daily = 0.4 * np.sin(np.pi * (hour_of_day - 8) / 10)
            else:  # Off hours
                daily = -0.3
                
        elif profile_type == "Industrial":
            # Industrial: relatively flat with slight daytime increase
            if 6 <= hour_of_day <= 22:  # Operating hours
                daily = 0.15 * np.sin(np.pi * (hour_of_day - 6) / 16)
            else:  # Night shift
                daily = -0.1
                
        else:  # Custom - use peak_hour parameter
            # Single peak around specified hour
            if abs(hour_of_day - peak_hour) <= 3:
                phase = np.pi * (hour_of_day - (peak_hour - 3)) / 6
                daily = 0.3 * np.sin(phase)
            else:
                daily = -0.2
        
        return daily * daily_variation
    
    def _get_weekly_component(self, day_of_week, hour_of_day, profile_type):
        """Calculate weekly load variation."""
        
        if profile_type == "Residential":
            # Residential: slightly higher on weekends
            if day_of_week >= 5:  # Weekend
                return 1.05
            else:  # Weekday
                return 1.0
                
        elif profile_type == "Commercial":
            # Commercial: much lower on weekends
            if day_of_week >= 5:  # Weekend
                if 10 <= hour_of_day <= 18:  # Retail hours
                    return 0.7
                else:
                    return 0.3
            else:  # Weekday
                return 1.0
                
        elif profile_type == "Industrial":
            # Industrial: varies by shift patterns
            if day_of_week >= 5:  # Weekend
                return 0.8  # Reduced weekend operations
            else:  # Weekday
                return 1.0
                
        else:  # Mixed or other
            if day_of_week >= 5:  # Weekend
                return 0.9
            else:
                return 1.0
    
    def get_template_profile(self, template_type):
        """Get predefined template load profiles for common feeder types."""
        
        templates = {
            "Typical Residential Feeder": {
                'profile_type': 'Residential',
                'seasonality': 0.35,
                'daily_variation': 0.5,
                'weekly_pattern': True,
                'noise_level': 0.08,
                'peak_hour': 19,
                'min_load_ratio': 0.25
            },
            
            "Commercial District": {
                'profile_type': 'Commercial',
                'seasonality': 0.4,
                'daily_variation': 0.6,
                'weekly_pattern': True,
                'noise_level': 0.05,
                'peak_hour': 14,
                'min_load_ratio': 0.2
            },
            
            "Industrial Complex": {
                'profile_type': 'Industrial',
                'seasonality': 0.2,
                'daily_variation': 0.3,
                'weekly_pattern': True,
                'noise_level': 0.04,
                'peak_hour': 15,
                'min_load_ratio': 0.4
            },
            
            "Rural Agricultural": {
                'profile_type': 'Mixed',
                'seasonality': 0.5,  # High seasonal variation for irrigation
                'daily_variation': 0.4,
                'weekly_pattern': False,
                'noise_level': 0.12,
                'peak_hour': 16,
                'min_load_ratio': 0.15
            },
            
            "Urban Mixed-Use": {
                'profile_type': 'Mixed',
                'seasonality': 0.3,
                'daily_variation': 0.45,
                'weekly_pattern': True,
                'noise_level': 0.06,
                'peak_hour': 18,
                'min_load_ratio': 0.3
            },
            
            "University Campus": {
                'profile_type': 'Commercial',
                'seasonality': 0.6,  # High seasonal variation due to academic calendar
                'daily_variation': 0.5,
                'weekly_pattern': True,
                'noise_level': 0.07,
                'peak_hour': 15,
                'min_load_ratio': 0.2
            }
        }
        
        if template_type in templates:
            params = templates[template_type]
            return self.generate_profile(**params)
        else:
            # Default template
            return self.generate_profile()
    
    def create_ev_charging_profile(self, ev_penetration, num_customers, charging_scenarios):
        """Create EV charging load profile based on adoption and charging patterns."""
        
        ev_profile = np.zeros(self.hours_per_year)
        num_evs = int(num_customers * ev_penetration)
        
        if num_evs == 0:
            return ev_profile
        
        # Charging parameters
        avg_charging_power = 7.2  # kW
        charging_efficiency = 0.9
        home_charging_pct = charging_scenarios.get('home_charging_pct', 80) / 100
        workplace_charging_pct = charging_scenarios.get('workplace_charging_pct', 15) / 100
        public_charging_pct = charging_scenarios.get('public_charging_pct', 5) / 100
        
        for hour in range(self.hours_per_year):
            hour_of_day = hour % 24
            day_of_week = (hour // 24) % 7
            
            # Home charging patterns
            if 18 <= hour_of_day <= 23:  # Evening peak
                home_charging_prob = 0.6
            elif 1 <= hour_of_day <= 6:  # Off-peak charging
                home_charging_prob = 0.3
            else:
                home_charging_prob = 0.1
            
            # Workplace charging (weekdays only)
            if day_of_week < 5 and 8 <= hour_of_day <= 17:
                workplace_charging_prob = 0.4
            else:
                workplace_charging_prob = 0.0
            
            # Public charging (distributed throughout day)
            public_charging_prob = 0.15
            
            # Calculate total EV load for this hour
            home_load = (num_evs * home_charging_pct * home_charging_prob * 
                        avg_charging_power * charging_efficiency / 1000)  # Convert to MW
            
            workplace_load = (num_evs * workplace_charging_pct * workplace_charging_prob * 
                            avg_charging_power * charging_efficiency / 1000)
            
            public_load = (num_evs * public_charging_pct * public_charging_prob * 
                          avg_charging_power * charging_efficiency / 1000)
            
            ev_profile[hour] = home_load + workplace_load + public_load
        
        return ev_profile
    
    def create_solar_generation_profile(self, capacity_mw, latitude=40.0):
        """Create solar PV generation profile for a given capacity."""
        
        solar_profile = np.zeros(self.hours_per_year)
        
        for hour in range(self.hours_per_year):
            day_of_year = hour // 24
            hour_of_day = hour % 24
            
            # Calculate solar irradiance (simplified model)
            if 6 <= hour_of_day <= 18:  # Daylight hours
                # Daily solar curve (sinusoidal)
                daily_factor = np.sin(np.pi * (hour_of_day - 6) / 12)
                
                # Seasonal variation based on latitude
                declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
                seasonal_factor = np.sin(np.radians(90 - abs(latitude - declination)))
                seasonal_factor = max(0.3, seasonal_factor)  # Minimum seasonal factor
                
                # Weather variation (simplified)
                weather_factor = 0.7 + 0.3 * np.random.random()
                
                # Calculate solar output
                solar_output = capacity_mw * daily_factor * seasonal_factor * weather_factor
                
            else:
                solar_output = 0.0
            
            solar_profile[hour] = max(0, solar_output)
        
        return solar_profile
    
    def analyze_profile_characteristics(self, profile):
        """Analyze load profile characteristics and provide statistics."""
        
        characteristics = {}
        
        # Basic statistics
        characteristics['peak_load'] = np.max(profile)
        characteristics['min_load'] = np.min(profile)
        characteristics['avg_load'] = np.mean(profile)
        characteristics['load_factor'] = np.mean(profile) / np.max(profile)
        characteristics['std_deviation'] = np.std(profile)
        characteristics['coefficient_of_variation'] = np.std(profile) / np.mean(profile)
        
        # Peak analysis
        top_1_percent = np.percentile(profile, 99)
        top_5_percent = np.percentile(profile, 95)
        top_10_percent = np.percentile(profile, 90)
        
        characteristics['top_1_percent_threshold'] = top_1_percent
        characteristics['top_5_percent_threshold'] = top_5_percent
        characteristics['top_10_percent_threshold'] = top_10_percent
        
        characteristics['hours_above_99th'] = np.sum(profile >= top_1_percent)
        characteristics['hours_above_95th'] = np.sum(profile >= top_5_percent)
        characteristics['hours_above_90th'] = np.sum(profile >= top_10_percent)
        
        # Seasonal analysis
        monthly_peaks = []
        monthly_averages = []
        monthly_load_factors = []
        
        days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        
        for month in range(12):
            start_day = sum(days_in_months[:month])
            end_day = start_day + days_in_months[month]
            start_hour = start_day * 24
            end_hour = min(end_day * 24, self.hours_per_year)
            
            month_profile = profile[start_hour:end_hour]
            
            monthly_peaks.append(np.max(month_profile))
            monthly_averages.append(np.mean(month_profile))
            monthly_load_factors.append(np.mean(month_profile) / np.max(month_profile))
        
        characteristics['monthly_peaks'] = monthly_peaks
        characteristics['monthly_averages'] = monthly_averages
        characteristics['monthly_load_factors'] = monthly_load_factors
        characteristics['peak_month'] = np.argmax(monthly_peaks) + 1
        characteristics['min_month'] = np.argmin(monthly_peaks) + 1
        
        # Daily patterns
        weekday_pattern = np.zeros(24)
        weekend_pattern = np.zeros(24)
        weekday_count = 0
        weekend_count = 0
        
        for hour in range(self.hours_per_year):
            day_of_year = hour // 24
            hour_of_day = hour % 24
            day_of_week = day_of_year % 7
            
            if day_of_week < 5:  # Weekday
                weekday_pattern[hour_of_day] += profile[hour]
                weekday_count += 1
            else:  # Weekend
                weekend_pattern[hour_of_day] += profile[hour]
                weekend_count += 1
        
        if weekday_count > 0:
            weekday_pattern /= (weekday_count / 24)
        if weekend_count > 0:
            weekend_pattern /= (weekend_count / 24)
        
        characteristics['weekday_pattern'] = weekday_pattern.tolist()
        characteristics['weekend_pattern'] = weekend_pattern.tolist()
        characteristics['peak_hour_weekday'] = np.argmax(weekday_pattern)
        characteristics['peak_hour_weekend'] = np.argmax(weekend_pattern)
        
        return characteristics
    
    def validate_profile(self, profile):
        """Validate load profile for common issues."""
        
        issues = []
        warnings = []
        
        # Check basic requirements
        if len(profile) != self.hours_per_year:
            issues.append(f"Profile length is {len(profile)}, expected {self.hours_per_year}")
        
        if np.any(profile < 0):
            issues.append("Profile contains negative values")
        
        if np.any(np.isnan(profile)):
            issues.append("Profile contains NaN values")
        
        if np.any(np.isinf(profile)):
            issues.append("Profile contains infinite values")
        
        # Check for reasonable characteristics
        if len(profile) == self.hours_per_year:
            load_factor = np.mean(profile) / np.max(profile)
            
            if load_factor < 0.2:
                warnings.append(f"Very low load factor ({load_factor:.3f}) - may indicate unrealistic peaking")
            
            if load_factor > 0.9:
                warnings.append(f"Very high load factor ({load_factor:.3f}) - may indicate insufficient variation")
            
            # Check for excessive variation
            cv = np.std(profile) / np.mean(profile)
            if cv > 0.8:
                warnings.append(f"High coefficient of variation ({cv:.3f}) - may be too variable")
            
            # Check for unrealistic patterns
            peak_hours = np.sum(profile >= np.percentile(profile, 95))
            if peak_hours > 500:  # More than ~2 weeks at 95th percentile
                warnings.append("Excessive number of peak hours - may need adjustment")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }
    
    def smooth_profile(self, profile, window_size=3):
        """Apply smoothing to reduce noise in load profile."""
        
        if window_size <= 1:
            return profile
        
        # Apply moving average smoothing
        smoothed = np.convolve(profile, np.ones(window_size)/window_size, mode='same')
        
        # Handle edge effects
        half_window = window_size // 2
        smoothed[:half_window] = profile[:half_window]
        smoothed[-half_window:] = profile[-half_window:]
        
        return smoothed
    
    def scale_profile_to_peak(self, profile, target_peak):
        """Scale profile to achieve specific peak value."""
        
        current_peak = np.max(profile)
        if current_peak > 0:
            scaling_factor = target_peak / current_peak
            return profile * scaling_factor
        else:
            return profile

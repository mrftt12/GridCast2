import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FeederSegment:
    """Represents a segment of a distribution feeder with electrical characteristics."""
    
    segment_id: str
    from_bus: int
    to_bus: int
    length_km: float
    conductor_type: str
    voltage_kv: float
    customers: int
    load_mw: float
    load_mvar: float
    resistance_ohm_per_km: float = 0.25
    reactance_ohm_per_km: float = 0.35
    susceptance_us_per_km: float = 10.0
    thermal_rating_a: float = 300.0

@dataclass
class LoadCharacteristics:
    """Represents load characteristics for different customer types."""
    
    residential_pct: float = 65.0
    commercial_pct: float = 25.0
    industrial_pct: float = 10.0
    power_factor: float = 0.9
    diversity_factor: float = 0.7
    load_density_kw_per_customer: float = 10.0
    peak_coincidence_factor: float = 0.8

class FeederModel:
    """
    Comprehensive electrical distribution feeder model.
    Provides detailed modeling of feeder electrical characteristics,
    load allocation, and network topology for utility planning applications.
    """
    
    def __init__(self, feeder_config: Dict):
        """
        Initialize feeder model with configuration parameters.
        
        Args:
            feeder_config: Dictionary containing feeder configuration
        """
        self.config = feeder_config
        self.segments = []
        self.load_characteristics = LoadCharacteristics()
        self.network_topology = "radial"
        self.voltage_regulation_equipment = []
        self.protection_equipment = []
        
        # Extract basic parameters
        self.name = feeder_config.get('name', 'Feeder_001')
        self.base_voltage_kv = feeder_config.get('base_voltage', 12.47)
        self.rated_capacity_mva = feeder_config.get('rated_capacity', 15.0)
        self.length_km = feeder_config.get('length_km', 5.2)
        self.total_customers = feeder_config.get('customers', 850)
        self.base_load_mw = feeder_config.get('base_load_mw', 8.5)
        self.dg_capacity_mw = feeder_config.get('dg_capacity_mw', 2.1)
        
        # Initialize feeder model
        self._initialize_load_characteristics()
        self._create_feeder_segments()
        self._allocate_loads()
        
    def _initialize_load_characteristics(self):
        """Initialize load characteristics based on feeder configuration."""
        
        # Extract load mix if available
        load_mix = self.config.get('load_mix', {})
        
        self.load_characteristics.residential_pct = load_mix.get('residential', 65.0)
        self.load_characteristics.commercial_pct = load_mix.get('commercial', 25.0)
        self.load_characteristics.industrial_pct = load_mix.get('industrial', 10.0)
        
        # Calculate load density
        if self.total_customers > 0:
            self.load_characteristics.load_density_kw_per_customer = (
                self.base_load_mw * 1000 / self.total_customers
            )
        
        # Adjust power factor based on load mix
        residential_pf = 0.92
        commercial_pf = 0.88
        industrial_pf = 0.85
        
        weighted_pf = (
            (self.load_characteristics.residential_pct / 100) * residential_pf +
            (self.load_characteristics.commercial_pct / 100) * commercial_pf +
            (self.load_characteristics.industrial_pct / 100) * industrial_pf
        )
        
        self.load_characteristics.power_factor = weighted_pf
    
    def _create_feeder_segments(self):
        """Create feeder segments based on network topology."""
        
        # Determine number of segments based on feeder length and customer count
        if self.length_km <= 2:
            num_segments = 3
        elif self.length_km <= 5:
            num_segments = 5
        elif self.length_km <= 10:
            num_segments = 8
        else:
            num_segments = 12
        
        # Adjust based on customer density
        customer_density = self.total_customers / self.length_km
        if customer_density > 200:  # High density urban
            num_segments = max(num_segments, int(self.length_km * 2))
        elif customer_density < 50:  # Low density rural
            num_segments = min(num_segments, max(3, int(self.length_km)))
        
        # Create segments
        avg_segment_length = self.length_km / num_segments
        
        for i in range(num_segments):
            # Variable segment lengths for more realistic model
            if i == 0:  # Main feeder segment
                segment_length = avg_segment_length * 1.2
            elif i == num_segments - 1:  # End segment
                segment_length = avg_segment_length * 0.8
            else:
                segment_length = avg_segment_length * (0.8 + 0.4 * np.random.random())
            
            # Determine conductor type based on position and loading
            conductor_type = self._determine_conductor_type(i, num_segments)
            
            # Create segment
            segment = FeederSegment(
                segment_id=f"{self.name}_Seg_{i+1}",
                from_bus=i,
                to_bus=i+1,
                length_km=segment_length,
                conductor_type=conductor_type,
                voltage_kv=self.base_voltage_kv,
                customers=0,  # Will be allocated later
                load_mw=0.0,  # Will be allocated later
                load_mvar=0.0,  # Will be allocated later
                **self._get_conductor_parameters(conductor_type)
            )
            
            self.segments.append(segment)
    
    def _determine_conductor_type(self, segment_index: int, total_segments: int) -> str:
        """Determine conductor type based on segment position and loading."""
        
        # Main feeder segments use larger conductors
        if segment_index < total_segments * 0.3:  # First 30% of segments
            if self.base_load_mw > 10:
                return "336_ACSR"  # Large conductor for high load
            else:
                return "4/0_ACSR"  # Medium conductor
        
        elif segment_index < total_segments * 0.7:  # Middle segments
            return "2/0_ACSR"  # Standard distribution conductor
        
        else:  # End segments
            return "1/0_ACSR"  # Smaller conductor for light loads
    
    def _get_conductor_parameters(self, conductor_type: str) -> Dict:
        """Get electrical parameters for different conductor types."""
        
        conductor_data = {
            "336_ACSR": {
                "resistance_ohm_per_km": 0.171,
                "reactance_ohm_per_km": 0.415,
                "susceptance_us_per_km": 11.5,
                "thermal_rating_a": 530
            },
            "4/0_ACSR": {
                "resistance_ohm_per_km": 0.272,
                "reactance_ohm_per_km": 0.447,
                "susceptance_us_per_km": 10.8,
                "thermal_rating_a": 340
            },
            "2/0_ACSR": {
                "resistance_ohm_per_km": 0.433,
                "reactance_ohm_per_km": 0.479,
                "susceptance_us_per_km": 10.1,
                "thermal_rating_a": 230
            },
            "1/0_ACSR": {
                "resistance_ohm_per_km": 0.547,
                "reactance_ohm_per_km": 0.495,
                "susceptance_us_per_km": 9.8,
                "thermal_rating_a": 180
            }
        }
        
        return conductor_data.get(conductor_type, conductor_data["2/0_ACSR"])
    
    def _allocate_loads(self):
        """Allocate loads to feeder segments based on customer distribution."""
        
        total_load_allocated = 0
        total_customers_allocated = 0
        
        for i, segment in enumerate(self.segments):
            # Allocate customers based on segment characteristics
            if i == 0:  # Substation segment - no load
                customer_allocation = 0
                load_allocation = 0
            else:
                # Higher customer density near substation, decreasing with distance
                distance_factor = 1.0 / (i + 1) ** 0.5
                
                # Segment length factor
                length_factor = segment.length_km / self.length_km
                
                # Combined allocation factor
                allocation_factor = distance_factor * length_factor
                
                # Normalize allocation
                remaining_customers = self.total_customers - total_customers_allocated
                remaining_segments = len(self.segments) - i
                
                if remaining_segments > 1:
                    customer_allocation = int(remaining_customers * allocation_factor / 
                                            sum([1.0/(j+1)**0.5 * seg.length_km/self.length_km 
                                                for j, seg in enumerate(self.segments[i:], i)]))
                else:
                    customer_allocation = remaining_customers
                
                # Load allocation based on customers and load density
                load_allocation = (customer_allocation * 
                                 self.load_characteristics.load_density_kw_per_customer / 1000)
                
                # Apply diversity factor
                load_allocation *= self.load_characteristics.diversity_factor
            
            # Update segment
            segment.customers = customer_allocation
            segment.load_mw = load_allocation
            segment.load_mvar = load_allocation * np.tan(np.arccos(self.load_characteristics.power_factor))
            
            total_load_allocated += load_allocation
            total_customers_allocated += customer_allocation
        
        # Normalize to match total load
        if total_load_allocated > 0:
            scaling_factor = self.base_load_mw / total_load_allocated
            for segment in self.segments:
                segment.load_mw *= scaling_factor
                segment.load_mvar *= scaling_factor
    
    def calculate_voltage_drop(self, load_profile: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate voltage drop along feeder for given load profile.
        
        Args:
            load_profile: 8760-hour load profile in per-unit
            
        Returns:
            Dictionary with voltage profiles for each segment
        """
        
        voltage_profiles = {}
        
        for hour in range(len(load_profile)):
            load_multiplier = load_profile[hour]
            
            # Start with substation voltage (typically 1.0 p.u.)
            voltage = 1.0
            
            for i, segment in enumerate(self.segments):
                if i == 0:  # Substation bus
                    voltage_profiles.setdefault(f'Bus_{i}', []).append(voltage)
                    continue
                
                # Calculate current through segment
                segment_load = segment.load_mw * load_multiplier
                segment_reactive = segment.load_mvar * load_multiplier
                
                # Current calculation (simplified)
                apparent_power = np.sqrt(segment_load**2 + segment_reactive**2)
                current_a = apparent_power * 1000 / (np.sqrt(3) * segment.voltage_kv * 1000)
                
                # Voltage drop calculation
                resistance_total = segment.resistance_ohm_per_km * segment.length_km
                reactance_total = segment.reactance_ohm_per_km * segment.length_km
                
                voltage_drop_real = current_a * resistance_total * segment_load / apparent_power if apparent_power > 0 else 0
                voltage_drop_reactive = current_a * reactance_total * segment_reactive / apparent_power if apparent_power > 0 else 0
                
                total_voltage_drop = np.sqrt(voltage_drop_real**2 + voltage_drop_reactive**2)
                
                # Convert to per-unit
                voltage_drop_pu = total_voltage_drop / (segment.voltage_kv * 1000)
                
                # Update voltage
                voltage -= voltage_drop_pu
                
                voltage_profiles.setdefault(f'Bus_{i+1}', []).append(voltage)
        
        return voltage_profiles
    
    def calculate_losses(self, load_profile: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """
        Calculate feeder losses for given load profile.
        
        Args:
            load_profile: 8760-hour load profile in per-unit
            
        Returns:
            Dictionary with loss calculations
        """
        
        total_losses = np.zeros(len(load_profile))
        segment_losses = {}
        
        for i, segment in enumerate(self.segments):
            if i == 0:  # Skip substation
                continue
            
            segment_loss_profile = np.zeros(len(load_profile))
            
            for hour in range(len(load_profile)):
                load_multiplier = load_profile[hour]
                
                # Calculate segment loading
                segment_load = segment.load_mw * load_multiplier
                
                # Current calculation
                current_a = segment_load * 1000 / (np.sqrt(3) * segment.voltage_kv)
                
                # I²R losses
                resistance_total = segment.resistance_ohm_per_km * segment.length_km
                loss_mw = 3 * (current_a**2) * resistance_total / 1e6  # Convert to MW
                
                segment_loss_profile[hour] = loss_mw
                total_losses[hour] += loss_mw
            
            segment_losses[f'Segment_{i}'] = segment_loss_profile
        
        return {
            'total_losses_mw': total_losses,
            'segment_losses': segment_losses,
            'annual_loss_mwh': np.sum(total_losses),
            'peak_loss_mw': np.max(total_losses),
            'avg_loss_rate_percent': (np.mean(total_losses) / self.base_load_mw) * 100
        }
    
    def calculate_thermal_loading(self, load_profile: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate thermal loading of feeder segments.
        
        Args:
            load_profile: 8760-hour load profile in per-unit
            
        Returns:
            Dictionary with thermal loading profiles
        """
        
        thermal_loading = {}
        
        for i, segment in enumerate(self.segments):
            if i == 0:  # Skip substation
                continue
            
            loading_profile = np.zeros(len(load_profile))
            
            for hour in range(len(load_profile)):
                load_multiplier = load_profile[hour]
                
                # Calculate segment loading
                segment_load = segment.load_mw * load_multiplier
                
                # Current calculation
                current_a = segment_load * 1000 / (np.sqrt(3) * segment.voltage_kv)
                
                # Thermal loading as percentage of rating
                loading_percent = (current_a / segment.thermal_rating_a) * 100
                
                loading_profile[hour] = loading_percent
            
            thermal_loading[f'Segment_{i}'] = loading_profile
        
        return thermal_loading
    
    def add_distributed_generation(self, dg_locations: List[int], dg_capacities: List[float]):
        """
        Add distributed generation to specified segments.
        
        Args:
            dg_locations: List of segment indices for DG placement
            dg_capacities: List of DG capacities in MW
        """
        
        for location, capacity in zip(dg_locations, dg_capacities):
            if 0 <= location < len(self.segments):
                segment = self.segments[location]
                
                # Reduce net load by DG output (simplified)
                # In practice, would need to account for generation profiles
                segment.load_mw = max(0, segment.load_mw - capacity * 0.3)  # Assume 30% capacity factor
    
    def add_voltage_regulation(self, equipment_type: str, location: int, settings: Dict):
        """
        Add voltage regulation equipment to the feeder model.
        
        Args:
            equipment_type: Type of equipment ('capacitor', 'regulator', 'ltc')
            location: Segment index for equipment placement
            settings: Equipment settings dictionary
        """
        
        regulation_equipment = {
            'type': equipment_type,
            'location': location,
            'settings': settings,
            'segment_id': self.segments[location].segment_id if location < len(self.segments) else None
        }
        
        self.voltage_regulation_equipment.append(regulation_equipment)
    
    def calculate_hosting_capacity(self, dg_type: str = 'solar') -> Dict[str, float]:
        """
        Calculate distributed generation hosting capacity for each segment.
        
        Args:
            dg_type: Type of distributed generation
            
        Returns:
            Dictionary with hosting capacity for each segment
        """
        
        hosting_capacity = {}
        
        for i, segment in enumerate(self.segments):
            if i == 0:  # Skip substation
                continue
            
            # Simplified hosting capacity calculation
            # Based on thermal limits and voltage rise constraints
            
            # Thermal constraint
            thermal_headroom = segment.thermal_rating_a * 0.8  # 80% of thermal rating
            current_loading = segment.load_mw * 1000 / (np.sqrt(3) * segment.voltage_kv)
            thermal_capacity = (thermal_headroom - current_loading) * np.sqrt(3) * segment.voltage_kv / 1000
            
            # Voltage rise constraint (simplified)
            # Assume 5% voltage rise limit
            voltage_rise_limit = 0.05 * segment.voltage_kv * 1000  # 5% in volts
            reactance_total = segment.reactance_ohm_per_km * segment.length_km
            voltage_capacity = voltage_rise_limit / reactance_total * segment.voltage_kv / 1000
            
            # Hosting capacity is the minimum of constraints
            hosting_cap = min(thermal_capacity, voltage_capacity)
            hosting_cap = max(0, hosting_cap)  # Ensure non-negative
            
            hosting_capacity[f'Segment_{i}'] = hosting_cap
        
        return hosting_capacity
    
    def get_protection_coordination_data(self) -> Dict[str, Dict]:
        """
        Get protection equipment coordination data.
        
        Returns:
            Dictionary with protection equipment settings and coordination data
        """
        
        protection_data = {}
        
        for i, segment in enumerate(self.segments):
            # Calculate fault current levels (simplified)
            fault_current = self._calculate_fault_current(i)
            
            # Determine protection settings
            if i == 0:  # Substation breaker
                protection_type = "Circuit Breaker"
                pickup_current = self.rated_capacity_mva * 1000 / (np.sqrt(3) * self.base_voltage_kv) * 1.25
                time_delay = 0.5  # seconds
            
            elif i < len(self.segments) * 0.3:  # Main feeder sections
                protection_type = "Recloser"
                pickup_current = segment.thermal_rating_a * 1.2
                time_delay = 0.3
            
            else:  # Lateral sections
                protection_type = "Fuse"
                pickup_current = segment.thermal_rating_a * 1.5
                time_delay = 0.1
            
            protection_data[f'Segment_{i}'] = {
                'protection_type': protection_type,
                'pickup_current_a': pickup_current,
                'time_delay_s': time_delay,
                'fault_current_a': fault_current,
                'coordination_margin': 0.2  # 200ms coordination margin
            }
        
        return protection_data
    
    def _calculate_fault_current(self, segment_index: int) -> float:
        """Calculate fault current at segment location (simplified)."""
        
        # Simplified fault current calculation
        # In practice, would use detailed impedance calculations
        
        base_fault_current = 8000  # Typical substation fault current in amperes
        
        # Reduce fault current with distance
        total_impedance = 0
        for i in range(segment_index + 1):
            segment = self.segments[i]
            segment_impedance = np.sqrt(
                (segment.resistance_ohm_per_km * segment.length_km)**2 +
                (segment.reactance_ohm_per_km * segment.length_km)**2
            )
            total_impedance += segment_impedance
        
        # Fault current with impedance
        voltage_base = self.base_voltage_kv * 1000 / np.sqrt(3)
        fault_current = voltage_base / (total_impedance + 0.1)  # Add source impedance
        
        return min(fault_current, base_fault_current)
    
    def export_model_data(self) -> Dict:
        """Export complete feeder model data for analysis."""
        
        model_data = {
            'feeder_info': {
                'name': self.name,
                'base_voltage_kv': self.base_voltage_kv,
                'rated_capacity_mva': self.rated_capacity_mva,
                'length_km': self.length_km,
                'total_customers': self.total_customers,
                'base_load_mw': self.base_load_mw,
                'network_topology': self.network_topology
            },
            'load_characteristics': {
                'residential_pct': self.load_characteristics.residential_pct,
                'commercial_pct': self.load_characteristics.commercial_pct,
                'industrial_pct': self.load_characteristics.industrial_pct,
                'power_factor': self.load_characteristics.power_factor,
                'diversity_factor': self.load_characteristics.diversity_factor,
                'load_density_kw_per_customer': self.load_characteristics.load_density_kw_per_customer
            },
            'segments': []
        }
        
        for segment in self.segments:
            segment_data = {
                'segment_id': segment.segment_id,
                'from_bus': segment.from_bus,
                'to_bus': segment.to_bus,
                'length_km': segment.length_km,
                'conductor_type': segment.conductor_type,
                'customers': segment.customers,
                'load_mw': segment.load_mw,
                'load_mvar': segment.load_mvar,
                'resistance_ohm_per_km': segment.resistance_ohm_per_km,
                'reactance_ohm_per_km': segment.reactance_ohm_per_km,
                'thermal_rating_a': segment.thermal_rating_a
            }
            model_data['segments'].append(segment_data)
        
        return model_data
    
    def validate_model(self) -> Dict[str, Union[bool, List[str]]]:
        """Validate the feeder model for consistency and realistic parameters."""
        
        errors = []
        warnings = []
        
        # Check basic parameters
        if self.base_voltage_kv <= 0:
            errors.append("Base voltage must be positive")
        
        if self.rated_capacity_mva <= 0:
            errors.append("Rated capacity must be positive")
        
        if self.length_km <= 0:
            errors.append("Feeder length must be positive")
        
        # Check load allocation
        total_allocated_load = sum(segment.load_mw for segment in self.segments)
        load_difference = abs(total_allocated_load - self.base_load_mw)
        
        if load_difference > 0.1:  # 0.1 MW tolerance
            warnings.append(f"Total allocated load ({total_allocated_load:.2f} MW) differs from base load ({self.base_load_mw:.2f} MW)")
        
        # Check customer allocation
        total_allocated_customers = sum(segment.customers for segment in self.segments)
        customer_difference = abs(total_allocated_customers - self.total_customers)
        
        if customer_difference > 5:  # 5 customer tolerance
            warnings.append(f"Total allocated customers ({total_allocated_customers}) differs from total customers ({self.total_customers})")
        
        # Check conductor sizing
        for i, segment in enumerate(self.segments):
            if segment.load_mw > 0:
                current_a = segment.load_mw * 1000 / (np.sqrt(3) * segment.voltage_kv)
                loading_pct = (current_a / segment.thermal_rating_a) * 100
                
                if loading_pct > 100:
                    warnings.append(f"Segment {i} is overloaded ({loading_pct:.1f}%)")
                elif loading_pct > 80:
                    warnings.append(f"Segment {i} has high loading ({loading_pct:.1f}%)")
        
        # Check voltage levels
        if self.base_voltage_kv < 4.0 or self.base_voltage_kv > 50.0:
            warnings.append(f"Unusual voltage level ({self.base_voltage_kv:.1f} kV)")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def update_configuration(self, new_config: Dict):
        """Update feeder configuration and regenerate model."""
        
        # Update configuration
        self.config.update(new_config)
        
        # Re-extract parameters
        self.name = self.config.get('name', self.name)
        self.base_voltage_kv = self.config.get('base_voltage', self.base_voltage_kv)
        self.rated_capacity_mva = self.config.get('rated_capacity', self.rated_capacity_mva)
        self.length_km = self.config.get('length_km', self.length_km)
        self.total_customers = self.config.get('customers', self.total_customers)
        self.base_load_mw = self.config.get('base_load_mw', self.base_load_mw)
        self.dg_capacity_mw = self.config.get('dg_capacity_mw', self.dg_capacity_mw)
        
        # Regenerate model
        self.segments.clear()
        self._initialize_load_characteristics()
        self._create_feeder_segments()
        self._allocate_loads()

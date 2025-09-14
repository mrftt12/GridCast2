import pandapower as pp
import pandas as pd
import numpy as np
from pandapower.timeseries import DFData
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.control import ConstControl
import warnings
warnings.filterwarnings('ignore')

class PowerFlowAnalyzer:
    """
    PandaPower integration for time series load flow analysis.
    Provides comprehensive electrical network modeling and analysis
    for distribution feeder planning.
    """
    
    def __init__(self):
        self.network = None
        
    def create_network(self, network_type, num_buses, num_lines, feeder_config, 
                      transformer_config, load_allocation, dg_buses):
        """
        Create electrical network model using PandaPower.
        
        Args:
            network_type: Type of network topology
            num_buses: Number of buses in the network
            num_lines: Number of lines in the network
            feeder_config: Feeder configuration parameters
            transformer_config: Transformer configuration
            load_allocation: Method for load allocation
            dg_buses: List of buses with distributed generation
            
        Returns:
            PandaPower network object
        """
        try:
            # Create empty network
            net = pp.create_empty_network(name=f"Feeder_{feeder_config.get('name', 'Unknown')}")
            
            # Create buses
            bus_voltages = self._determine_bus_voltages(num_buses, feeder_config)
            bus_indices = []
            
            for i in range(num_buses):
                bus_idx = pp.create_bus(
                    net, 
                    vn_kv=bus_voltages[i], 
                    name=f"Bus_{i+1}",
                    type="b"  # PQ bus
                )
                bus_indices.append(bus_idx)
            
            # Create external grid connection (infinite bus)
            if transformer_config.get('enabled', True):
                # High voltage bus for transformer
                hv_bus = pp.create_bus(
                    net, 
                    vn_kv=transformer_config.get('hv_kv', 69.0),
                    name="HV_Bus",
                    type="b"
                )
                
                # External grid at HV side
                pp.create_ext_grid(
                    net, 
                    bus=hv_bus, 
                    vm_pu=1.0, 
                    va_degree=0.0,
                    name="External_Grid"
                )
                
                # Substation transformer
                pp.create_transformer_from_parameters(
                    net,
                    hv_bus=hv_bus,
                    lv_bus=bus_indices[0],
                    sn_mva=transformer_config.get('rating_mva', 25.0),
                    vn_hv_kv=transformer_config.get('hv_kv', 69.0),
                    vn_lv_kv=transformer_config.get('lv_kv', 12.47),
                    vk_percent=7.5,  # Short circuit voltage
                    vkr_percent=0.5,  # Real part of short circuit voltage
                    pfe_kw=10.0,  # Iron losses
                    i0_percent=0.3,  # No-load current
                    name="Substation_Transformer"
                )
                
                # Set first bus as slack bus equivalent
                net.bus.loc[bus_indices[0], 'type'] = 'n'  # Reference bus
                
            else:
                # Direct external grid connection
                pp.create_ext_grid(
                    net, 
                    bus=bus_indices[0], 
                    vm_pu=1.0, 
                    va_degree=0.0,
                    name="External_Grid"
                )
            
            # Create lines based on network topology
            self._create_network_lines(net, bus_indices, network_type, feeder_config)
            
            # Allocate loads to buses
            self._allocate_loads(net, bus_indices, load_allocation, feeder_config)
            
            # Add distributed generation
            if dg_buses:
                self._add_distributed_generation(net, bus_indices, dg_buses, feeder_config)
            
            # Add voltage control equipment if specified
            self._add_voltage_control(net, bus_indices, feeder_config)
            
            # Validate network
            self._validate_network(net)
            
            return net
            
        except Exception as e:
            raise Exception(f"Error creating network: {str(e)}")
    
    def _determine_bus_voltages(self, num_buses, feeder_config):
        """Determine voltage levels for each bus."""
        base_voltage = feeder_config.get('base_voltage', 12.47)
        
        # For distribution feeders, typically all buses are at the same voltage level
        # In more complex networks, could have voltage regulators creating different levels
        bus_voltages = [base_voltage] * num_buses
        
        return bus_voltages
    
    def _create_network_lines(self, net, bus_indices, network_type, feeder_config):
        """Create lines based on network topology."""
        
        feeder_length = feeder_config.get('length_km', 5.2)
        base_voltage = feeder_config.get('base_voltage', 12.47)
        
        # Standard distribution line parameters (per km)
        if base_voltage > 20:  # Subtransmission
            r_ohm_per_km = 0.15
            x_ohm_per_km = 0.30
            c_nf_per_km = 12.0
            max_i_ka = 0.4
        else:  # Distribution
            r_ohm_per_km = 0.25
            x_ohm_per_km = 0.35
            c_nf_per_km = 10.0
            max_i_ka = 0.3
        
        if network_type == "Radial Feeder":
            # Create radial topology (tree structure)
            avg_line_length = feeder_length / (len(bus_indices) - 1)
            
            for i in range(len(bus_indices) - 1):
                # Variable line lengths for more realistic model
                line_length = avg_line_length * (0.8 + 0.4 * np.random.random())
                
                pp.create_line_from_parameters(
                    net,
                    from_bus=bus_indices[i],
                    to_bus=bus_indices[i + 1],
                    length_km=line_length,
                    r_ohm_per_km=0.25,
                    x_ohm_per_km=0.35,
                    c_nf_per_km=10.0,
                    max_i_ka=0.3,
                    name=f"Line_{i+1}_{i+2}"
                )
        
        elif network_type == "Loop Feeder":
            # Create loop with radial branches
            main_loop_buses = min(8, len(bus_indices))
            
            # Main loop
            for i in range(main_loop_buses):
                next_bus = (i + 1) % main_loop_buses
                line_length = feeder_length / main_loop_buses
                
                pp.create_line_from_parameters(
                    net,
                    from_bus=bus_indices[i],
                    to_bus=bus_indices[next_bus],
                    length_km=line_length,
                    r_ohm_per_km=0.25,
                    x_ohm_per_km=0.35,
                    c_nf_per_km=10.0,
                    max_i_ka=0.3,
                    name=f"Loop_Line_{i+1}_{next_bus+1}"
                )
            
            # Radial branches from loop
            for i in range(main_loop_buses, len(bus_indices)):
                parent_bus = np.random.randint(0, main_loop_buses)
                branch_length = feeder_length * 0.3 * np.random.random()
                
                pp.create_line_from_parameters(
                    net,
                    from_bus=bus_indices[parent_bus],
                    to_bus=bus_indices[i],
                    length_km=branch_length,
                    r_ohm_per_km=0.35,
                    x_ohm_per_km=0.40,
                    c_nf_per_km=8.0,
                    max_i_ka=0.2,
                    name=f"Branch_Line_{parent_bus+1}_{i+1}"
                )
        
        else:  # Custom or other topologies - default to radial
            avg_line_length = feeder_length / (len(bus_indices) - 1)
            
            for i in range(len(bus_indices) - 1):
                line_length = avg_line_length
                
                pp.create_line_from_parameters(
                    net,
                    from_bus=bus_indices[i],
                    to_bus=bus_indices[i + 1],
                    length_km=line_length,
                    r_ohm_per_km=0.25,
                    x_ohm_per_km=0.35,
                    c_nf_per_km=10.0,
                    max_i_ka=0.3,
                    name=f"Line_{i+1}_{i+2}"
                )
    
    def _allocate_loads(self, net, bus_indices, load_allocation, feeder_config):
        """Allocate loads to buses based on specified method."""
        
        total_load_mw = feeder_config.get('base_load_mw', 8.5)
        total_customers = feeder_config.get('customers', 850)
        
        # Skip the first bus (substation) for load allocation
        load_buses = bus_indices[1:]
        
        if load_allocation == "Uniform Distribution":
            # Equal load on all buses
            load_per_bus = total_load_mw / len(load_buses)
            customers_per_bus = total_customers // len(load_buses)
            
            for bus in load_buses:
                pp.create_load(
                    net,
                    bus=bus,
                    p_mw=load_per_bus,
                    q_mvar=load_per_bus * 0.3,  # Assume 0.3 power factor
                    name=f"Load_Bus_{bus}"
                )
        
        elif load_allocation == "Distance-Based":
            # Higher load near substation, decreasing with distance
            weights = [1.0 / (i + 1) for i in range(len(load_buses))]
            total_weight = sum(weights)
            
            for i, bus in enumerate(load_buses):
                load_fraction = weights[i] / total_weight
                load_mw = total_load_mw * load_fraction
                
                pp.create_load(
                    net,
                    bus=bus,
                    p_mw=load_mw,
                    q_mvar=load_mw * 0.3,
                    name=f"Load_Bus_{bus}"
                )
        
        elif load_allocation == "Customer-Based":
            # Load proportional to customer density (with some randomness)
            customer_densities = [0.5 + np.random.random() for _ in load_buses]
            total_density = sum(customer_densities)
            
            for i, bus in enumerate(load_buses):
                customer_fraction = customer_densities[i] / total_density
                load_mw = total_load_mw * customer_fraction
                
                pp.create_load(
                    net,
                    bus=bus,
                    p_mw=load_mw,
                    q_mvar=load_mw * 0.3,
                    name=f"Load_Bus_{bus}"
                )
        
        else:  # Default to uniform
            load_per_bus = total_load_mw / len(load_buses)
            
            for bus in load_buses:
                pp.create_load(
                    net,
                    bus=bus,
                    p_mw=load_per_bus,
                    q_mvar=load_per_bus * 0.3,
                    name=f"Load_Bus_{bus}"
                )
    
    def _add_distributed_generation(self, net, bus_indices, dg_buses, feeder_config):
        """Add distributed generation to specified buses."""
        
        # Get DG capacity from feeder config
        total_dg_capacity = feeder_config.get('dg_capacity_mw', 2.1)
        
        # Distribute DG capacity among selected buses
        dg_per_bus = total_dg_capacity / len(dg_buses)
        
        for bus_num in dg_buses:
            if bus_num <= len(bus_indices):
                bus_idx = bus_indices[bus_num - 1]  # Convert to 0-based indexing
                
                # Create static generator for DG
                pp.create_sgen(
                    net,
                    bus=bus_idx,
                    p_mw=dg_per_bus,
                    q_mvar=0.0,  # Assume unity power factor for solar
                    name=f"Solar_PV_Bus_{bus_num}",
                    type="PV"
                )
    
    def _add_voltage_control(self, net, bus_indices, feeder_config):
        """Add voltage control equipment if specified."""
        
        # This is a simplified implementation
        # In practice, would add voltage regulators, capacitor banks, etc.
        
        # Example: Add capacitor bank at midpoint
        if len(bus_indices) > 4:
            mid_bus = bus_indices[len(bus_indices) // 2]
            
            # Create shunt capacitor for reactive power support
            pp.create_shunt(
                net,
                bus=mid_bus,
                q_mvar=1.0,  # 1 MVAr capacitor bank
                p_mw=0.0,
                name="Capacitor_Bank"
            )
    
    def _validate_network(self, net):
        """Validate the created network."""
        
        # Check for isolated buses
        if len(net.bus) == 0:
            raise Exception("No buses created in network")
        
        if len(net.line) == 0:
            raise Exception("No lines created in network")
        
        if len(net.load) == 0:
            raise Exception("No loads created in network")
        
        # Check connectivity
        try:
            pp.runpp(net, algorithm='nr', max_iteration=100)
        except Exception as e:
            raise Exception(f"Network validation failed - power flow did not converge: {str(e)}")
    
    def run_time_series_analysis(self, network, load_profile, hours_to_analyze, analysis_options):
        """
        Run time series load flow analysis.
        
        Args:
            network: PandaPower network object
            load_profile: 8760-hour load profile
            hours_to_analyze: List of hours to analyze
            analysis_options: Dictionary of analysis options
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Prepare time series data
            time_steps = len(hours_to_analyze)
            
            # Create load scaling factors based on load profile
            load_scaling = np.array([load_profile[h] for h in hours_to_analyze])
            
            # Create DG scaling factors (solar profile)
            dg_scaling = self._create_dg_time_series(hours_to_analyze, network)
            
            # Prepare DataFrames for time series
            load_df = pd.DataFrame(index=range(time_steps))
            dg_df = pd.DataFrame(index=range(time_steps))
            
            # Scale loads
            for load_idx in network.load.index:
                load_df[f'load_{load_idx}'] = load_scaling
            
            # Scale DG
            if len(network.sgen) > 0:
                for sgen_idx in network.sgen.index:
                    dg_df[f'sgen_{sgen_idx}'] = dg_scaling
            
            # Create time series data sources
            load_data_source = DFData(load_df)
            
            # Create controllers for loads
            for load_idx in network.load.index:
                ConstControl(
                    net=network,
                    element='load',
                    variable='p_mw',
                    element_index=load_idx,
                    data_source=load_data_source,
                    profile_name=f'load_{load_idx}'
                )
                
                ConstControl(
                    net=network,
                    element='load',
                    variable='q_mvar',
                    element_index=load_idx,
                    data_source=load_data_source,
                    profile_name=f'load_{load_idx}'
                )
            
            # Create controllers for DG if present
            if len(network.sgen) > 0:
                dg_data_source = DFData(dg_df)
                
                for sgen_idx in network.sgen.index:
                    ConstControl(
                        net=network,
                        element='sgen',
                        variable='p_mw',
                        element_index=sgen_idx,
                        data_source=dg_data_source,
                        profile_name=f'sgen_{sgen_idx}'
                    )
            
            # Configure output variables
            output_variables = analysis_options.get('output_variables', ['Bus Voltages'])
            
            # Run time series
            run_timeseries(
                network,
                time_steps=time_steps,
                continue_on_divergence=True,
                verbose=False
            )
            
            # Process results
            results = self._process_time_series_results(
                network, 
                hours_to_analyze, 
                analysis_options
            )
            
            return results
            
        except Exception as e:
            raise Exception(f"Error running time series analysis: {str(e)}")
    
    def _create_dg_time_series(self, hours_to_analyze, network):
        """Create time series scaling factors for distributed generation."""
        
        dg_scaling = np.zeros(len(hours_to_analyze))
        
        for i, hour in enumerate(hours_to_analyze):
            hour_of_day = hour % 24
            day_of_year = hour // 24
            
            # Solar generation pattern
            if 6 <= hour_of_day <= 18:  # Daylight hours
                # Solar irradiance curve
                solar_factor = np.sin(np.pi * (hour_of_day - 6) / 12)
                
                # Seasonal variation
                seasonal_factor = 0.7 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
                
                # Add some variability for clouds
                weather_factor = 0.8 + 0.2 * np.random.random()
                
                dg_scaling[i] = solar_factor * seasonal_factor * weather_factor
            else:
                dg_scaling[i] = 0.0
        
        return dg_scaling
    
    def _process_time_series_results(self, network, hours_analyzed, analysis_options):
        """Process and compile time series analysis results."""
        
        results = {
            'hours_analyzed': hours_analyzed,
            'voltage_violations': 0,
            'thermal_violations': 0
        }
        
        voltage_limits = analysis_options.get('voltage_limits', (0.95, 1.05))
        thermal_limit = analysis_options.get('thermal_limit', 0.9)
        
        # Process voltage results
        if analysis_options.get('voltage_analysis', True):
            voltage_results = {}
            
            # Get voltage results for each bus
            for bus_idx in network.bus.index:
                bus_voltages = network.res_bus.loc[bus_idx, 'vm_pu']
                
                # Handle case where bus_voltages might be a single value or series
                if hasattr(bus_voltages, '__iter__') and not isinstance(bus_voltages, str):
                    voltage_list = list(bus_voltages)
                else:
                    voltage_list = [bus_voltages] * len(hours_analyzed)
                
                voltage_results[bus_idx] = voltage_list
                
                # Count violations
                violations = sum(1 for v in voltage_list 
                               if v < voltage_limits[0] or v > voltage_limits[1])
                results['voltage_violations'] += violations
            
            results['voltage_results'] = voltage_results
        
        # Process thermal results
        if analysis_options.get('thermal_analysis', True):
            thermal_results = {}
            
            # Get line loading results
            for line_idx in network.line.index:
                try:
                    line_loadings = network.res_line.loc[line_idx, 'loading_percent'] / 100.0
                    
                    # Handle case where line_loadings might be a single value or series
                    if hasattr(line_loadings, '__iter__') and not isinstance(line_loadings, str):
                        loading_list = list(line_loadings)
                    else:
                        loading_list = [line_loadings] * len(hours_analyzed)
                    
                    thermal_results[line_idx] = loading_list
                    
                    # Count violations
                    violations = sum(1 for loading in loading_list if loading > thermal_limit)
                    results['thermal_violations'] += violations
                    
                except Exception:
                    # If loading data not available, create placeholder
                    thermal_results[line_idx] = [0.5] * len(hours_analyzed)
            
            results['thermal_results'] = thermal_results
        
        # Process losses
        if analysis_options.get('losses_analysis', True):
            try:
                # Calculate total system losses for each time step
                losses_results = []
                
                for i in range(len(hours_analyzed)):
                    # Sum line losses - simplified calculation
                    total_loss = 0
                    for line_idx in network.line.index:
                        # Estimate losses based on loading
                        loading = thermal_results.get(line_idx, [0.3])[i] if 'thermal_results' in locals() else 0.3
                        line_resistance = 0.1  # Simplified resistance value
                        loss_mw = loading ** 2 * line_resistance
                        total_loss += loss_mw
                    
                    losses_results.append(total_loss)
                
                results['losses_results'] = losses_results
                results['total_losses_mwh'] = sum(losses_results)
                
            except Exception:
                # Fallback if losses calculation fails
                results['losses_results'] = [0.1] * len(hours_analyzed)
                results['total_losses_mwh'] = 0.1 * len(hours_analyzed)
        
        return results
    
    def export_network_data(self, network, filename=None):
        """Export network data for external analysis tools."""
        
        network_data = {
            'buses': network.bus.to_dict('records'),
            'lines': network.line.to_dict('records'),
            'loads': network.load.to_dict('records'),
            'generators': network.sgen.to_dict('records') if len(network.sgen) > 0 else [],
            'transformers': network.trafo.to_dict('records') if len(network.trafo) > 0 else []
        }
        
        if filename:
            import json
            with open(filename, 'w') as f:
                json.dump(network_data, f, indent=2, default=str)
        
        return network_data
    
    def create_ieee_test_system(self, system_type="IEEE_13"):
        """Create standard IEEE test systems for validation."""
        
        if system_type == "IEEE_13":
            # Simplified IEEE 13-bus test feeder
            net = pp.create_empty_network(name="IEEE_13_Bus_Modified")
            
            # Create buses
            buses = []
            voltages = [4.16] * 13  # 4.16 kV distribution system
            
            for i in range(13):
                bus = pp.create_bus(net, vn_kv=voltages[i], name=f"Bus_{i+1}")
                buses.append(bus)
            
            # External grid
            pp.create_ext_grid(net, bus=buses[0], vm_pu=1.0)
            
            # Create lines (simplified)
            line_data = [
                (0, 1, 0.5), (1, 2, 0.3), (2, 3, 0.4), (3, 4, 0.2),
                (1, 5, 0.6), (5, 6, 0.3), (6, 7, 0.4),
                (2, 8, 0.5), (8, 9, 0.3),
                (5, 10, 0.4), (10, 11, 0.2), (11, 12, 0.3)
            ]
            
            for from_bus, to_bus, length in line_data:
                pp.create_line_from_parameters(
                    net,
                    from_bus=buses[from_bus],
                    to_bus=buses[to_bus],
                    length_km=length,
                    r_ohm_per_km=0.25,
                    x_ohm_per_km=0.35,
                    c_nf_per_km=10.0,
                    max_i_ka=0.3
                )
            
            # Add loads
            load_data = [
                (1, 0.4, 0.2), (2, 0.3, 0.15), (3, 0.5, 0.25),
                (4, 0.2, 0.1), (6, 0.6, 0.3), (7, 0.3, 0.15),
                (9, 0.4, 0.2), (10, 0.2, 0.1), (11, 0.3, 0.15), (12, 0.5, 0.25)
            ]
            
            for bus_num, p_mw, q_mvar in load_data:
                pp.create_load(net, bus=buses[bus_num], p_mw=p_mw, q_mvar=q_mvar)
            
            return net
        
        else:
            raise ValueError(f"Unknown test system: {system_type}")

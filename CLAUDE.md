# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
streamlit run app.py --server.port 5000
```

The application runs on port 5000 and provides a multi-page Streamlit interface for electric utility distribution system planning.

### Dependencies
Dependencies are managed via `pyproject.toml`. Key packages include:
- `streamlit` - Web application framework
- `pandapower` - Electrical network analysis 
- `pandas` and `numpy` - Data manipulation
- `plotly` - Interactive visualizations
- `scikit-learn` - Machine learning for forecasting

## Architecture Overview

### Application Structure
This is a Streamlit-based electric utility Integrated Resource Planning (IRP) tool with the following architecture:

**Main Application (`app.py`)**
- Entry point and navigation hub
- Manages global session state for feeder configuration, load profiles, and analysis results
- Provides overview dashboard with system status

**Page Modules (`pages/`)**
- `1_Feeder_Configuration.py` - Define electrical feeder parameters
- `2_Load_Forecasting.py` - Generate 10-year load forecasts with EV adoption scenarios
- `3_Load_Profile_Editor.py` - Edit and visualize 8760-hour load profiles
- `4_Load_Flow_Analysis.py` - Run PandaPower load flow studies
- `5_Dashboard.py` - Executive dashboard with planning insights

**Core Utilities (`utils/`)**
- `load_forecasting.py` - LoadForecaster class for ML-based forecasting
- `pandapower_integration.py` - PowerFlowAnalyzer for electrical network modeling
- `load_profiles.py` - LoadProfileGenerator for statistical load curve generation
- `feeder_model.py` - Data models for electrical feeder components
- `data_export.py` - Export utilities for analysis results

### State Management
All application data is stored in Streamlit session state:
- `feeder_config` - Electrical feeder parameters and characteristics
- `load_profile` - 8760-hour base load profile array
- `forecast_scenarios` - Load growth and EV adoption scenarios
- `forecast_results` - Generated forecast data and projections
- `network_model` - PandaPower network object
- `loadflow_results` - Load flow analysis results

### Key Analysis Capabilities
- **Load Forecasting**: 10-year projections with EV adoption, load growth, and climate factors
- **Network Modeling**: PandaPower integration for AC load flow analysis
- **Time Series Analysis**: 8760-hour load profile manipulation and visualization
- **Scenario Planning**: Multiple forecast scenarios for utility planning uncertainty
- **Data Export**: CSV, JSON, and formatted report generation

### Technical Integration Points
- **PandaPower**: Professional electrical network analysis library for load flow calculations
- **Plotly**: Interactive visualization for technical charts and load curves
- **Scikit-learn**: Machine learning algorithms for load forecasting models
- **Session State**: Maintains analysis continuity across multi-page workflow

### Data Flow
1. Configure feeder parameters and base load characteristics
2. Generate/edit 8760-hour load profiles
3. Run load forecasting with multiple scenarios
4. Create electrical network model in PandaPower
5. Execute time series load flow analysis
6. View results in executive dashboard
7. Export data in utility-standard formats

This tool is designed for utility engineers and planners working on distribution system capacity planning, EV integration studies, and long-term resource planning.
# Electric Utility Resource Planning Tool

## Overview

This is a comprehensive electric utility distribution system planning and analysis tool built with Streamlit. The application provides utilities with advanced capabilities for load forecasting, feeder configuration, load profile analysis, and electrical network studies. It integrates multiple analytical tools including PandaPower for load flow analysis, machine learning for forecasting, and interactive visualization for planning scenarios.

The tool is designed to help utility engineers and planners make informed decisions about distribution system upgrades, capacity planning, and resource allocation over 10-year planning horizons. It specifically addresses modern utility challenges including electric vehicle adoption, distributed generation integration, and climate change impacts on load growth.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based multi-page web application
- **Page Structure**: Modular design with 5 main sections:
  - Main dashboard (`app.py`)
  - Feeder Configuration (`pages/1_Feeder_Configuration.py`)
  - Load Forecasting (`pages/2_Load_Forecasting.py`) 
  - Load Profile Editor (`pages/3_Load_Profile_Editor.py`)
  - Load Flow Analysis (`pages/4_Load_Flow_Analysis.py`)
  - Results Dashboard (`pages/5_Dashboard.py`)
- **State Management**: Streamlit session state for maintaining configuration and analysis results across pages
- **Visualization**: Plotly for interactive charts and technical visualizations

### Backend Architecture
- **Core Engine**: Python-based analytical modules in `utils/` directory
- **Load Forecasting**: Machine learning-based forecasting engine (`utils/load_forecasting.py`) using scikit-learn
- **Network Modeling**: PandaPower integration (`utils/pandapower_integration.py`) for electrical network analysis
- **Data Models**: Dataclass-based models for feeder segments and load characteristics (`utils/feeder_model.py`)
- **Profile Generation**: Statistical load profile generator (`utils/load_profiles.py`) for 8760-hour analysis
- **Export Capabilities**: Comprehensive data export utilities (`utils/data_export.py`)

### Data Storage Solutions
- **Session-based Storage**: All data stored in Streamlit session state during application runtime
- **No Persistent Database**: Application operates entirely in memory with export capabilities
- **Data Formats**: Pandas DataFrames for tabular data, NumPy arrays for time series profiles
- **Export Options**: CSV, JSON, and formatted text report generation

### Analysis Capabilities
- **Time Series Analysis**: 8760-hour load profile generation and manipulation
- **Electrical Network Analysis**: AC load flow studies using PandaPower
- **Forecasting Models**: Multi-scenario 10-year load growth projections
- **EV Integration Modeling**: Electric vehicle adoption impact analysis
- **Distributed Generation**: Solar and other DG integration studies
- **Climate Impact Assessment**: Temperature sensitivity and climate change factor modeling

## External Dependencies

### Core Technical Dependencies
- **Streamlit**: Web application framework for utility dashboard interface
- **PandaPower**: Professional electrical network analysis and load flow calculations
- **Pandas/NumPy**: Data manipulation and numerical computing for load profiles
- **Plotly**: Interactive visualization for technical charts and system diagrams
- **Scikit-learn**: Machine learning algorithms for load forecasting models

### Utility Industry Integration
- **Load Profile Standards**: 8760-hour annual load curve methodology
- **Electrical Engineering**: AC power flow analysis with voltage and current calculations
- **Forecasting Methods**: Industry-standard load growth projection techniques
- **Planning Horizons**: 10-year utility planning cycle compatibility

### Data Exchange Formats
- **CSV Export**: Standard utility data interchange format
- **JSON Export**: Modern API-compatible data format
- **Text Reports**: Executive summary and technical report generation
- **Plotly Charts**: Interactive web-based visualization export

### Analysis Standards
- **IEEE Distribution Standards**: Standard voltage levels and equipment ratings
- **Per-Unit System**: Normalized electrical calculations for universal application
- **Time Series Standards**: Hourly resolution with seasonal and daily patterns
- **Scenario Planning**: Multiple forecast scenarios for planning uncertainty
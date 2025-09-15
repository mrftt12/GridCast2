import streamlit as st
import pandas as pd
import os
import tempfile
from datetime import datetime
from pathlib import Path
import asyncio
from analyst import run_full_agent
import base64
from typing import Optional
import plotly.express as px

# for plot formating
import plotly.io as pio
pio.templates["custom"] = pio.templates["seaborn"]
pio.templates.default = "custom"
pio.templates["custom"].layout.autosize = True

# Page configuration
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary directory and return path"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def main():
    st.title("📊 AI Data Analyst")
    st.markdown("Upload your CSV data and get comprehensive analysis with insights!")
    
    
    if st.session_state.uploaded_file_path:
        st.markdown("### File Summary")
        table_summary = pd.read_csv(st.session_state.uploaded_file_path)
        st.write(table_summary.head())
    
    # Sidebar for analysis history and file info
    with st.sidebar:
        st.header("📋 Upload File")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload your dataset in CSV format"
        )
        
        
        if uploaded_file is not None:
            file_path = save_uploaded_file(uploaded_file)
            if file_path:
                st.session_state.uploaded_file_path = file_path
                st.success(f"✅ File uploaded: {uploaded_file.name}")
        
        
        if st.button("Clear History", type="secondary"):
            st.session_state.uploaded_file_path = None
            st.rerun()


    # Analysis query input
    st.subheader("💬 Analysis Query")
    user_query = st.text_area(
        "What would you like to analyze?",
        placeholder="e.g., What are the key trends in the sales data? Show me correlations between different variables.",
        height=120,
        help="Describe what analysis you want to perform on your data"
    )
    
    # Analysis button
    analyze_button = st.button(
        "🔍 Analyze Data", 
        type="primary",
        disabled=not (uploaded_file and user_query),
        help="Upload a file and enter a query to start analysis"
    )
    
    
    if analyze_button:
        if not st.session_state.uploaded_file_path:
            st.error("⚠️ Please upload a CSV file first")
        elif not user_query.strip():
            st.error("⚠️ Please enter an analysis query")
        else:
            with st.spinner("🔄 Analyzing your data... This may take a few minutes."):
                try:
                    # Run analysis asynchronously
                    result = run_full_agent(user_query, st.session_state.uploaded_file_path)
                    
                    if result:
                        st.session_state.current_analysis = result
                        st.session_state.analysis_history.append(result)
                        st.success("✅ Analysis completed successfully!")
                    else:
                        st.error("❌ Analysis failed. Please try again.")
                        
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")

    st.markdown("--------------------------------")
    st.header("📊 Analysis Results")
        
    if st.session_state.current_analysis:
        analysis = st.session_state.current_analysis
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Report", "📈 Metrics", "🖼️ Visualizations", "💡 Conclusion"])
        
        with tab1:
            st.subheader("Analysis Report")
            if analysis.analysis_report:
                st.markdown(analysis.analysis_report)
            else:
                st.warning("No analysis report available")
        
        with tab2:
            st.subheader("Key Metrics")
            if analysis.metrics:
                for i, metric in enumerate(analysis.metrics, 1):
                    st.write(f"{i}. {metric}")
            else:
                st.warning("No metrics calculated")
        
        with tab3:
            st.subheader("Visualizations")
            
            # Try to display HTML first, fallback to PNG
            
            if analysis.image_html_path:
                            try:
                                with open(analysis.image_html_path, 'r', encoding='utf-8') as f:
                                    html_content = f.read()
                                st.components.v1.html(html_content, height=500, scrolling=True)
                            except Exception as e:
                                st.error(f"Error displaying HTML: {str(e)}")

            elif analysis.image_png_path:
                st.image(analysis.image_png_path)
            else:
                st.warning("No visualizations available")
        
        with tab4:
            st.subheader("Conclusion & Recommendations")
            if analysis.conclusion:
                st.markdown(analysis.conclusion)
            else:
                st.warning("No conclusion available")
        
        # Download options
        st.subheader("💾 Download Results")
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            if analysis.analysis_report:
                st.download_button(
                    label="📥 Download Report (MD)",
                    data=analysis.analysis_report,
                    file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        
        with col_download2:
            # Create a summary text for download
            summary_text = f"""
Analysis Summary
================
Query: {user_query}
File: {st.session_state.uploaded_file_path}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Report:
{analysis.analysis_report}

Metrics:
{chr(10).join(f"• {metric}" for metric in analysis.metrics) if analysis.metrics else 'No metrics'}

Conclusion:
{analysis.conclusion}
"""
            st.download_button(
                label="📥 Download Summary (TXT)",
                data=summary_text,
                file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    else:
        st.info("👆 Upload a CSV file and enter your analysis query to get started!")
        
    if st.session_state.analysis_history and len(st.session_state.analysis_history) > 1:
        st.header("📋 Analysis History")
        for analysis in st.session_state.analysis_history[0:len(st.session_state.analysis_history)-1]:
            with st.expander(f"Analysis {analysis.analysis_report[:50]}..."):
                st.markdown(analysis.analysis_report)
                st.image(analysis.image_png_path)

if __name__ == "__main__":
    main()
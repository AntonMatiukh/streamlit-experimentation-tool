"""Minimum sample size page Streamlit app file"""

import streamlit as st  # 🎈 data web app development
import pandas as pd
from pathlib import Path


st.set_page_config(layout="wide")

st.title("🔬 Minimum sample size")

# Read data directly from file or using memory
try:
    df = st.session_state.processed_data
    if df is None:
        st.subheader('Please load your data')
except:
    st.subheader('Please load your data')
    st.stop()


uploaded_f = st.sidebar.file_uploader("Choose a CSV file", type=["csv", "parquet"])

if uploaded_f is not None:
    st.cache_data.clear()
    st.session_state.processed_data = None
    if Path(uploaded_f.name).suffix == '.parquet':
        df = pd.read_parquet(uploaded_f)
    elif Path(uploaded_f.name).suffix == '.csv':
        df = pd.read_csv(uploaded_f)
    st.session_state.processed_data = df

if df is not None:
    level_1_column_1, level_1_column_2, level_1_column_3, level_1_column_4 = st.columns(4)

    with level_1_column_1:
        p_value = st.number_input(
            "Significance level:",
            value=0.05
        )

    with level_1_column_2:
        power = st.number_input(
            "Power:",
            value=0.8
        )

    with level_1_column_3:
        data_input_option = st.selectbox(
            "Use data or input manually?",
            ("Autofill", "Manually")
        )

    if data_input_option == "Manually":
        with level_1_column_4:
            metric_type = st.selectbox(
                "Choose experiment metric type:",
                ("Conversion", "Continuous", "Ratio")
            )


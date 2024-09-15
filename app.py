"""Main Streamlit app file"""

import streamlit as st  # 🎈 data web app development
import time
import pandas as pd
import datetime
from pathlib import Path


st.set_page_config(layout="wide")

# Hello world
st.title("Hi dear friend 🎈")

st.subheader("Please upload your CSV file with experiment data (format example below)")

# Create df example to describe structure
df_example = pd.DataFrame({'user_id':['test_12345'],
                           'experiment_exposure':[datetime.datetime(2024, 9, 1, 10, 5, 0)],
                           'timestamp':[datetime.datetime(2024, 9, 1, 13, 15, 35)],
                           'value':[12.6]})

st.write(df_example)

# Load DataFrame

if "processed_data" not in st.session_state:
    st.session_state.processed_data = None

uploaded_f = st.sidebar.file_uploader("Choose a CSV file", type=["csv", "parquet"])

if uploaded_f is not None:
    st.cache_data.clear()
    st.session_state.processed_data = None
    if Path(uploaded_f.name).suffix == '.parquet':
        df = pd.read_parquet(uploaded_f)
    elif Path(uploaded_f.name).suffix == '.csv':
        df = pd.read_csv(uploaded_f)
    st.session_state.processed_data = df

if st.session_state.processed_data is None:
    st.subheader('Please load your data')
else:
    st.subheader("Your DataFrame example")
    st.write(st.session_state.processed_data.head())

# st.sidebar.image('sidebar_gif.gif', use_column_width=True)

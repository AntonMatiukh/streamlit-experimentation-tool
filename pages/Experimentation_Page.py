"""Experimentation page Streamlit app file"""

import time  # to simulate a real time data, time loop
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import plotly.graph_objects as go
import plotly.figure_factory as ff
import json
from pathlib import Path

st.set_page_config(layout="wide")

st.title("📈 Experimentation")


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
    left_co, cent_co, last_co = st.columns([1, 3, 1])
    with cent_co:
        gif_runner = st.image('funny_pink_cat.gif',
                              use_column_width=True,
                              # width=400
                              )

    # Make some files transformation and preparation
    @st.cache_data
    def process_df(df):
        df_function = df.copy()
        df_function['is_after_experiment'] = np.where(pd.to_datetime(df_function['timestamp']) > \
                                                      pd.to_datetime(df_function['experiment_exposure']),
                                                      1,
                                                      0)
        df_function['lifetime'] = (pd.to_datetime(df_function['timestamp']) - \
                                   pd.to_datetime(df_function['experiment_exposure'])).dt.days
        df_function['max_lifetime'] = (pd.to_datetime('today') - pd.to_datetime(df_function['experiment_exposure'])).dt.days
        df_function['experiment_exposure_dt'] = pd.to_datetime(df_function['experiment_exposure']).dt.date
        df_function['dt'] = pd.to_datetime(df_function['timestamp']).dt.date
        df_function = df_function[(abs(df_function['lifetime']) < 3)
                                    & (df_function['max_lifetime'] >= df_function['lifetime'])]
        df_function = df_function[df_function['is_after_experiment'] == 1]
        df_function = df_function.sort_values(by=['user_id', 'timestamp'])
        return df_function

    try:
        df_transformed = process_df(df)
    except:
        st.error('Please load correct format DataFrame')

    gif_runner.empty()

    df_grouped = df_transformed.groupby(['user_id','variation','experiment_exposure_dt'],
                                        as_index=False)['value'].sum()\
                    .groupby(['experiment_exposure_dt','variation']).agg({'user_id':'nunique',
                                                                          'value':'sum'}).reset_index()
    df_grouped['user_id'] = df_grouped.groupby('variation')['user_id'].cumsum()
    df_grouped['value'] = df_grouped.groupby('variation')['value'].cumsum()
    df_grouped['mean'] = df_grouped['value'] / df_grouped['user_id']
    df_grouped['mean_shift'] = df_grouped.groupby('experiment_exposure_dt')['mean'].shift(1)

    st.write(df_grouped)


    # progress_bar = st.progress(0)
    # status_text = st.empty()
    #
    # chart = st.line_chart(np.zeros(shape=(1, 1)))
    # x = np.arange(0, 100 * np.pi, 0.1)
    #
    # for i in range(1, 101):
    #     y = np.sin(x[i])
    #     status_text.text("%i%% Complete" % i)
    #     chart.add_rows([y])
    #     progress_bar.progress(i)
    #     time.sleep(0.05)
    #
    # progress_bar.empty()
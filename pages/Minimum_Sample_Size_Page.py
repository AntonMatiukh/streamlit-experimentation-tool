"""Minimum sample size page Streamlit app file"""

import streamlit as st  # 🎈 data web app development
import pandas as pd
from pathlib import Path
from src.stats.tests.ttest import TTest
import numpy as np
import time


st.set_page_config(layout="wide")

st.title("🔬 Minimum sample size")

uploaded_f = st.sidebar.file_uploader("Choose a CSV file", type=["csv", "parquet"])

if uploaded_f is not None:
    st.cache_data.clear()
    st.session_state.processed_data = None
    if Path(uploaded_f.name).suffix == '.parquet':
        df = pd.read_parquet(uploaded_f)
    elif Path(uploaded_f.name).suffix == '.csv':
        df = pd.read_csv(uploaded_f)
    st.session_state.processed_data = df

level_1_column_1, level_1_column_2, level_1_column_3, level_1_column_4 = st.columns(4)

# Select level of significance
with level_1_column_1:
    alpha = st.number_input(
        "Significance level:",
        value=0.05
    )

# Select power level
with level_1_column_2:
    power = st.number_input(
        "Power:",
        value=0.8
    )

# Select percent in control group
with level_1_column_3:
    r = st.number_input(
        "Control group percent:",
        value=0.5
    )

# Is one side test
with level_1_column_4:
    is_one_side = st.selectbox(
        "Is one side test?",
        (1, 0)
    )

level_2_column_1, level_2_column_2 = st.columns(2)

# Select how to input data
with level_2_column_1:
    data_input_option = st.selectbox(
        "Use data or input manually?",
        ("Manually", "Autofill")
    )

if data_input_option == "Manually":
    with level_2_column_2:
        metric_type = st.selectbox(
            "Choose experiment metric type:",
            ("Conversion", "Continuous", "Ratio")
        )

    level_3_column_1, level_3_column_2, level_3_column_3, level_3_column_4 = st.columns(4)

    if metric_type == "Conversion":
        with level_3_column_1:
            p = st.number_input(
                "Conversion rate",
                value=0.25
            )

        if st.button('Calculate Sample Size'):
            df_min_sample_size = TTest().min_sample_size_df(p=p,
                                                            alpha=alpha,
                                                            power=power,
                                                            is_one_side=is_one_side,
                                                            r=r)
            df_min_sample_size = pd.pivot(columns="uplift", index="test_name", values="n", data=df_min_sample_size)
            st.dataframe(df_min_sample_size)

    elif metric_type == "Continuous":
        with level_3_column_1:
            mean = st.number_input(
                "Mean",
                value=3
            )

        with level_3_column_2:
            std = st.number_input(
                "std",
                value=2
            )

        mu = np.log(mean ** 2 / np.sqrt(std ** 2 + mean ** 2))
        sigma = np.sqrt(np.log(1 + (std ** 2 / mean ** 2)))

        progress_bar = st.progress(0)
        status_text = st.empty()

        chart = st.line_chart(np.zeros(shape=(1, 1)))
        x = np.arange(0, 101, 1)

        control_list = []

        for i in range(0, 101):
            control_list.append(np.random.lognormal(mean=mu, sigma=sigma, size=1_000))
            control_tmp = np.concatenate(control_list)

            y = np.mean(control_tmp)
            status_text.text("%i%% Complete" % i)
            chart.add_rows([y])
            progress_bar.progress(i)
            time.sleep(0.05)

        progress_bar.empty()

else:

    # Read data directly from file or using memory
    try:
        df = st.session_state.processed_data
        if df is None:
            st.subheader('Please load your data')
    except:
        st.subheader('Please load your data')
        st.stop()

    if df is not None:
        st.text('Autodata')









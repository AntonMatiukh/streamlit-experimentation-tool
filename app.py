"""Main Streamlit app file"""

import streamlit as st  # 🎈 data web app development
import time


st.set_page_config(layout="wide")

st.title("🎈 Navigation to Streamlit dashboards pages")

st.subheader("🔬 Minimum sample size")
st.subheader("📈 Experimentation")

# Показываем анимацию
st.image('funny_cat.gif', use_column_width=True)

# Выполняем долгую операцию
def long_running_function():
    # Имитация долгой работы
    time.sleep(60)

long_running_function()

st.success('Done!')

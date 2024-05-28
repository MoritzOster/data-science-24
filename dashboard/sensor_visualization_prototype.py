import streamlit as st
import pandas as pd
import numpy as np
import time

@st.cache_data
def load_data(file_path):
    return pd.read_parquet(file_path)

def simulate_real_time_data(data, batch_size):
    for start in range(0, len(data), batch_size):
        end = min(start + batch_size, len(data))
        yield data.iloc[start:end]

file_path = './2024.02.14_22.00.40_Grinding/raw/Sampling2000KHz_AEKi-0.parquet'

data = load_data(file_path)

st.title('Real-Time AE Signal Visualization')
st.write("Simulating real-time data from AE measurements.")

chart = st.line_chart()

batch_size = 2000 * 1  # 1 second worth of data
total_points = len(data)
current_time = 0

for batch in simulate_real_time_data(data, batch_size=batch_size):
    chart.add_rows(batch)
    current_time += len(batch) / 2000  
    time.sleep(0.001)  
st.write("Simulation complete.")

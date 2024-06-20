import streamlit as st
import pandas as pd
import numpy as np
import time

@st.cache_data
def load_data(file_path):
    return pd.read_parquet(file_path)


file_path = './2024.02.14_22.00.40_Grinding/raw/Sampling2000KHz_AEKi-0.parquet'
data = load_data(file_path)

st.title('Real-Time AE Signal Visualization')
st.write("Simulating real-time data from AE measurements.")

chart = st.line_chart()


downsample_factor = 1000
sampling_rate = 2e6
# Calculate sleep time based on desired animation speed
sleep_time = downsample_factor / sampling_rate  

downsampled_data = data[::downsample_factor]

total_points_after_downsampling = len(downsampled_data)
current_index = 0

update_batch_size = 100
update_batch = []
# progress = st.progress(0)  # Initialize progress bar

while current_index < total_points_after_downsampling:
    current_measurement = downsampled_data.iloc[current_index]
    update_batch.append(current_measurement)
    current_index += 1
    # Sleep to simulate real-time data
    time.sleep(sleep_time)

    if len(update_batch) < update_batch_size:
            continue

    chart.add_rows(update_batch)  
    update_batch = []  
    


st.write("Simulation complete.")

from data_provider import DataProvider
import streamlit as st
import pandas as pd
import time
import altair as alt

def visualize_measurement(file_path, sampling_rate, downsample_factor, update_batch_size):
    data_provider = DataProvider(file_path, sampling_rate, downsample_factor)
    chart = st.line_chart()
    update_batch = []    
    while (1):
        next_measurement = data_provider.next()
        if next_measurement is None:
            break
        update_batch.append(next_measurement)
        if len(update_batch) < update_batch_size:
            continue
        chart.add_rows(update_batch)  
        update_batch = []



# Path to the .parquet file
file_path = './2024.02.14_22.00.40_Grinding/raw/Sampling2000KHz_AEKi-0.parquet'
sampling_rate = 2e6
downsample_factor = 1000
update_batch_size = 100

visualize_measurement(file_path, sampling_rate, downsample_factor, update_batch_size)
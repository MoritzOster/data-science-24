import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from data_provider import DataProvider

# Path to the .parquet file
file_path = './2024.02.14_22.00.40_Grinding/raw/Sampling2000KHz_AEKi-0.parquet'

# Initialize DataProvider with the desired parameters
data_provider = DataProvider(file_path, sampling_rate=2000000, downsample_factor=1000)

# Streamlit UI
st.title('Real-Time AE Signal Visualization')
st.write("Simulating real-time data from AE measurements.")

# Print the values of the data_provider object
st.write(f"Total points: {data_provider.start}")
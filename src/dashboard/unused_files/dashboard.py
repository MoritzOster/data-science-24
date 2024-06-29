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

# Initialize Plotly figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=[], y=[], mode='lines'))

# Create a placeholder for the plot
plotly_chart = st.plotly_chart(fig, use_container_width=True)

# Define the window size in seconds
window_size_seconds = 5

# Function to update the plot with a sliding window of the last N seconds
def update_plot(fig, new_data, timestamp):
    x_values = list(fig.data[0].x)
    y_values = list(fig.data[0].y)
    
    # Append the new data point
    x_values.append(timestamp)
    y_values.append(new_data)

    # Keep only the data points within the window size
    min_timestamp = timestamp - window_size_seconds
    x_values = [x for x in x_values if x >= min_timestamp]
    y_values = y_values[-len(x_values):]

    # Assign the updated lists back to the figure
    fig.data[0].x = x_values
    fig.data[0].y = y_values

# Simulate real-time data
for data_point in data_provider.start():
    timestamp = data_point.name / 2000000  # Assuming the index is used as timestamp
    update_plot(fig, data_point, timestamp)
    plotly_chart.plotly_chart(fig, use_container_width=True)

st.write("Simulation complete.")

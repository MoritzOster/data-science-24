import streamlit as st
import pandas as pd
import time
import plotly.graph_objs as go

@st.cache_data
def load_data(file_path):
    return pd.read_parquet(file_path)

file_path = './2024.02.14_22.00.40_Grinding/raw/Sampling2000KHz_AEKi-0.parquet'
data = load_data(file_path)

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

downsample_factor = 1000
sampling_rate = 2e6
# Calculate sleep time based on desired animation speed
sleep_time = downsample_factor / sampling_rate  
downsampled_data = data[::downsample_factor]

total_points_after_downsampling = len(downsampled_data)
current_index = 0
update_batch_size = 100
update_batch = []

while current_index < total_points_after_downsampling:
    current_measurement = downsampled_data.iloc[current_index]
    update_batch.append(current_measurement)
    current_index += 1
    # Sleep to simulate real-time data
    time.sleep(sleep_time)

    if len(update_batch) < update_batch_size:
        continue

    timestamp = current_measurement.name / 2000000  
    update_plot(fig, current_measurement, timestamp)
    plotly_chart.plotly_chart(fig, use_container_width=True)

    update_batch = []

st.write("Simulation complete.")

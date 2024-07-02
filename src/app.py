import sys
import os
import csv
from data_provider import DataProvider
from recording_provider import DataProviderProvider
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
from prodetect import prodetect_predict
from datetime import datetime


data_path= '../data/example_recordings'
dataProviderProvider = DataProviderProvider(data_path, downsample_factor=100000)

# Assuming data_provider is already imported and configured
def load_data(data_provider):
    while True:
        data = data_provider.next()
        yield data

def plot_data(data_provider):
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'b-')  # Initialize line object
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.25, 0.25)  # Adjust based on your data range
    placeholder = st.empty()  # Placeholder for the figure

    start_time = time.time()
    while True:
        data = next(load_data(data_provider), None)
        if data is None:
            break
        elapsed_time = time.time() - start_time
        if elapsed_time > 60:  # Stop after 60 seconds
            break
        line.set_xdata(np.append(line.get_xdata(), elapsed_time))
        line.set_ydata(np.append(line.get_ydata(), data[0]))
        ax.figure.canvas.draw_idle()
        placeholder.pyplot(fig)


st.title('Real-time Data Streaming')
def run():
    data_provider = dataProviderProvider.get_any_data_provider()
    status_placeholder = st.empty()
    plot_data(data_provider)

    anomaly_prediction = prodetect_predict(data_provider.get_ae_path())
    print(anomaly_prediction)
    result = anomaly_prediction
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_filename = 'predictions.csv'

    # Write prediction results to CSV
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if file is empty
        if file.tell() == 0:
            writer.writerow(['Timestamp', 'AnomalyPrediction', 'Path'])
        writer.writerow([timestamp, result[0], data_provider.get_current_path()])

    if result[0]:
        status_placeholder.error('**Failure!**', icon="🚨")
    else:
        status_placeholder.success('**Success!**', icon="✅")

while True: 
    run()
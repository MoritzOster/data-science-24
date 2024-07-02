import sys
import os
import csv
from data_provider import DataProvider
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
from prodetect import prodetect_predict
from datetime import datetime

# Calculate the path to the src directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(base_dir, 'src')
sys.path.insert(0, src_path)

data_path_NOK = '../data/example_recordings/NOK_Measurements'
data_path_OK = '../data/example_recordings/OK_Measurements'

# example_ae_path = data_path_OK + '/2024.02.14_22.00.40_Grinding'
example_ae_path = data_path_NOK + '/2024.02.15_02.27.22_Grinding'
# data/example_recordings/NOK_Measurements/2024.02.15_02.27.22_Grinding
data_provider = DataProvider(example_ae_path, downsample_factor=100000)

# Assuming data_provider is already imported and configured
def load_data():
    while True:
        data = data_provider.next()
        yield data

def plot_data():
    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'b-')  # Initialize line object
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.25, 0.25)  # Adjust based on your data range
    placeholder = st.empty()  # Placeholder for the figure

    start_time = time.time()
    while True:
        data = next(load_data(), None)
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
status_placeholder = st.empty()
plot_data()

anomaly_prediction = prodetect_predict(example_ae_path + '/raw/Sampling2000KHz_AEKi-0.parquet')
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
    writer.writerow([timestamp, result[0], example_ae_path])

if result[0]:
    status_placeholder.error('**Failure!**', icon="🚨")
else:
    status_placeholder.success('**Success!**', icon="✅")
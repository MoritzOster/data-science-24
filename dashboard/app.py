import sys
import os
from data_provider import DataProvider
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
# Calculate the path to the src directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(base_dir, 'src')
sys.path.insert(0, src_path)
from prodetect import prodetect_predict





data_provider = DataProvider('./2024.02.14_22.00.40_Grinding', downsample_factor=100000)




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
        if elapsed_time > 60:  # Stop after 20 seconds
            break
        line.set_xdata(np.append(line.get_xdata(), elapsed_time))
        line.set_ydata(np.append(line.get_ydata(), data[0]))
        ax.figure.canvas.draw_idle()
        placeholder.pyplot(fig)
        # time.sleep(0.02)  # Adjust based on desired streaming rate

st.title('Real-time Data Streaming')
status_placeholder = st.empty()
plot_data()

result = output = prodetect_predict("C:/Users/ebadu/OneDrive/Documents/data-science-24/dashboard/2024.02.14_22.00.40_Grinding/raw/Sampling2000KHz_AEKi-0.parquet")
if result[0]:
    status_placeholder.error('**Failure!**', icon="🚨")
else:
    status_placeholder.success('**Success!**', icon="✅")
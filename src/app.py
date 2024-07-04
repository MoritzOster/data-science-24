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
import pandas as pd
import time

st.set_page_config(layout="wide")
data_path= '../data/example_recordings'
dataProviderProvider = DataProviderProvider(data_path, downsample_factor=100000)

def load_data(data_provider):
    while True:
        data = data_provider.next()
        yield data

def plot_data(data_provider, progress_bar):
    
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    lines = [ax.plot([], [], 'b-')[0] for ax in axs]
    
    x_labels = ['Time (s)', 'Time (s)', 'Time (s)', 'Time (s)']
    y_labels = ['Sampling2000KHz AEKi', 'Grinding spindle current L1', 'Grinding spindle current L2', 'Grinding spindle current L3']
    titles = ['AEKi w.r.t Time', 'Current L1 w.r.t Time', 'Current L2 w.r.t Time', 'Current L3 w.r.t Time']
    
    # Set labels and titles for all subplots
    for ax, x_label, y_label, title in zip(axs, x_labels, y_labels, titles):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.set_xlim(0, 60)  # Adjust as needed
        ax.set_ylim(-0.25, 0.25)  # Adjust as needed
    
    # Placeholder for the figure
    placeholder = st.empty()
    
    start_time = time.time()
    while True:
        data = next(load_data(data_provider), None)
        if data is None:
            break
        
        elapsed_time = time.time() - start_time
        if elapsed_time > 60:
            break
        progress_bar.progress(10)
        
        for i, line in enumerate(lines):
            line.set_xdata(np.append(line.get_xdata(), elapsed_time))
            line.set_ydata(np.append(line.get_ydata(), data[i]))
        
        for ax in axs:
            ax.figure.canvas.draw_idle()
        
        placeholder.pyplot(fig)


st.title('Real-time Data Streaming')
def read_and_plot_last_predictions(prediction_placeholder):
    csv_filename = 'predictions.csv'
    try:
        df = pd.read_csv(csv_filename)
        last_predictions = df['AnomalyPrediction'].tail(10).tolist()
        colors = ['red' if pred else 'green' for pred in last_predictions][::-1]

        fig, ax = plt.subplots(figsize=(10, 2))
        bars = ax.barh([0]*10, [1]*10, left=range(10), color=colors, height=0.2, edgecolor='black', linewidth=0.4)
        ax.axis('off') 
        ax.set_xlim(-0.5, 9.5) 
        prediction_placeholder.pyplot(fig)
    except Exception as e:
        st.error(f"Failed to read or plot predictions: {e}")

# def run():
#     data_provider = dataProviderProvider.get_any_data_provider()
#     with st.container():
#         col1, col2, col3, col4 = st.columns([4,1, 1, 1])
#         ee = col4.empty()
#         status_placeholder = col3.empty()
#         process_placeholder = col2.empty()
#         progress_bar = col1.progress(0)

#     prediction_placeholder = st.empty()
    
#     plot_data(data_provider)
#     progress_bar.progress(25)
#     anomaly_prediction = prodetect_predict(data_provider.get_ae_path())
#     result = anomaly_prediction[0]
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     csv_filename = 'predictions.csv'
#     progress_bar.progress(75)
#     with open(csv_filename, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         if file.tell() == 0:
#             writer.writerow(['Timestamp', 'AnomalyPrediction', 'Path'])
#         writer.writerow([timestamp, result, data_provider.get_current_path()])
#     progress_bar.progress(100)
#     process_placeholder.markdown('<h1 style="color: green;">✅</h1>', unsafe_allow_html=True)
#     if result:
#         status_placeholder.error('**Failure!**', icon="🚨")
#     else:
#         status_placeholder.success('**Success!**', icon="✅")
    
    # read_and_plot_last_predictions(prediction_placeholder)

def run():
    data_provider = dataProviderProvider.get_any_data_provider()

    with st.container():
        col1, col2 = st.columns([4, 1])

        with col2:
            progress_bar = st.progress(0)
            process_placeholder = st.empty()
            status_placeholder = st.empty()
            prediction_placeholder = st.empty()

        with col1:
            progress_bar.progress(1)
            
            plot_data(data_provider, progress_bar)
            progress_bar.progress(25)

        with col1:
            progress_bar.progress(35)
            anomaly_prediction = prodetect_predict(data_provider.get_ae_path())
            result = anomaly_prediction[0]
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            csv_filename = 'predictions.csv'
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(['Timestamp', 'AnomalyPrediction', 'Path'])
                writer.writerow([timestamp, result, data_provider.get_current_path()])
            time.sleep(1)
            progress_bar.progress(50) 

        with col1:
            time.sleep(1)
            progress_bar.progress(75) 
            # process_placeholder.markdown('<h1 style="color: green;">✅</h1>', unsafe_allow_html=True)


        with col2:
            progress_bar.progress(100) 
            if result:
                status_placeholder.error('**Anomaly Detected**')
            else:
                status_placeholder.success('**No Anomaly Detected**')
            read_and_plot_last_predictions(prediction_placeholder)
def main():
    # if 'first_click' not in st.session_state:
    #     st.session_state.first_click = True
    # button_text = "Predict" if st.session_state.first_click else "Next Prediction"
    if st.button('Predict'):
        # st.session_state.first_click = False
        run()
        

main()
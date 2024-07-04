import sys
import os
import csv

import pandas as pd
from data_provider import DataProvider
from recording_provider import DataProviderProvider
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
from prodetect import prodetect_predict
from collections import deque
from datetime import datetime
import pandas as pd
import time

st.set_page_config(layout="wide")
data_path= '../data/example_recordings'
dataProviderProvider = DataProviderProvider(data_path, downsample_factor=2000000)

def load_data(data_provider):
    while True:
        data = data_provider.next()
        yield data

def plot_data(data_provider, placeholder):
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs = axs.flatten()
    
    lines = [ax.plot([], [], 'b-')[0] for ax in axs]
    
    x_labels = ['Time (s)', 'Time (s)', 'Time (s)', 'Time (s)']
    y_labels = ['Sampling2000KHz AEKi', 'Grinding spindle current L1', 'Grinding spindle current L2', 'Grinding spindle current L3']
    titles = ['AEKi w.r.t Time', 'Current L1 w.r.t Time', 'Current L2 w.r.t Time', 'Current L3 w.r.t Time']
    
    # Set labels and titles for all subplots
    for ax, x_label, y_label, title in zip(axs, x_labels, y_labels, titles):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.set_xlim(0, 25)  # Adjust as needed
        ax.set_ylim(-0.25, 0.25)  # Adjust as needed

    data_gen = load_data(data_provider)

    x_data = []
    y_data = [[], [], [], []]    
    
    # start_time = time.time()
    while True:
        data = next(data_gen, None)
        if data is None:
            break

        # elapsed_time = time.time() - start_time
        # if elapsed_time > 25:
        #     break
        # progress_bar.progress(10)

        y_data[0].append(data[0])
        y_data[1].append(data[2])
        y_data[2].append(data[3])
        y_data[3].append(data[4])
        x_data.append(data[5])
        
        for i, line in enumerate(lines):
            line.set_xdata(x_data)
            line.set_ydata(y_data[i])
        
        for ax in axs:
            ax.figure.canvas.draw_idle()
        
        placeholder.pyplot(fig)


def init_table(placeholder):
    predictions = deque(maxlen=10)
    last_10 = pd.read_csv('predictions.csv').tail(10)
    last_10 = last_10.drop(columns=['Path'])
    last_10['AnomalyPrediction'] = last_10['AnomalyPrediction'].apply(lambda x: '🚨' if x else '✅')
    last_10 = last_10.rename(columns={'AnomalyPrediction': 'Result'})
    last_10 = last_10.iloc[::-1]

    for row in last_10.itertuples(index=False):
        predictions.append(row)

    update_table(placeholder, predictions)

    return predictions

def update_table(placeholder, preds):
    with placeholder.container():
        df = pd.DataFrame(preds, columns=['Timestamp', 'Result'])
        df = df.set_index('Timestamp')
        st.dataframe(df)

def run(plot_p, image_p, status_p, last_process_p, last_10_p, preds):
    data_provider_instance = dataProviderProvider.get_any_data_provider()

    status_p.warning('Recording...')
    # progress_p.progress(1)
    plot_data(data_provider_instance, plot_p)
    # progress_p.progress(25)
    image_p.image(data_provider_instance.image_path)

    status_p.warning('Processing...')
    # progress_p.process(35)
    anomaly_prediction = prodetect_predict(data_provider_instance.get_ae_path())
    status_p.success('Done!')
    # print(anomaly_prediction)
    result = anomaly_prediction
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_filename = 'predictions.csv'

    # Write prediction results to CSV
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if file is empty
        if file.tell() == 0:
            writer.writerow(['Timestamp', 'AnomalyPrediction', 'Path'])
        writer.writerow([timestamp, result[0], data_provider_instance.get_current_path()])

    if result[0]:
        last_process_p.error('**Failure!**', icon="🚨")
        preds.appendleft([timestamp, "🚨"])
    else:
        last_process_p.success('**Success!**', icon="✅")
        preds.appendleft([timestamp, "✅"])


    update_table(last_10_p, preds)

    time.sleep(3)


def main():
    st.title('Grinding Process Anomaly Detection')

    # Create two columns
    col1, col2 = st.columns([4, 5])

    with col1:
        st.text('Current process recordings:')
        plot_placeholder = st.empty()
        st.text('Reference for debugging:')
        image_placeholder = st.empty()

    with col2:
        st.text('Detection status:')
        # progress_bar = st.progress(0)
        status_placeholder = st.empty()
        st.text('Prediction of last process:')
        last_process_placeholder = st.warning('Unknown')
        st.text('Last 10 processes:')
        last_10_placeholder = st.empty()
        predictions = init_table(last_10_placeholder)

    while True: 
        run(
            plot_placeholder, 
            image_placeholder,
            status_placeholder, 
            last_process_placeholder,
            last_10_placeholder, 
            predictions
        )

if __name__ == "__main__":
    main()

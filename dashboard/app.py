import csv

import pandas as pd
from recording_provider import DataProviderProvider
import streamlit as st
import matplotlib.pyplot as plt
import time
import pickle
from collections import deque
from datetime import datetime
import pandas as pd
import time
import numpy as np

st.set_page_config(layout="wide")
data_path= '../data/example_recordings'
dataProviderProvider = DataProviderProvider(data_path, cluster_factor=1000, downsample_factor=100)

def plot_data(data_provider, plot_placeholder, status):

    scaler = pickle.load(open("../data/scaler.pkl", "rb"))
    model = pickle.load(open("../data/logistic_regression_model.pkl", "rb"))
    
    fig, axs = plt.subplots(4, 1, figsize=(9, 10))
    plt.subplots_adjust(hspace=0.9)

    lines = [ax.plot([], [], color='cornflowerblue')[0] for ax in axs]
    
    x_labels = ['Time (ms)', 'Time (ms)', 'Time (ms)', 'Time (ms)']
    y_labels = ['Sampling2000KHz AEKi', 'Grinding spindle current L1', 'Grinding spindle current L2', 'Grinding spindle current L3']
    titles = ['AEKi w.r.t Time', 'Current L1 w.r.t Time', 'Current L2 w.r.t Time', 'Current L3 w.r.t Time']
    
    # Set labels and titles for all subplots
    for ax, x_label, y_label, title in zip(axs, x_labels, y_labels, titles):
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.set_xlim(0, 25000) 
        ax.set_ylim(-0.4, 0.4)

    x = []
    y_ae = []
    y_l1 = []
    y_l2 = []
    y_l3 = []

    while True:
        ae_data, current_data, grinding_l1_data, grinding_l2_data, grinding_l3_data, x_data = data_provider.next()
        if len(ae_data) == 0:
            break

        x.extend(x_data)

        for i, line in enumerate(lines):
            line.set_xdata(x)
            if i == 0:
                y_ae.extend(ae_data)
                line.set_ydata(y_ae)
            elif i == 1:
                y_l1.extend(grinding_l1_data)
                line.set_ydata(y_l1)
            elif i == 2:
                y_l2.extend(grinding_l2_data)
                line.set_ydata(y_l2)
            elif i == 3:
                y_l3.extend(grinding_l3_data)
                line.set_ydata(y_l3)

        for ax in axs:
            ax.figure.canvas.draw_idle()

        # Update prediction
        ae_std = np.array([np.std(y_ae)]).reshape(1, -1)
        feature = scaler.transform(ae_std)
        if (model.predict(feature)):
            status.error('Possible anomaly detected!') 
        
        plot_placeholder.pyplot(fig)


    ae_std = np.array([np.std(y_ae)]).reshape(1, -1)
    feature = scaler.transform(ae_std)
    return model.predict(feature)

def init_table(df_placeholder, last_pred_p):
    predictions = deque(maxlen=10)
    last_10 = pd.read_csv('predictions.csv').tail(10)
    last_10 = last_10.drop(columns=['Path'])
    if last_10['AnomalyPrediction'].iloc[9]:
        last_pred_p.error('**Failure!**', icon="🚨")
    else:
        last_pred_p.success('**Success!**', icon="✅")

    last_10['AnomalyPrediction'] = last_10['AnomalyPrediction'].apply(lambda x: '🚨' if x else '✅')
    last_10 = last_10.rename(columns={'AnomalyPrediction': 'Result'})
    last_10 = last_10.iloc[::-1]

    for row in last_10.itertuples(index=False):
        predictions.append(row)

    update_table(df_placeholder, predictions)

    return predictions

def update_table(placeholder, preds):
    with placeholder.container():
        df = pd.DataFrame(preds, columns=['Timestamp', 'Result'])
        st.dataframe(df, use_container_width=True, hide_index=True)

def update_statistics(placeholder):
    p_left, p_right = placeholder

    colors = ['#66b3ff', '#ff9999']

    history = pd.read_csv('predictions.csv')
    history['Timestamp'] = pd.to_datetime(history['Timestamp'])
    last_24_hours = history[history['Timestamp'] >= (pd.Timestamp.now() - pd.Timedelta(hours=24))]
    num_anom_24 = last_24_hours['AnomalyPrediction'].sum()
    num_proc_24 = len(last_24_hours)
    num_anomalies = history['AnomalyPrediction'].sum()
    num_processes = len(history)

    fig_24, ax_24 = plt.subplots()
    ax_24.pie([num_proc_24 - num_anom_24, num_anom_24], radius=0.8, colors=colors)
    fig_total, ax_total = plt.subplots()
    ax_total.pie([num_processes - num_anomalies, num_anomalies], radius=0.8, colors=colors)

    with p_left.container():
        st.text('Anomalies in the \nlast 24h:')
        st.markdown(f"<h1 style='text-align: center; font-size: 50px;'>{num_anom_24}</h1>", unsafe_allow_html=True)
        st.text('Processes in the \nlast 24h:')
        st.markdown(f"<h1 style='text-align: center; font-size: 50px;'>{num_proc_24}</h1>", unsafe_allow_html=True)
        st.pyplot(fig_24)

    with p_right.container():
        st.text('Total number of \nanomalies:')
        st.markdown(f"<h1 style='text-align: center; font-size: 50px;'>{num_anomalies}</h1>", unsafe_allow_html=True)
        st.text('Total number of \nprocesses:')
        st.markdown(f"<h1 style='text-align: center; font-size: 50px;'>{num_processes}</h1>", unsafe_allow_html=True)
        st.pyplot(fig_total)

def run(plot_p, status_p, last_process_p, last_10_p, preds, stats_p):
    data_provider_instance = dataProviderProvider.get_any_data_provider()

    status_p.warning('No anomaly detected so far.')
    result = plot_data(data_provider_instance, plot_p, status_p)
    status_p.warning('Grinding process completed.')    
    # image_p.image(data_provider_instance.image_path)

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
    update_statistics(stats_p)

    time.sleep(4)


def main():

    # Create two columns
    _, col1, _, col2, _ = st.columns([1, 8, 1, 8, 1])

    with col1:
        st.title('Grinding Process Anomaly Detection')
        st.text('Current process recordings:')
        plot_placeholder = st.empty()
        # st.text('Reference for debugging:')
        # image_placeholder = st.empty()

    with col2:
        st.title('')
        st.text('Detection status:')
        status_placeholder = st.empty()
        st.text('Prediction of last process:')
        last_process_placeholder = st.warning('Unknown')

        st.title('')

        col3, col4, col5 = st.columns([1, 1, 1])

    with col3:
        stats_placeholder_left = st.empty()
    
    with col4:
        stats_placeholder_right = st.empty()
        
    with col5:
        st.text('Last 10 processes:')
        last_10_placeholder = st.empty()
    
    stats_placeholder = stats_placeholder_left, stats_placeholder_right
    
    predictions = init_table(last_10_placeholder, last_process_placeholder)
    update_statistics(stats_placeholder)

    while True: 
        run(
            plot_placeholder,
            # image_placeholder,
            status_placeholder, 
            last_process_placeholder,
            last_10_placeholder, 
            predictions,
            stats_placeholder
        )

if __name__ == "__main__":
    main()

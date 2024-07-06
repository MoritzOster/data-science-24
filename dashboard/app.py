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
    
    fig, axs = plt.subplots(4, 1, figsize=(8, 9))
    plt.subplots_adjust(hspace=0.8)

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
        st.dataframe(df, use_container_width=True, hide_index=True)

def run(plot_p, image_p, status_p, last_process_p, last_10_p, preds):
    data_provider_instance = dataProviderProvider.get_any_data_provider()

    status_p.warning('No anomaly detected so far.')
    result = plot_data(data_provider_instance, plot_p, status_p)
    status_p.warning('Grinding process completed.')    
    image_p.image(data_provider_instance.image_path)

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

    time.sleep(4)


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
        status_placeholder = st.empty()
        st.text('Prediction of last process:')
        last_process_placeholder = st.warning('Unknown')

        col3, col4 = st.columns([2, 1])

    with col3:
        stats_placeholder = st.empty()
        
    with col4:
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

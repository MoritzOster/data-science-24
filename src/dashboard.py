from data_provider import DataProvider
import streamlit as st
from amplitude_graph import AmplitudeGraph

data_provider = DataProvider('./2024.02.14_22.00.40_Grinding', downsample_factor=2000)
meta = data_provider.meta_data()
st.set_page_config(layout="wide")
process_type = meta['process']
title_placeholder = st.empty()
title_placeholder.title(f"Ongoing {process_type} Process")
st.write("<span style='font-size: 24px;'>AE Measurement</span>", unsafe_allow_html=True)
ae_anomaly_placeholder = st.empty()
ae_graph = AmplitudeGraph(st, update_batch_size=100)
st.write("<span style='font-size: 24px;'>LRMS of the Current Measurement</span>", unsafe_allow_html=True)
current_anomaly_placeholder = st.empty()
lrms_grinding_graph = AmplitudeGraph(st, update_batch_size=100)

# anomaly detection messages
ae_anomaly_detected = "<span style='color:red; font-size: 20px;'>Anomaly in AE signal detected!</span>"
ae_no_anomaly_detected = "<span style='color:green; font-size: 20px;'>No anomaly in AE signal detected.</span>"
ae_anomalous = False
current_anomaly_detected = "<span style='color:red; font-size: 20px;'>Anomaly in LRMS signal detected!</span>"
current_no_anomaly_detected = "<span style='color:green; font-size: 20px;'>No anomaly in LRMS signal detected.</span>"
current_anomalous = False

# Update the graph with the incoming data, as long as it keeps coming
while True:
    next_sample = data_provider.next()
    if next_sample is None:
        break

    # insert anomaly detection logic here, currently just an example for demonstration purposes
    if ae_anomalous or next_sample[0]["AEKi_rate2000000_clipping0_batch0"] > 0.04 or \
            next_sample[0]["AEKi_rate2000000_clipping0_batch0"] < -0.04:
        ae_anomaly_placeholder.markdown(ae_anomaly_detected, unsafe_allow_html=True)
        ae_anomalous = True
    else:
        ae_anomaly_placeholder.markdown(ae_no_anomaly_detected, unsafe_allow_html=True)
    if current_anomalous or next_sample[1] > 0.25 or next_sample[1] < -0.25:
        current_anomaly_placeholder.markdown(current_anomaly_detected, unsafe_allow_html=True)
        current_anomalous = True
    else:
        current_anomaly_placeholder.markdown(current_no_anomaly_detected, unsafe_allow_html=True)

    if ae_anomalous or current_anomalous:
        title_placeholder.title(f"Ongoing {process_type} Process Anomalous")

    ae_sample, current_sample = next_sample
    ae_graph.update_graph(ae_sample)
    lrms_grinding_graph.update_graph(current_sample)

# Update the title to indicate that the simulation is complete
if ae_anomalous or current_anomalous:
    title_placeholder.title(f"{process_type} Process finished with anomalies")
else:
    title_placeholder.title(f"{process_type} Process finished without anomalies")

from data_provider import DataProvider
import streamlit as st
from amplitude_graph import AmplitudeGraph

data_provider = DataProvider('./2024.02.14_22.00.40_Grinding', downsample_factor=2000)
meta = data_provider.meta_data()
process_type = meta['process']
title_placeholder = st.empty()  
title_placeholder.title(f"Ongoing {process_type} Process")  
st.write("AE Measurement")
ae_graph = AmplitudeGraph(st, update_batch_size=100)
st.write("LRMS of the Current Measurement")
lrms_grinding_graph = AmplitudeGraph(st, update_batch_size=100)

# Update the graph with the incoming data, as long as it keeps coming
while True:
    next_sample = data_provider.next()
    if next_sample is None:
        break
    ae_sample, current_sample = next_sample
    ae_graph.update_graph(ae_sample)
    lrms_grinding_graph.update_graph(current_sample)

# Update the title to indicate that the simulation is complete
title_placeholder.title(f"{process_type} Process finished")

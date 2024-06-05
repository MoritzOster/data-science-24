from data_provider import DataProvider
import streamlit as st
import pandas as pd

class AmplitudeGraph:
    def __init__(self, st_object,update_batch_size):
        self.update_batch_size = update_batch_size
        self.chart = st_object.line_chart()
        self.update_batch = []
        self.st_object = st_object

    def update_graph(self, next_measurement):
        self.update_batch.append(next_measurement)
        if len(self.update_batch) < self.update_batch_size:
            return True
        self.chart.add_rows(self.update_batch)
        self.update_batch = []
        return True

# left_column, right_column = st.columns(2)
left_column, right_column = st, st

# ae_file_path = './2024.02.14_22.00.40_Grinding'
ae_graph = AmplitudeGraph(left_column, update_batch_size= 100)

# lrms_grinding_path = './2024.02.14_22.00.40_Grinding/raw/Sampling2000KHz_AEKi-0.parquet'
lrms_grinding_graph = AmplitudeGraph(right_column, update_batch_size= 100)
# data_provider = DataProvider('./raw/Sampling2000KHz_AEKi-0.parquet', sampling_rate = 2e6, downsample_factor = 1000)

data_provider = DataProvider('./2024.02.14_22.00.40_Grinding', downsample_factor = 1000)
while (1):
    ae_sample, current_sample = data_provider.next()
    if (ae_sample is None) | (current_sample is None):
        break
    ae_graph.update_graph(ae_sample) 
    lrms_grinding_graph.update_graph(current_sample)
        
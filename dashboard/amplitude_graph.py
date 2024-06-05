from data_provider import DataProvider
import streamlit as st

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

        
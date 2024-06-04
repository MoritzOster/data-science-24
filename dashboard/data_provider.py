import pandas as pd
import time

class DataProvider:
    def __init__(self, file_path, sampling_rate, downsample_factor):
        self.data = pd.read_parquet(file_path)
        self.total_points = len(self.data)
        self.sampling_rate = sampling_rate
        # Downsample the data by only considering every downsample_factor element
        self.downsampled_data = self.data[::downsample_factor]
        self.downsampled_total_points = len(self.downsampled_data)
        self.current_index = 0
        # Downsample factor = 1 -> update for every sample taken (at the sampling rate)
        # e.g. 2 MHz = 2e6 samples per second -> update every 1 / 2e6 seconds
        self.sleep_time = downsample_factor / sampling_rate

    def next(self):
        if (self.current_index == self.downsampled_total_points):
            return
        current_measurement = self.downsampled_data.iloc[self.current_index]
        self.current_index += 1
        # Print how many percent of the data has been processed
        # print(f"{self.current_index / self.downsampled_total_points * 100:.2f}% processed")
        # Sleep to simulate real-time data
        time.sleep(self.sleep_time)
        # print (current_measurement)
        return current_measurement

# Usage example
# file_path = './2024.02.14_22.00.40_Grinding/raw/Sampling2000KHz_AEKi-0.parquet'
# sampling_rate = 2e6  # 2 MHz
# downsample_factor = 1000
# data_provider = DataProvider(file_path, sampling_rate, downsample_factor)

# while (1):
#     next_measurement = data_provider.next()
#     if next_measurement is None:
#         break
#     print(next_measurement)

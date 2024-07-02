import pandas as pd
import time
import json

class DataProvider:
    def __init__(self, file_path, downsample_factor):
        ae_data = pd.read_parquet(file_path + '/raw/Sampling2000KHz_AEKi-0.parquet')
        current_data = pd.read_parquet(file_path + '/raw/Sampling100KHz_Irms_Grinding-Grinding spindle current L1-Grinding spindle current L2-Grinding spindle current L3-0.parquet')

        # Assuming a sampling rate of 2 MHz for ae and 0.1 MHz for lrms
        ae_downsample_factor = downsample_factor
        current_downsample_factor = ae_downsample_factor // 20
        sampling_rate = 2e6

        # Downsample the data by only considering every downsample_factor element
        self.downsampled_ae_data = ae_data[::ae_downsample_factor]

        self.downsampled_total_points = len(self.downsampled_ae_data)
        self.index = 0
        # Downsample factor = 1 -> update for every sample taken (at the sampling rate)
        # e.g. 2 MHz = 2e6 samples per second -> update every 1 / 2e6 seconds
        # Works for both ae and current data, since current data is downsampled by a factor of 20 less
        self.sleep_time = downsample_factor / sampling_rate

        # Read the meta.json file
        with open(file_path + '/meta.json', 'r') as f:
            self.meta = json.load(f)

    def meta_data(self):
        return self.meta

    def next(self):
        if self.index >= self.downsampled_total_points:
            return None
        ae_sample = self.downsampled_ae_data.iloc[self.index]
        self.index += 1
        # Print how many percent of the data has been processed
        # print(f"{self.index / self.downsampled_total_points * 100:.2f}% processed")
        # Sleep to simulate real-time data
        time.sleep(self.sleep_time)
        return (ae_sample)

    def next_batch(self, batch_size):
        batch = []
        while len(batch) < batch_size:
            next_sample = self.next()
            if next_sample is None:
                break
            batch.append(next_sample)
        return batch


import pandas as pd
import json

class DataProvider:
    def __init__(self, file_path, cluster_factor, downsample_factor):

        self.ae_data = pd.read_parquet(file_path + '/raw/Sampling2000KHz_AEKi-0.parquet')['AEKi_rate2000000_clipping0_batch0'][::20][::downsample_factor]
        current_data = pd.read_parquet(file_path + '/raw/Sampling100KHz_Irms_Grinding-Grinding spindle current L1-Grinding spindle current L2-Grinding spindle current L3-0.parquet')#[::downsample_factor]
        self.current_lrms_data = current_data['Irms_Grinding_rate100000_clipping0_batch0'][::downsample_factor]
        self.grinding_l1_data = current_data['Grinding spindle current L1_rate100000_clipping0_batch0'][::downsample_factor]
        self.grinding_l2_data = current_data['Grinding spindle current L2_rate100000_clipping0_batch0'][::downsample_factor]
        self.grinding_l3_data = current_data['Grinding spindle current L3_rate100000_clipping0_batch0'][::downsample_factor]

        # For debugging purposes:
        self.image_path = file_path + '/raw_fig.png'

        # Assuming a sampling rate of 2 MHz for ae and 0.1 MHz for lrms
        ae_downsample_factor = downsample_factor
        current_downsample_factor = ae_downsample_factor // 20
        sampling_rate = 2e6
        self.index = 0

        # Downsample factor = 1 -> update for every sample taken (at the sampling rate)
        # e.g. 2 MHz = 2e6 samples per second -> update every 1 / 2e6 seconds
        # Works for both ae and current data, since current data is downsampled by a factor of 20 less
        self.cluster_factor = cluster_factor
        self.sleep_time = downsample_factor / sampling_rate

        self.file_path = file_path
        # Read the meta.json file
        with open(file_path + '/meta.json', 'r') as f:
            self.meta = json.load(f)

    def meta_data(self):
        return self.meta

    def get_current_path(self):
        return self.file_path
    
    def get_ae_path(self):
        return self.file_path + '/raw/Sampling2000KHz_AEKi-0.parquet'
    
    def next(self):
        start = self.index * self.cluster_factor
        self.index += 1
        end = self.index * self.cluster_factor

        min_size = min(self.ae_data.size, 
                       self.grinding_l1_data.size, 
                       self.grinding_l2_data.size,
                       self.grinding_l3_data.size)

        end = min(end, min_size - 1)

        ae_sample = self.ae_data.iloc[int(start):int(end)]

        current_sample = self.current_lrms_data.iloc[int(start):int(end)]
        grinding_sample_l1 = self.grinding_l1_data.iloc[int(start):int(end)]
        grinding_sample_l2 = self.grinding_l2_data.iloc[int(start):int(end)]
        grinding_sample_l3 = self.grinding_l3_data.iloc[int(start):int(end)]
        # time.sleep(self.sleep_time)

        return ae_sample, current_sample, grinding_sample_l1, grinding_sample_l2, grinding_sample_l3, list(range(start, end))

# Usage example
# file_path = '../data/example_recordings/'

# downsample_factor = 2000
# data_provider = DataProvider(file_path, downsample_factor)

# while True:
#     next_measurement = data_provider.next()
#     if next_measurement is None:
#         break
#     print(next_measurement)

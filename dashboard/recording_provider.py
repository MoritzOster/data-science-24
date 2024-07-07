import os
import numpy as np

from data_provider import DataProvider


class DataProviderProvider:
    def __init__(self, data_path, cluster_factor, downsample_factor):
        self.downsample_factor = downsample_factor
        self.cluster_factor = cluster_factor
        
        ok_data_path = data_path + '/OK_Measurements'
        nok_data_path = data_path + '/NOK_Measurements'

        self.ok_data_paths = []
        #Iterate through the files in the OK_Measurements folder and append the paths to the list
        for file in os.listdir(ok_data_path):
            path = os.path.join(ok_data_path, file)
            if os.path.isdir(path):
                self.ok_data_paths.append(path)
        self.nok_data_paths = []
        #Iterate through the files in the NOK_Measurements folder and append the paths to the list
        for file in os.listdir(nok_data_path):
            path = os.path.join(nok_data_path, file)
            if os.path.isdir(path):
                self.nok_data_paths.append(path)


    def get_ok_data_provider(self):
        # Pick a random path from the list of ok data paths
        random_index = np.random.randint(0, len(self.ok_data_paths))
        random_path = self.ok_data_paths[random_index]
        return DataProvider(random_path, self.cluster_factor, self.downsample_factor)
    
    def get_nok_data_provider(self):
        # Pick a random path from the list of nok data paths
        random_index = np.random.randint(0, len(self.nok_data_paths))
        random_path = self.nok_data_paths[random_index]
        return DataProvider(random_path, self.cluster_factor, self.downsample_factor)
    
    def get_any_data_provider(self):
        # Randomly pick between ok and nok data
        if np.random.rand() < 0.5:
            return self.get_ok_data_provider()
        else:
            return self.get_nok_data_provider()
        

# Usage example
# file_path = '../data/example_recordings'

# downsample_factor = 2000
# data_provider = DataProviderProvider(file_path, downsample_factor)



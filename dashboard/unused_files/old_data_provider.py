import pandas as pd
import time

class DataProvider:
    def __init__(self, file_path, batch_size, sleep_time, sampling_rate = 2000000):
        self.data = pd.read_parquet(file_path)
        self.batch_size = batch_size
        self.sleep_time = sleep_time
        self.total_points = len(self.data)
        self.current_index = 0
        self.sampling_rate = sampling_rate

    def get_next_batch(self):
        if self.current_index >= self.total_points:
            return None
        end_index = min(self.current_index + self.batch_size, self.total_points)
        batch = self.data.iloc[self.current_index:end_index]
        self.current_index = end_index
        # print(f"Processed {self.current_index/self.total_points*100:.2f}% of data")
        return batch

    def run(self):
        start_time = time.time()
        while self.current_index < self.total_points:
            batch = self.get_next_batch()
            if batch is not None:
                yield batch
            elapsed_time = time.time() - start_time
            expected_time = self.current_index / self.sampling_rate
            sleep_time = max(0, expected_time - elapsed_time)
            time.sleep(sleep_time)

# Example usage
if __name__ == "__main__":
    provider = DataProvider('path/to/datafile.parquet', batch_size=1000, sleep_time=0.1, sampling_rate=200000)
    for batch in provider.run():
        # Process the batch
        pass

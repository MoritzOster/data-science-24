import pyarrow.parquet as pq
import matplotlib.pyplot as plt
# Open the Parquet file
parquet_file = pq.ParquetFile('raw/Sampling100KHz_Irms_Grinding-Grinding spindle current L1-Grinding spindle current L2-Grinding spindle current L3-0.parquet')

# Read the Parquet file into a pandas DataFrame
df = parquet_file.read().to_pandas()

# Now you can work with the DataFrame 'df'

print(len(df))
print(df.iloc[0])

# data = df['AEKi_rate2000000_clipping0_batch0']

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df, marker='o', markersize=3, linestyle='')
plt.title('Sensor Data with Spikes')
plt.xlabel('Time')
plt.ylabel('Sensor Reading')
plt.grid(True)
plt.show()
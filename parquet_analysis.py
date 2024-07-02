import pandas as pd

# Load the Parquet file
file_path = 'features.parquet'
df = pd.read_parquet(file_path)

# Print the columns
print(df.columns)

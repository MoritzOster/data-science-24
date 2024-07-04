import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import re
# Define the path to the OK_measurement folder
folder_path = "NOK_Test"

# Initialize an empty list to store DataFrame for each top-level folder
dfs = []

# Iterate over each top-level folder in the OK_measurement folder
for top_folder in os.listdir(folder_path):
    print(top_folder)
    top_folder_path = os.path.join(folder_path, top_folder)
    print(top_folder_path)
    # Check if the item in the top_folder_path is a directory
    if os.path.isdir(top_folder_path):
        print("yes")
        # Initialize an empty list to store DataFrame for each raw folder
        raw_dfs = []
        
        # Iterate over each folder inside the top-level folder
        for folder in os.listdir(top_folder_path):
            folder_path_ = os.path.join(top_folder_path, folder)
            
            # Check if the item in the folder_path is a directory
            if os.path.isdir(folder_path_):
                
                # Define the path to the raw folder
                raw_folder_path = folder_path_
                print(folder_path_)
                # Check if the raw folder exists
                if os.path.exists(raw_folder_path):

                    parquet_files = [file for file in os.listdir(raw_folder_path) if file.endswith('.parquet')]
                    for file in parquet_files:
                        # Assuming your data has numerical columns
                        df = pd.read_parquet(os.path.join(raw_folder_path, file))
                        numerical_data = df.select_dtypes(include=[float, int])

                        # Standardize the data
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(numerical_data)

                        # Apply PCA
                        pca = PCA()
                        pca_data = pca.fit_transform(scaled_data)

                        # Explained variance ratio
                        explained_variance = pca.explained_variance_ratio_
                        cumulative_explained_variance = explained_variance.cumsum()

                        # Scree plot
                        # plt.figure(figsize=(10, 6))
                        # plt.plot(range(1, len(explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='--')
                        # plt.xlabel('Number of Components')
                        # plt.ylabel('Cumulative Explained Variance')
                        # plt.title('Scree Plot')
                        # plt.grid()
                        # plt.show()

                        # Choose the number of components that explain at least 90-95% variance
                        n_components = next(i for i, cumulative_variance in enumerate(cumulative_explained_variance) if cumulative_variance >= 0.9) + 1
                        print(f"Number of components to retain: {n_components}")
                        date_time_str = re.search(r'(\d{4}.\d{2}.\d{2})_(\d{2}.\d{2}.\d{2})', top_folder)
                        datetime_format = ""
                        if date_time_str:
                            date_str = date_time_str.group(1).replace('.', '-')
                            time_str = date_time_str.group(2).replace('.', ':')
                            datetime_str = f"{date_str} {time_str}"
                            datetime_format = pd.to_datetime(datetime_str, format='%Y-%m-%d %H:%M:%S')
                        # Apply PCA with the chosen number of components
                        if n_components != 1:
                            pca = PCA(n_components=n_components)
                            reduced_data = pca.fit_transform(scaled_data)

                            # Create a DataFrame from the reduced data
                            reduced_df = pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(n_components)])
                            fin_df = df.sample(n=5000, ignore_index=True)
                            
                            fin_df['Time'] = datetime_format
                            fin_df.to_csv(f'nok_pca_analysis_v1/{top_folder}_{file}.csv', index=False)
                        else:
                            fin_df = df.sample(n=5000, ignore_index=True)
                            fin_df['Time'] = datetime_format
                            fin_df.to_csv(f'nok_pca_analysis_v1/{top_folder}_{file}.csv', index=False)



                    

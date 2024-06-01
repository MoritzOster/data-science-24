import pandas as pd
from sklearn.preprocessing import StandardScaler

def get_data(data_path):
    data = pd.read_parquet(data_path)

    ok_data = data.loc[data['anomaly'] == False]
    ok_data = ok_data.drop('anomaly', axis=1)
    ok_eval_data = ok_data.iloc[:5]
    ok_data = ok_data.drop(range(5))
    
    nok_data = data.loc[data['anomaly'] == True]
    nok_data = nok_data.drop('anomaly', axis=1)

    # Ensure the data is of type float32
    ok_data = ok_data.astype('float32')
    ok_eval_data = ok_eval_data.astype('float32')
    nok_data = nok_data.astype('float32')
    
    # Convert the DataFrame to a NumPy array
    ok_array = ok_data.values
    nok_array = nok_data.values
    ok_eval_array = ok_eval_data.values

    scaler = StandardScaler()
    ok_array = scaler.fit_transform(ok_array)
    nok_array = scaler.transform(nok_array)
    ok_eval_array = scaler.transform(ok_eval_array)
    
    return ok_array, nok_array, ok_eval_array
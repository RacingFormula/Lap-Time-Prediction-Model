import pandas as pd

def preprocess_data(input_data):
    if isinstance(input_data, str):
        data = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        data = input_data
    else:
        raise ValueError("Input must be a file path or a pandas DataFrame.")

    features = data.drop(columns=['LapTime'])
    labels = data['LapTime']
    return features, labels

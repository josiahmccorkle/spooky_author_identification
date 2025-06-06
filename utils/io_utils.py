import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# --------------------------------
#  Load Data
# --------------------------------
output_dir = "./outputs"

def load_data(isTest):
    filePath = './data/'
    files = [f for f in os.listdir(filePath) if f.endswith('.csv')]
    print(files)
    expected_files = {"train.csv", "test.csv"}
    dfs = {file: pd.read_csv(os.path.join(filePath, file)) for file in files if file in expected_files}
    try:
        df = dfs["test.csv"] if isTest else dfs["train.csv"]
        print(df.head())
        return df

    except KeyError as e:
        raise FileNotFoundError(f"Missing file: {e.args[0]} in './data/'")


import pandas as pd
from config.config_path import PATHS

def preprocessing(df):
    df = df.drop(['instant','dteday','atemp','casual','registered'],axis=1)
    return df

if __name__ == "__main__":
    df = pd.read_csv(PATHS['raw_data'],delimiter=',')
    df = preprocessing(df)
    df.to_parquet(PATHS['preprocessed_data'])

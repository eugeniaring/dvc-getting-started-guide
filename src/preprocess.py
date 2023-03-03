import pandas as pd
from sklearn.model_selection  import train_test_split
from config.config_path import PATHS,cb_features

def preprocessing(df):
    df = df.drop(['instant','dteday','atemp','casual','registered'],axis=1)
    return df

def split(df):
    X,y = df[cb_features['feature_names']],df[cb_features['target'][0]]
    return train_test_split(X,y,test_size=0.2,random_state=123) 

def change_dtype(df):
    for c in ['season','yr','mnth','hr','holiday','weekday','workingday','weathersit']:
        df[c] = df[c].astype('object')

if __name__ == "__main__":
    df = pd.read_csv(PATHS['raw_data'],delimiter=',')
    df = preprocessing(df)
    df.to_parquet(PATHS['preprocessed_data'])

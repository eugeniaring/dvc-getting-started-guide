import pandas as pd
from dvc.api import params_show
from sklearn.model_selection import train_test_split


def preprocessing(df):
    df = df.drop(['instant','dteday','atemp','casual','registered'],axis=1)
    return df

def split(df,test_size):
    X,y = df[params_show()['cb_features']['feature_names']],df[params_show()['cb_features']['target']]
    return train_test_split(X,y,test_size=test_size,random_state=123) 

def change_dtype(df):
    for c in ['season','yr','mnth','hr','holiday','weekday','workingday','weathersit']:
        df[c] = df[c].astype('object')

if __name__ == "__main__":
    df = pd.read_csv(params_show()['PATHS']['raw_data'],delimiter=',')
    df = preprocessing(df)
    df.to_parquet(params_show()['PATHS']['preprocessed_data'])

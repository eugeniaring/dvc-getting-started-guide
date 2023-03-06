import pandas as pd
from dvc.api import params_show
from sklearn.model_selection import train_test_split


def preprocessing(df):
    df = df.drop(['instant','dteday','atemp','casual','registered'],axis=1)
    return df

def split(df,test_size):
    X = df[params_show()['cb_features']['feature_names']]
    idx_train, idx_test = train_test_split(X.index,test_size=test_size,random_state=123)
    return idx_train, idx_test

def change_dtype(df):
    for c in ['season','yr','mnth','hr','holiday','weekday','workingday','weathersit']:
        df[c] = df[c].astype('object')

if __name__ == "__main__":
    PATHS = params_show()['PATHS']
    df = pd.read_csv(PATHS['raw_data'],delimiter=',')
    df = preprocessing(df)
    df.to_csv(PATHS['preprocessed_data'],index=False)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from config.config_path import PATHS,cb_features
from preprocess import change_dtype
from catboost import CatBoostRegressor
import pickle
from preprocess import split
from dvc.api import params_show

def split(df,test_size):
    X,y = df[cb_features['feature_names']],df[cb_features['target'][0]]
    return train_test_split(X,y,test_size=test_size,random_state=123) 

def change_dtype(df):
    for c in ['season','yr','mnth','hr','holiday','weekday','workingday','weathersit']:
        df[c] = df[c].astype('object')

if __name__ == "__main__":
    df = pd.read_parquet(PATHS['preprocessed_data'])
    change_dtype(df)
    print(df.info())
    #print(list(df.columns))
    X_train, X_test, y_train, y_test = split(df,test_size=params_show()['split'])
    categorical_indices = np.where(df.dtypes=='object')[0]
    categorical_indices = categorical_indices.tolist()
    print(categorical_indices)
    cb = CatBoostRegressor(n_estimators=200,
                       loss_function='RMSE',
                       learning_rate=0.1,
                       depth=8, task_type='CPU',
                       random_state=1,
                       verbose=False)
    cb.fit(X_train, y_train,cat_features=categorical_indices)
    pickle.dump(cb,open("model/catboost.pickle",'wb'))

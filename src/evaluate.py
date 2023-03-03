import pandas as pd
from config.config_path import PATHS,cb_features
from catboost import CatBoostRegressor
import pickle
from preprocess import split
from sklearn.metrics import mean_absolute_error

if __name__ == "__main__":
    df = pd.read_parquet(PATHS['preprocessed_data'])
    X_train, X_test, y_train, y_test = split(df)
    cb = pickle.load(open('model/catboost.pickle', 'rb'))
    y_pred_train = cb.predict(X_train)
    y_pred = cb.predict(X_test)
    print('train MAE: ',mean_absolute_error(y_train,y_pred_train))
    print('test MAE: ',mean_absolute_error(y_test,y_pred))
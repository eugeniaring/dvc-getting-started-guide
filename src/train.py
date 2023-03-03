import pandas as pd
from sklearn.model_selection import train_test_split
from config.config_path import PATHS,cb_features
from preprocess import change_dtype
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

if __name__ == "__main__":
    df = pd.read_parquet(PATHS['preprocessed_data'])
    change_dtype(df)
    print(df.info())
    print(list(df.columns))
    X,y = df[cb_features['feature_names']],df[cb_features['target'][0]]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=123)

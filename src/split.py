import pandas as pd
from dvc.api import params_show
from preprocess import change_dtype, split
import pickle

def save_txt(file,path):
    with open(path, "wb") as fp:
        pickle.dump(file, fp)


if __name__ == "__main__":
    PATHS = params_show()['PATHS']
    df = pd.read_parquet(PATHS['preprocessed_data'])
    change_dtype(df)
    print(df.info())
    X_train, X_test, y_train, y_test = split(df,test_size=params_show()['split']['ratio'])
    save_txt(X_train,PATHS['features']['train'])
    save_txt(X_test,PATHS['features']['test'])
    save_txt(y_train,PATHS['target']['train'])
    save_txt(y_test,PATHS['target']['test'])
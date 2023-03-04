import pandas as pd
from config.config_path import PATHS,cb_features
import pickle
from train import split
from sklearn.metrics import mean_absolute_error
from dvc.api import params_show
import json

if __name__ == "__main__":
    df = pd.read_parquet(PATHS['preprocessed_data'])
    X_train, X_test, y_train, y_test = split(df,test_size=params_show()['split']['ratio'])
    cb = pickle.load(open('model/catboost.pickle', 'rb'))
    y_pred_train = cb.predict(X_train)
    y_pred = cb.predict(X_test)
    print('train MAE: ',mean_absolute_error(y_train,y_pred_train))
    print('test MAE: ',mean_absolute_error(y_test,y_pred))
    diz_eval = {'train_mae':mean_absolute_error(y_train,y_pred_train),
                'test_mae':mean_absolute_error(y_test,y_pred)}
    with open('evaluation/metrics.json', "w") as fd:
        json.dump(diz_eval,fd,indent=4,)
    
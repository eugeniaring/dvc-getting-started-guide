import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
import pickle
from dvc.api import params_show
import json

def read_file(path):

	with open(path, 'rb') as fp:
		f = pickle.load(fp)

	return f

def train_catboost(train_data):
    X_train,y_train = train_data 
    categorical_indices = np.where(X_train.dtypes=='object')[0]
    categorical_indices = categorical_indices.tolist()
    cb = CatBoostRegressor(n_estimators=200,
                        loss_function='RMSE',
                        learning_rate=0.1,
                        depth=8, task_type='CPU',
                        random_state=1,
                        verbose=False)
    cb.fit(X_train, y_train,cat_features=categorical_indices)
    pickle.dump(cb,open("model/catboost.pickle",'wb'))
    return cb

def eval_catboost(train_data,test_data,cb):
    X_train, y_train = train_data
    X_test, y_test = test_data
    y_pred_train = cb.predict(X_train)
    y_pred = cb.predict(X_test)

    print('train MAE: ',mean_absolute_error(y_train,y_pred_train))
    print('test MAE: ',mean_absolute_error(y_test,y_pred))

    diz_eval = {'train_mae':mean_absolute_error(y_train,y_pred_train),
                'test_mae':mean_absolute_error(y_test,y_pred)}
    with open('evaluation/metrics.json', "w") as fd:
        json.dump(diz_eval,fd,indent=4,)
      

if __name__ == "__main__":
    PATHS = params_show()['PATHS']
    train_df = pd.read_csv(PATHS['train'])
    test_df = pd.read_csv(PATHS['test'])
    X_train, y_train = train_df[params_show()['cb_features']['feature_names']],train_df[params_show()['cb_features']['target']]
    X_test, y_test = test_df[params_show()['cb_features']['feature_names']],test_df[params_show()['cb_features']['target']]
    cb = train_catboost((X_train,y_train))
    eval_catboost((X_train, y_train), (X_test, y_test),cb)


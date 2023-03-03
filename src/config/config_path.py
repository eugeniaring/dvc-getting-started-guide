import os

PATHS = {'raw_data': os.path.join('data','raw','hour.csv'),
         'preprocessed_data': os.path.join('data','preprocessed','hour.parquet'),
         }


cb_features = {'target':['cnt'],
            'feature_names':['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'hum', 'windspeed']
            }

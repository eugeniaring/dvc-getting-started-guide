import pandas as pd
from dvc.api import params_show
from preprocess import change_dtype, split


if __name__ == "__main__":
    PATHS = params_show()['PATHS']
    df = pd.read_csv(PATHS['preprocessed_data'],delimiter=',')
    change_dtype(df)
    print(df.info())
    idx_train, idx_test = split(df,test_size=params_show()['split']['ratio'])
    train_df = df.iloc[idx_train]
    test_df = df.iloc[idx_test]
    train_df.to_csv(PATHS['train'],index=False)
    test_df.to_csv(PATHS['test'],index=False)

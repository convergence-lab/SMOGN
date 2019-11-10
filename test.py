import numpy as np
import pandas as pd
import warnings
import re
from smogn import SMOGN
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from glob import glob


def test():
    data_sets = glob("Imbalanced-Regression-DataSets/CSV_data/*.csv")

    sum_improvement = 0
    n_data = 0
    for dataset in data_sets:
        if re.match(".*a[1-7].csv$", dataset) != None:
            continue
        n_data += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            print("=====================")
            print("Dataset: " + dataset)
            df = pd.read_csv(dataset)
            str_columns1 = []
            str_columns2 = []
            for i, col in enumerate(df):
                if df[col].dtype == "object":
                    str_columns1 += [col]
                    str_columns2 += [i]
            target_col = df.columns[0]
            df2 = pd.get_dummies(df, columns=str_columns1)

            X = df2.drop(target_col, axis=1).values
            y = df2[target_col].values
            scaler = RobustScaler()
            X2 = scaler.fit_transform(X)

            np.random.seed(0)
            train_X, test_X = X[:int(len(X) * 0.8)], X[int(len(X) * 0.8):]
            train_y, test_y = y[:int(len(y) * 0.8)], y[int(len(y) * 0.8):]

            cls0 = RandomForestRegressor(n_estimators=10)
            cls0.fit(train_X, train_y)
            pred0 = cls0.predict(test_X)
            mse0 = mean_squared_error(test_y, pred0)
            print(f"RandomForrest \t\tMSE: {mse0:.3}")

            np.random.seed(0)
            train_df, test_df = df.iloc[:int(len(X) * 0.8)], df[int(len(X) * 0.8):]

            sm = SMOGN()
            new_df = sm.fit_transform(train_df, target_col)

            df3 = pd.concat([new_df, test_df])
            df3 = pd.get_dummies(df3, columns=str_columns1)

            X = df3.drop(target_col, axis=1).values
            y = df3[target_col].values

            train_X, test_X = X[:int(len(X) * 0.8)], X[int(len(X) * 0.8):]
            train_y, test_y = y[:int(len(y) * 0.8)], y[int(len(y) * 0.8):]

            scaler = RobustScaler()
            train_X = scaler.fit_transform(train_X)

            cls1 = RandomForestRegressor(n_estimators=10)
            cls1.fit(train_X, train_y)
            pred1 = cls1.predict(scaler.transform(test_X))
            mse1 = mean_squared_error(test_y, pred1)
            print(f"SMOGN RandomForrest \tMSE: {mse1:.3}")
            rel_improve = (mse0 - mse1) / ((mse0 + mse1) / 2.) * 100.
            print(f"Relative Improvement \t{rel_improve:.3} %")
            sum_improvement += rel_improve
    print()
    print(f"Avg Improvemnet {sum_improvement/n_data:.3} %")


if __name__ == "__main__":
    test()

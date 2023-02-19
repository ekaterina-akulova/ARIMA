import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def std_mean(data, col):
    std = data[col].std()
    mean = data[col].mean()
    return std, mean


def data_resample(data):
    data = pd.DataFrame(data.cnt.resample("5MIN", base=1).sum())
    return data

def data_split(data):
    train_set, test_set = np.split(data, [int(.67 * len(data))])
    return train_set, test_set


def standardising_data(train, test):
    scaler = StandardScaler()
    scaler.fit(train['cnt'].values.reshape(-1, 1))
    train['stand_value'] = scaler.transform(train['cnt'].values.reshape(-1, 1))
    test['stand_value'] = scaler.transform(test['cnt'].values.reshape(-1, 1))
    train = train.drop(columns=['cnt'], axis=1)
    test = test.drop(columns=['cnt'], axis=1)
    train = train.rename(columns={'stand_value': 'cnt'})
    test = test.rename(columns={'stand_value': 'cnt'})
    return train, test

def standardising_res(train, test):
    scaler = StandardScaler()
    scaler.fit(train['cnt'].values.reshape(-1, 1))
    test['stand_value'] = scaler.transform(test['cnt'].values.reshape(-1, 1))
    test['stand_predict'] = scaler.transform(test['predict'].values.reshape(-1, 1))
    test = test.drop(columns=['cnt'], axis=1)
    test = test.rename(columns={'stand_value': 'cnt'})
    test = test.drop(columns=['predict'], axis=1)
    test = test.rename(columns={'stand_predict': 'predict'})
    return test

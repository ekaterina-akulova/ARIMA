import pandas as pd
import numpy as np


def predict(coef, history):
    yhat = 0.0
    for i in range(1, len(coef)+1):
        yhat += coef[i-1] * history[-i]
    return yhat


def difference(dataset):
    diff = list()
    for i in range(1, len(dataset)):
        value = dataset[i] - dataset[i - 1]
        diff.append(value)
    return np.array(diff)


def get_predictions_arima(model, train_set, test_set):
    predictions = list()
    train, test = train_set.values, test_set.values
    history = [x for x in train]
    for t in range(len(test)):
        ar_coef, ma_coef = model.arparams, model.maparams
        resid = model.resid
        diff = difference(history)
        yhat = history[-1] + predict(ar_coef, diff) + predict(ma_coef, resid)
        obs = test_set.values[t]
        predictions.append(yhat)
        history.append(obs)
    prediction = pd.DataFrame(predictions)
    prediction.index = test_set.index
    test_set['predict'] = prediction
    return test_set


def get_prediction_auto_arima(model, train_sqrt, test_sqrt, test):
    history = [x for x in train_sqrt.values]
    predictions = list()
    predict_sqrt = list()
    for t in range(len(test)):
        model.fit(history)
        output = model.predict(n_periods=1)
        predict_sqrt.append(output[0])
        yhat = output[0] ** 2
        predictions.append(yhat)
        obs = test_sqrt.values[t]
        history.append(obs)
    prediction = pd.DataFrame(predictions)
    prediction.index = test.index
    test['predict'] = prediction
    return test


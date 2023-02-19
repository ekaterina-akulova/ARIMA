from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from training.steps.test_stationarity import isStationary
import sys


def model_auto_arima(train_set):
    if isStationary(train_set):
        model = auto_arima(train_set, start_p=1, start_q=1, max_p=3, max_q=3, m=7,
                           start_P=0, seasonal=True, d=1, D=1, trace=True,
                           error_action='ignore', suppress_warnings=True,
                           stepwise=True)
        return model
    else:
        sys.exit("Data is not stationary.")


def model_arima(train, p, d, q):
    if isStationary(train):
        model = ARIMA(train.values, order=(p, d, q))
        model_fit = model.fit()
        return model_fit
    else:
        sys.exit("Data is not stationary.")

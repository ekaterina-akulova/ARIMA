from statsmodels.tsa.stattools import adfuller


def isStationary(data, maxval=0.05):
    test = adfuller(data, autolag='AIC')
    p = test[1]
    if p <= maxval:
        return 1
    else:
        return 0



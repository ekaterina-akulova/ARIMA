from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def validation(pred_test):
    rmse = mean_squared_error(pred_test['cnt'], pred_test['predict'], squared=False)
    mse = mean_squared_error(pred_test['cnt'], pred_test['predict'])
    mae = mean_absolute_error(pred_test['cnt'], pred_test['predict'])
    return rmse, mse, mae
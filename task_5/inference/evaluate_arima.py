from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from training.steps.prepare_data import standardising_data, data_split, data_resample
from training.steps.load_data import load_data


def arima_params_evaluate():
    file = "data/tsData.json"
    data = load_data(file)
    data = data_resample(data)
    best = get_eval(data)
    print(best)


def evaluate_arima_model(X, arima_order):
    train, test = data_split(X)
    train, test = standardising_data(train, test)
    train, test = train.values, test.values
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    error = mean_squared_error(test, predictions)
    return error


def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    check_list = []
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    check_list.append(f'ARIMA%s : {order}, MSE=%.3f: {mse}')
                    print('ARIMA%s MSE=%.3f' % (order, mse))
                except:
                    continue
    with open('data/evaluate_arima.txt', 'w') as f:
        f.writelines(f'{row}\n' for row in check_list)
    return best_score, best_cfg


def get_eval(df):
    p_values = [0, 1, 2, 4]
    d_values = range(0, 3)
    q_values = range(0, 3)
    best = evaluate_models(df, p_values, d_values, q_values)
    return best




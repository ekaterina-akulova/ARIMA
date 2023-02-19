import pickle
import warnings

from detect_anomalies.detect import detect_std, detect
from inference.validate_model import validation
from training.steps.load_data import load_data
from training.steps.model_create import model_auto_arima, model_arima
from training.steps.predict import get_prediction_auto_arima, get_predictions_arima
from training.steps.prepare_data import data_split, data_resample, std_mean, standardising_res
from training.steps.prepare_data import standardising_data
from inference.evaluate_arima import arima_params_evaluate

import numpy as np

warnings.filterwarnings("ignore")


def save_model(model, output_model_path):
    with open(output_model_path, 'wb') as f:
        pickle.dump(model, f)


def get_params_eval():
    arima_params_evaluate()


def pipeline_arima(use_new_model=False, input_model_path='', output_model_path='data/model.pkl'):
    file = "data/tsData.json"
    data = load_data(file)
    data = data_resample(data)
    train, test = data_split(data)
    if use_new_model:
        model = model_arima(train, 1, 1, 1)
        save_model(model, output_model_path)
    else:
        model_pickle = open(input_model_path, "rb")
        model = pickle.load(model_pickle)
    prediction_test = get_predictions_arima(model, train, test)
    res, pred_test = detect(prediction_test, test)
    res.to_csv('data/result.csv')
    pred_test = standardising_res(train, pred_test)
    rmse, mse, mae = validation(pred_test)
    percent_anomaly = len(res) / len(test) * 100
    print("Percent anomaly = ", percent_anomaly)
    print("RMSE = ", rmse, "\nMSE = ", mse, "\nMAE = ", mae)


def pipeline_arima_stand(use_new_model=False, input_model_path='', output_model_path='data/model_arima.pkl'):
    file = "data/tsData.json"
    data = load_data(file)
    data = data_resample(data)
    train, test = data_split(data)
    std, mean = std_mean(train, 'cnt')
    train, test = standardising_data(train, test)
    if use_new_model:
        model = model_arima(train, 2, 1, 2)
        save_model(model, output_model_path)
    else:
        model_pickle = open(input_model_path, "rb")
        model = pickle.load(model_pickle)
    prediction_test = get_predictions_arima(model, train, test)
    res, pred_test = detect_std(prediction_test, test, std, mean)
    res.to_csv('data/result_arima.csv')
    rmse, mse, mae = validation(pred_test)
    percent_anomaly = len(res) / len(test) * 100
    print("Percent anomaly = ", percent_anomaly)
    print("RMSE = ", rmse, "\nMSE = ", mse, "\nMAE = ", mae)


def pipeline_auto_arima(use_new_model=False, input_model_path='', output_model_path='data/model_autoarima.pkl'):
    file = "data/tsData.json"
    data = load_data(file)
    data = data_resample(data)
    train, test = data_split(data)
    train_sqrt, test_sqrt = np.sqrt(train), np.sqrt(test)
    if use_new_model:
        model = model_auto_arima(train_sqrt)
        save_model(model, output_model_path)
    else:
        model_pickle = open(input_model_path, "rb")
        model = pickle.load(model_pickle)
    prediction_test = get_prediction_auto_arima(model, train_sqrt, test_sqrt, test)
    res, pred_test = detect(prediction_test, test)
    res.to_csv('data/result_auto_arima.csv')
    pred_test = standardising_res(train, pred_test)
    rmse, mse, mae = validation(pred_test)
    print(rmse, mse, mae)
    percent_anomaly = len(res) / len(test) * 100
    print("Percent anomaly = ", percent_anomaly)
    print("RMSE = ", rmse, "\nMSE = ", mse, "\nMAE = ", mae)


pipeline_auto_arima(input_model_path='data/model_autoarima.pkl')
# pipeline_arima(input_model_path='data/model.pkl')
#pipeline_arima_stand(input_model_path='data/model_arima.pkl')
#get_params_eval()

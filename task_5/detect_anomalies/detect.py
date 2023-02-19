import numpy as np

from inference.visualize import pred_anomaly_on_test_data


def calculate_prediction_errors(input_data):
    return (abs(input_data['cnt'] - input_data['predict'])).to_numpy()


def detect_anomalies(pred_error_threshold, df):
    test_reconstruction_errors = calculate_prediction_errors(df)
    predicted_anomalies = list(
        map(lambda v: 1 if v > pred_error_threshold else 0, test_reconstruction_errors))
    df['anomaly_predicted'] = predicted_anomalies
    indexes = [i for i, x in enumerate(predicted_anomalies) if x == 1]
    return indexes


def detect_std(prediction_test, test_set, std, mean):
    anomaly_config = 3
    test_pred_errors = calculate_prediction_errors(prediction_test)
    pred_error_threshold = np.mean(test_pred_errors) + anomaly_config * np.std(test_pred_errors)
    test_anomalies_idxs = detect_anomalies(pred_error_threshold, test_set)

    pred_anomaly_on_test_data(prediction_test, test_anomalies_idxs)

    detected_anomalies = test_set['cnt'].iloc[test_anomalies_idxs]
    orig_det_anom = (detected_anomalies * std) + mean
    return orig_det_anom, prediction_test


def detect(prediction_test, test_set):
    anomaly_config = 3
    test_pred_errors = calculate_prediction_errors(prediction_test)
    pred_error_threshold = np.mean(test_pred_errors) + anomaly_config * np.std(test_pred_errors)
    test_anomalies_idxs = detect_anomalies(pred_error_threshold, test_set)

    pred_anomaly_on_test_data(prediction_test, test_anomalies_idxs)

    detected_anomalies = test_set['cnt'].iloc[test_anomalies_idxs]
    return detected_anomalies, prediction_test

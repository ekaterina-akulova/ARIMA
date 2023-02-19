from itertools import cycle
import plotly.graph_objects as go
import numpy as np



def pred_anomaly_on_train_data(train_set, train_anomalies_idxs):
    layout = dict(xaxis=dict(title='Time'), yaxis=dict(title='Count'))

    fig = go.Figure(layout=layout)

    fig.add_trace(go.Scatter(x=train_set.index, y=train_set['stand_value'],
                             mode='markers',
                             marker=dict(color='blue')))

    fig.add_trace(go.Scatter(x=train_set.index, y=train_set['predict'],
                             mode='markers',
                             marker=dict(color='orange')))

    fig.add_trace(go.Scatter(x=train_set.index, y=train_set['stand_value'].iloc[train_anomalies_idxs],
                             mode='markers',
                             marker=dict(color='red')))

    nam = cycle(['Actual', 'Prediction', 'Detected Anomaly'])
    fig.for_each_trace(lambda t: t.update(name=next(nam)))
    fig.show()


def pred_anomaly_on_test_data(test_set, test_anomalies_idxs):
    ano_ind = np.where(test_set["anomaly_predicted"] == 1)
    layout = dict(xaxis=dict(title='Time'), yaxis=dict(title='Count'))

    fig = go.Figure(layout=layout)

    fig.add_trace(go.Scatter(x=test_set.index, y=test_set['cnt'],
                             mode='markers',
                             marker=dict(color='blue')))

    fig.add_trace(go.Scatter(x=test_set.index, y=test_set['predict'],
                             mode='markers',
                             marker=dict(color='orange')))

    fig.add_trace(go.Scatter(x=test_set.iloc[ano_ind].index, y=test_set['cnt'].iloc[test_anomalies_idxs],
                             mode='markers',
                             marker=dict(color='red')))

    nam = cycle(['Actual', 'Prediction', 'Detected Anomaly'])
    fig.for_each_trace(lambda t: t.update(name=next(nam)))
    fig.show()

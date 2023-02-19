import pandas as pd


def load_data(path):
    dataset = pd.read_json(path)
    return dataset

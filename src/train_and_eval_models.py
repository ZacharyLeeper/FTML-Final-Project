import numpy as np

from data import data_preprocess, split_data, DATA_PATH, noisy_data
from models import MODELS, save_models
from train_2 import train_models
from viz import model_results, graph_dp, graph_eo, graph_pred

if __name__ == '__main__':
    data = data_preprocess(DATA_PATH)
    data = noisy_data(data)
    train, test = split_data(data)

    models, thresholds = train_models(*train)
    results = model_results(models, thresholds, *test)

    graph_dp(results, MODELS)
    graph_eo(results, MODELS)
    graph_pred(results, thresholds, MODELS)

    filenames = [f'../models/{name}' for name in MODELS]
    save_models(models, thresholds, filenames)

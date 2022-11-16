import numpy as np

from data import data_preprocess, split_data, DATA_PATH
from models import MODELS
from train_test import train_models, eval_models
from viz import graph, print_model_results

if __name__ == '__main__':
    data = data_preprocess(DATA_PATH)
    train, test = split_data(data)

    models, thresholds = train_models(*train)
    print_model_results(models, thresholds, *test)
    # accuracy = eval_models(models, thresholds, *test)
    # graph(np.arange(4), accuracy, 'Model', 'Accuracy', MODELS, 'test', 'accuracy')
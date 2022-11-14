import numpy as np

from ftml.data import data_preprocess, split_data, DATA_PATH
from ftml.models import MODELS
from ftml.train_test import train_models, eval_models
from ftml.viz import graph

if __name__ == '__main__':
    data = data_preprocess(DATA_PATH)
    train, test = split_data(data)

    models = train_models(*train)
    accuracy = eval_models(models, *test)
    graph(np.arange(4), accuracy, 'Model', 'Accuracy', MODELS, 'test', 'accuracy')
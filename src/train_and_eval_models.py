import matplotlib.pyplot as plt 

from data import data_preprocess, split_data, DATA_PATH, noisy_data
from models import MODELS, save_models
from train_test import train_models
from viz import model_results, graph_dp, graph_eo

# Script for training neural net with all fairness constraints

if __name__ == '__main__':
    data = data_preprocess(DATA_PATH)
    data = noisy_data(data)
    train, test = split_data(data)

    models, thresholds = train_models(*train)
    results = model_results(models, thresholds, *test)

    for i in range(len(models)):
        fig, ax = plt.subplots(nrows=1, ncols=2, facecolor=(.9, .9, .9), edgecolor=(.9, .9, .9), label='Loan Applications Accepted')
        graph_dp(ax[0], results[i])
        graph_eo(ax[1], results[i])
        fig.tight_layout()
        plt.savefig(f'../results/{MODELS[i]}.png')

    filenames = [f'../models/{name}' for name in MODELS]
    save_models(models, thresholds, filenames)

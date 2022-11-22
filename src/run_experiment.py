from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np

from data import data_preprocess, split_data, DATA_PATH, noisy_data
from models import MODELS, load_model
from train_2 import train_models
from viz import model_results, graph_dp, graph_eo, make_sliders

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str)
    args = parser.parse_args()

    path = args.model
    model, threshold = load_model(path)
    print(model, threshold)
    
    data = data_preprocess(DATA_PATH)
    data = noisy_data(data)
    _, test = split_data(data)
    results = model_results([model], [threshold], *test)[0]

    pred_min = np.min(results['pred'])
    pred_max = np.max(results['pred'])
    scaled_threshold = tuple([1-(threshold[i]-pred_min) / (pred_max-pred_min) for i in range(4)])

    fig, ax = plt.subplots(nrows=1, ncols=2, facecolor=(.9, .9, .9), edgecolor=(.9, .9, .9), label='Loan Applications Accepted')
    fig.subplots_adjust(bottom=0.50)
    m_slider, f_slider = make_sliders(fig, scaled_threshold)

    def update_slider(_):
        threshold = (((1-m_slider.val)*(pred_max-pred_min))+pred_min,
                     ((1-m_slider.val)*(pred_max-pred_min))+pred_min,
                     ((1-f_slider.val)*(pred_max-pred_min))+pred_min,
                     ((1-f_slider.val)*(pred_max-pred_min))+pred_min)
        results = model_results([model], [threshold], *test)
        ax[0].clear()
        ax[1].clear()
        graph_dp(ax[0], results[0])
        graph_eo(ax[1], results[0])

    m_slider.on_changed(update_slider)
    f_slider.on_changed(update_slider)

    graph_dp(ax[0], results)
    graph_eo(ax[1], results)

    fig.tight_layout()
    plt.show()
    plt.savefig('../results/test.png')

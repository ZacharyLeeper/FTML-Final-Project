from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np

from data import data_preprocess, split_data, DATA_PATH, noisy_data
from models import MODELS, load_model
from train_2 import train_models
from viz import model_results, graph_dp, graph_eo, make_sliders

def scale_threshold(val, p_min, p_max):
    return (val - p_min) / (p_max - p_min)

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

    fig, ax = plt.subplots(nrows=1, ncols=2, facecolor=(.9, .9, .9), edgecolor=(.9, .9, .9), label='Loan Applications Accepted')
    fig.subplots_adjust(bottom=0.50)
    m_slider, f_slider = make_sliders(fig, threshold)

    def update_slider(val):
        new_thresholds = (m_slider.val, m_slider.val, f_slider.val, f_slider.val)
        results = model_results([model], [new_thresholds], *test)
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
    # plt.pause(10000)
    plt.savefig('./results/test.png')

from argparse import ArgumentParser
import matplotlib.pyplot as plt

from data import data_preprocess, split_data, DATA_PATH, noisy_data
from models import MODELS, load_model
from train_2 import train_models
from viz import model_results, graph_dp, graph_eo, make_sliders, update_slider


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
    results = model_results([model], [threshold], *test)

    plt.ion()
    fig, ax = plt.subplots(nrows=1, ncols=2, facecolor=(.9, .9, .9), edgecolor=(.9, .9, .9), label='Loan Applications Accepted')
    m_slider, f_slider = make_sliders(fig, threshold)
    m_slider.on_changed(update_slider)
    f_slider.on_changed(update_slider)
    fig.subplots_adjust(bottom=0.25)
    graph_dp(ax[0], results[0])
    graph_eo(ax[1], results[0])

    fig.tight_layout()
    plt.pause(20)
    plt.savefig('../results/test.png')

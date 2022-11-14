import matplotlib.pyplot as plt

results_dir = 'results'

def graph(all_models, all_metrics, xlabel, ylabel, legend, save_dir=None, name=None):
    for x,y in zip(all_models, all_metrics):
        plt.bar(x, y, width=1.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)

    if save_dir and name:
        plt.savefig(f'{results_dir}/{save_dir}/{name}.png')
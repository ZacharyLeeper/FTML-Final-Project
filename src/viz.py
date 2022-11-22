import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

results_dir = './results/'

# unused
def graph(all_models, all_metrics, xlabel, ylabel, legend, save_dir=None, name=None):
    for x,y in zip(all_models, all_metrics):
        plt.bar(x, y, width=1.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)

    if save_dir and name:
        plt.savefig(f'{results_dir}/{save_dir}/{name}.png')

def model_results(models, thresholds, data, labels):
    print('--------------------------------------------')
    results = []
    for model, threshold in zip (models, thresholds):
        m_lo, m_hi, f_lo, f_hi = (threshold[0], threshold[1], threshold[2], threshold[3])
        m_idx = data['Gender'] == 1
        f_idx = data['Gender'] == 0

        pred = model.predict(data)

        m_pred = pred[m_idx]
        f_pred = pred[f_idx]

        m_label = labels[m_idx]
        f_label = labels[f_idx]

        m_tp_label = m_label == 1
        m_tn_label = m_label == 0

        f_tp_label = f_label == 1
        f_tn_label = f_label == 0

        m_random_thresholds = np.random.sample(size=m_pred.shape[0]) * (m_hi-m_lo) + m_lo
        m_tpos = m_pred[m_tp_label] >= m_random_thresholds[m_tp_label]
        m_fpos = m_pred[m_tn_label] >= m_random_thresholds[m_tn_label]
        m_tneg = m_pred[m_tn_label] < m_random_thresholds[m_tn_label]
        m_fneg = m_pred[m_tp_label] < m_random_thresholds[m_tp_label]

        f_random_thresholds = np.random.sample(size=f_pred.shape[0]) * (f_hi-f_lo) + f_lo
        f_tpos = f_pred[f_tp_label] >= f_random_thresholds[f_tp_label]
        f_fpos = f_pred[f_tn_label] >= f_random_thresholds[f_tn_label]
        f_tneg = f_pred[f_tn_label] < f_random_thresholds[f_tn_label]
        f_fneg = f_pred[f_tp_label] < f_random_thresholds[f_tp_label]

        m_tpr = np.sum((m_tpos)/m_label[m_tp_label].shape[0])
        m_fpr = np.sum((m_fpos)/m_label[m_tp_label].shape[0])
        m_tnr = np.sum((m_tneg)/m_label[m_tn_label].shape[0])
        m_fnr = np.sum((m_fneg)/m_label[m_tn_label].shape[0])

        f_tpr = np.sum((f_tpos)/f_label[f_tp_label].shape[0])
        f_fpr = np.sum((f_fpos)/f_label[f_tp_label].shape[0])
        f_tnr = np.sum((f_tneg)/f_label[f_tn_label].shape[0])
        f_fnr = np.sum((f_tneg)/f_label[f_tn_label].shape[0])

        m_dp = (np.sum(m_tpos) + np.sum(m_fpos))/m_label.shape[0]
        f_dp = (np.sum(f_tpos) + np.sum(f_fpos))/f_label.shape[0]

        accuracy = np.sum(m_tpos)
        accuracy += np.sum(m_tneg)
        accuracy += np.sum(f_tpos)
        accuracy += np.sum(f_tneg)
        accuracy /= labels.shape[0]
        
        print(f'Model Results for {model}:')
        print(f'Total Accuracy: {accuracy}')
        print(f'\tMale predictions:\n\t\tTP rate: {m_tpr}\n\t\tFP rate: {m_fpr}\n\t\tTN rate: {m_tnr}\n\t\tFN rate: {m_fnr}')
        print(f'\tFemale predictions:\n\t\tTP rate: {f_tpr}\n\t\tFP rate: {f_fpr}\n\t\tTN rate: {f_tnr}\n\t\tFN rate: {f_fnr}')
        print(f'\tDP: Male {m_dp} | Female {f_dp}')

        dp = (m_dp, f_dp)
        tpr = (m_tpr, f_tpr)
        fpr = (m_fpr, f_fpr)
        tnr = (m_tnr, f_tnr)
        fnr = (m_fnr, f_fnr)
        profit = 2000*np.sum(m_tpos) - 10000*np.sum(m_fpos) + 2000*np.sum(f_tpos) - 10000*np.sum(f_fpos)
        print(f'\tProfit:${profit}')

        results.append({'dp':dp, 'tpr':tpr, 'fpr':fpr, 'tnr':tnr, 'fnr':fnr, 'profit':profit, 'pred':pred})
    return results

def graph_dp(ax, results):
    dp = results['dp']
    ax.bar([0], dp[0], color='steelblue', edgecolor='black', tick_label='')
    ax.bar([1], dp[1], color='palevioletred', edgecolor='black', tick_label='')

    ax.legend(['Male', 'Female'])
    ax.set_ylabel('Proportion of Accepted Applications')
    ax.set_title('Loan Applications Accepted')
    ax.set_xticks([])
    labels = [i*10 for i in range(1,11)]
    ax.set_yticks([i/100 for i in labels], labels=[f'{i}%' for i in labels])

def graph_eo(ax, results):
    tpr = results['tpr']
    fpr = results['fpr']
    width = 0.45

    ax.bar([0, 0.5], [tpr[0], fpr[0]], width=width, color='steelblue', edgecolor='black', tick_label='')
    ax.bar([1, 1.5], [tpr[1], fpr[1]], width=width, color='palevioletred', edgecolor='black', tick_label=' ')

    ax.legend(['Male', 'Female'])
    ax.set_ylabel('Proportion of Accepted Applications')
    ax.set_title('Loans Received by Sex')
    ax.set_xticks([0, 1, 0.5, 1.5], labels=[*(['Will Pay Back']*2),*(['Will Default']*2)])
    labels = [i*10 for i in range(1,11)]
    ax.set_yticks([i/100 for i in labels], labels=[f'{i}%' for i in labels])

def graph_pred(fig, results, thresholds, filenames):
    for model_results, t, name in zip(results, thresholds, filenames):
        pred = model_results['pred']

        plt.figure()

        pred_min = np.min(pred)
        pred = pred - pred_min
        pred_max = np.max(pred)
        plt.scatter(np.linspace(0, 1, pred.shape[0]), pred/pred_max)
        x = np.linspace(0, 1)
        t1 = (np.average(t[0:2])-pred_min) / pred_max
        t2 = (np.average(t[2:])-pred_min) / pred_max
        plt.plot(x, np.ones(x.shape[0])*t1, color='red')
        plt.plot(x, np.ones(x.shape[0])*t2, color='red')

        plt.savefig(f'{results_dir}{name}_preds.png')

def make_sliders(fig, thresholds):
    m_avg = np.average(thresholds[:2])
    f_avg = np.average(thresholds[2:])

    m_ax = fig.add_axes([0.15, 0.04, 0.75, 0.01])
    m_threshold_slider = Slider(ax=m_ax, label='Threshold (Male)', valmin=0, valmax=1.0, valinit=m_avg, color = 'steelblue')
    f_ax = fig.add_axes([0.15, 0.02, 0.75, 0.01])
    f_threshold_slider = Slider(ax=f_ax, label='Threshold (Female)', valmin=0, valmax=1.0, valinit=f_avg, color = "palevioletred")
    return m_threshold_slider, f_threshold_slider

def update_slider(model, data, labels, ax, m_slider, f_slider):
    new_thresholds = (m_slider.val, m_slider.val, f_slider.val, f_slider.val)
    results = model_results([model], [new_thresholds], data, labels)
    # graph_dp(ax[0], results[0])
    # graph_eo(ax[1], results[0])

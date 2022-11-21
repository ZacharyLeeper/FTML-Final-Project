import matplotlib.pyplot as plt
import numpy as np

results_dir = '../results/'

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

def graph_dp(results, filenames):
    for model_results, name in zip(results, filenames):
        plt.figure()

        dp = model_results['dp']
        plt.bar([0], dp[0])
        plt.bar([1], dp[1])

        plt.legend(['Male', 'Female'])
        plt.ylabel('Proportion of Accepted Applications')
        plt.title('Loans Received by Sex')
        plt.xticks([])
        plt.yticks([0.25, 0.5, 0.75, 1.0], labels=['25%', '50%', '75%', '100%'])

        plt.savefig(f'{results_dir}{name}_dp.png')

def graph_eo(results, filenames):
    for model_results, name in zip(results, filenames):
        tpr = model_results['tpr']
        fpr = model_results['fpr']
        width = 0.45

        plt.figure()

        plt.bar([0, 0.5], [tpr[0], fpr[0]], width=width)
        plt.bar([1, 1.5], [tpr[1], fpr[1]], width=width)

        plt.legend(['Male', 'Female'])
        plt.ylabel('Proportion of Accepted Applications')
        plt.title('Loans Received by Sex')
        plt.xticks([0, 1, 0.5, 1.5], labels=[*(['Will Pay Back']*2),*(['Will Default']*2)])
        plt.yticks([0.25, 0.5, 0.75, 1.0], labels=['25%', '50%', '75%', '100%'])

        plt.savefig(f'{results_dir}{name}_eo.png')

def graph_pred(results, filenames):
    for model_results, name in zip(results, filenames):
        pred = model_results['pred']

        plt.figure()

        plt.scatter(np.linspace(0, 1, pred.shape[0]), pred)

        # plt.legend(['Male', 'Female'])
        # plt.ylabel('Proportion of Accepted Applications')
        # plt.title('Loans Received by Sex')
        # plt.xticks([0, 1, 0.5, 1.5], labels=[*(['Will Pay Back']*2),*(['Will Default']*2)])
        # plt.yticks([0.25, 0.5, 0.75, 1.0], labels=['25%', '50%', '75%', '100%'])

        plt.savefig(f'{results_dir}{name}_preds.png')

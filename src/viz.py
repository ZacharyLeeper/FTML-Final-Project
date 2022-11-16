import matplotlib.pyplot as plt
import numpy as np

results_dir = '../results'

def graph(all_models, all_metrics, xlabel, ylabel, legend, save_dir=None, name=None):
    for x,y in zip(all_models, all_metrics):
        plt.bar(x, y, width=1.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)

    if save_dir and name:
        plt.savefig(f'{results_dir}/{save_dir}/{name}.png')

def print_model_results(models, thresholds, data, labels):
    print(thresholds)
    print('--------------------------------------------')
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
        m_pos = m_pred[m_tp_label] >= m_random_thresholds[m_tp_label]
        m_neg = m_pred[m_tn_label] < m_random_thresholds[m_tn_label]

        f_random_thresholds = np.random.sample(size=f_pred.shape[0]) * (f_hi-f_lo) + f_lo
        f_pos = f_pred[f_tp_label] >= f_random_thresholds[f_tp_label]
        f_neg = f_pred[f_tn_label] < f_random_thresholds[f_tn_label]

        m_tpr = np.sum((m_pos == m_label[m_tp_label])/m_label[m_tp_label].shape[0])
        m_fpr = np.sum((m_pos != m_label[m_tp_label])/m_label[m_tp_label].shape[0])
        m_tnr = np.sum((m_neg == m_label[m_tn_label])/m_label[m_tn_label].shape[0])
        m_fnr = np.sum((m_neg != m_label[m_tn_label])/m_label[m_tn_label].shape[0])

        f_tpr = np.sum((f_pos == f_label[f_tp_label])/f_label[f_tp_label].shape[0])
        f_fpr = np.sum((f_pos != f_label[f_tp_label])/f_label[f_tp_label].shape[0])
        f_tnr = np.sum((f_neg == f_label[f_tn_label])/f_label[f_tn_label].shape[0])
        f_fnr = np.sum((f_neg != f_label[f_tn_label])/f_label[f_tn_label].shape[0])

        m_dp = np.sum(m_pos)/m_label.shape[0]
        f_dp = np.sum(f_pos)/f_label.shape[0]
        
        print('Model Results:')
        print(f'\tMale predictions:\n\t\tTP rate: {m_tpr}\n\t\tFP rate: {m_fpr}\n\t\tTN rate: {m_tnr}\n\t\tFN rate: {m_fnr}')
        print(f'\tFemale predictions:\n\t\tTP rate: {f_tpr}\n\t\tFP rate: {f_fpr}\n\t\tTN rate: {f_tnr}\n\t\tFN rate: {f_fnr}')
        print(f'\tDP: Male {m_dp} | Female {f_dp}')
from fairlearn.reductions import GridSearch, DemographicParity, EqualizedOdds
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import cross_validate
from models import all_models
import random

import numpy as np

def train_models(data, labels):
    trained_models = []
    thresholds = []
    for model, _ in all_models():
        if not isinstance(model, MLPRegressor):
            model = GridSearch(model, EqualizedOdds())
            model.fit(data, labels, sensitive_features=data['Gender'])
        else:
            model.fit(data, labels)

        pred = model.predict(data)

        m_idx = data['Gender'] == 1
        f_idx = data['Gender'] == 0
        m_pred = pred[m_idx]
        f_pred = pred[f_idx]

        m_label = labels[m_idx]
        f_label = labels[f_idx]

        m_tp_label = m_label == 1
        m_tn_label = m_label == 0

        f_tp_label = f_label == 1
        f_tn_label = f_label == 0

        m_pos = m_pred[m_tp_label]
        m_neg = m_pred[m_tn_label]

        f_pos = f_pred[f_tp_label]
        f_neg = f_pred[f_tn_label]

        # print(pred)

        for constraint in ['none', 'dp', 'eo']:
            if constraint == 'none':
                if not isinstance(model, MLPRegressor):
                    continue
                best_accuracy = 0
                best_threshold = -1
                for t in np.linspace(np.min(pred), np.max(pred)):
                    acc = accuracy_score(labels, pred >= t)
                    if acc > best_accuracy:
                        best_accuracy = acc
                        best_threshold = t
                    
                trained_models.append(model)
                thresholds.append((best_threshold,best_threshold,best_threshold,best_threshold))

            elif constraint == 'dp':
                if not isinstance(model, MLPRegressor):
                    continue
                violation_tolerance = 0.01
                best_accuracy = 0
                best_m_t = -1
                best_f_t = -1
                for t1 in np.linspace(np.min(m_pred), np.max(m_pred)):
                    for t2 in np.linspace(np.min(f_pred), np.max(f_pred)):
                        acc = np.sum(m_pos >= t1) + np.sum(m_neg < t1) + np.sum(f_pos >= t2) + np.sum(f_neg < t2)
                        acc /= labels.shape[0]
                        m_dp = np.sum(m_pred >= t1)/m_label.shape[0]
                        f_dp = np.sum(f_pred >= t2)/f_label.shape[0]
                        violation = np.abs(m_dp - f_dp)
                        if violation < violation_tolerance and acc > best_accuracy:
                            print(violation)
                            best_accuracy = acc
                            best_m_t = t1
                            best_f_t = t2

                trained_models.append(model)
                thresholds.append((best_m_t, best_m_t, best_f_t, best_f_t))

            elif constraint == 'eo':
                violation_tolerance = 0.01
                best_accuracy = 0
                best_m_t1 = 0.51
                best_m_t2 = -1
                best_f_t1 = 0.51
                best_f_t2 = -1
                for t1 in np.linspace(np.min(m_pred), np.max(m_pred)):
                    for t2 in np.linspace(np.min(f_pred), np.max(f_pred)):
                        acc = np.sum(m_pos >= t1) + np.sum(m_neg < t1) + np.sum(f_pos >= t2) + np.sum(f_neg < t2)
                        acc /= labels.shape[0]
                        m_tp = np.sum(m_pos >= t1)
                        m_fp = np.sum(m_neg >= t1)
                        m_tn = np.sum(m_neg < t1)
                        m_fn = np.sum(m_pos < t1)
                        m_tpr = m_tp/(m_tp + m_fn)
                        m_fpr = m_fp/(m_fp + m_tn)

                        f_tp = np.sum(f_pos >= t2)
                        f_fp = np.sum(f_neg >= t2)
                        f_tn = np.sum(f_neg < t2)
                        f_fn = np.sum(f_pos < t2)
                        f_tpr = f_tp/(f_tp + f_fn)
                        f_fpr = f_fp/(f_fp + f_tn)
                        if np.abs(m_tpr - f_tpr) < violation_tolerance and acc > best_accuracy:
                            print('tpr', np.abs(m_tpr - f_tpr))
                            best_accuracy = acc
                            best_m_t1 = t1
                            best_f_t1 = t2
                        if np.abs(m_fpr - f_fpr) < violation_tolerance and acc >= best_accuracy:
                            print('fpr', np.abs(m_fpr - f_fpr))
                            best_accuracy = acc
                            best_m_t2 = t1
                            best_f_t2 = t2

                if best_m_t1 > best_m_t2:
                    temp = best_m_t2
                    best_m_t2 = best_m_t1
                    best_m_t1 = temp
                if best_f_t1 > best_f_t2:
                    temp = best_f_t2
                    best_f_t2 = best_f_t1
                    best_f_t1 = temp
                trained_models.append(model)
                thresholds.append((best_m_t1, best_m_t2, best_f_t1, best_f_t2))

    return trained_models, thresholds

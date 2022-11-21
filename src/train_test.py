from fairlearn.reductions import GridSearch, DemographicParity, EqualizedOdds
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import cross_validate
from models import all_models
import random

import numpy as np

def train_models(data, labels):
    trained_models = []
    thresholds = []
    for model, constraint in all_models():
        # model.fit(data, labels)
        # scores = cross_validate(model, data, labels, return_estimator=True)
        # models = scores['estimator']
        # print(scores)
        # model = VotingRegressor(scores['estimator'])
        # model.fit(data, labels)
        if constraint:
            model = GridSearch(model, constraint)
            model.fit(data, labels, sensitive_features=data['Gender'])
            trained_models.append(model)
            pred = model.predict(data)
            m_idx = data['Gender'] == 1
            f_idx = data['Gender'] == 0
            m_pred = pred[m_idx]
            f_pred = pred[f_idx]

            sub_m_pred = np.sort(m_pred[labels[m_idx] == 1])
            sub_f_pred = np.sort(f_pred[labels[f_idx] == 1])

            m_threshold1 = sub_m_pred[int(sub_m_pred.shape[0] * 0.18)]
            f_threshold1 = sub_f_pred[int(sub_f_pred.shape[0] * 0.18)]
            if isinstance(constraint, DemographicParity):
                thresholds.append([m_threshold1, m_threshold1, f_threshold1, f_threshold1])
            if isinstance(constraint, EqualizedOdds):
                sub_m_pred = np.sort(m_pred[labels[m_idx] == 0])
                sub_f_pred = np.sort(f_pred[labels[f_idx] == 0])
                m_threshold2 = sub_m_pred[int(sub_m_pred.shape[0] * 0.82)]
                f_threshold2 = sub_f_pred[int(sub_f_pred.shape[0] * 0.82)]
                m_min, m_max = min(m_threshold1, m_threshold2), max(m_threshold1, m_threshold2)
                f_min, f_max = min(f_threshold1, f_threshold2), min(f_threshold1, f_threshold2)
                thresholds.append([m_min, m_max, f_min, f_max])
        else:
            # print("Number of Non Defaulters: ", np.sum(labels == 1), "Number of Defaulters: ", np.sum(labels == 0))
            model.fit(data, labels)
            trained_models.append(model)
            pred = model.predict(data)
            sub_pred = pred[labels == 1]
            sorted_pred = np.sort(sub_pred)
            threshold = sorted_pred[int(sub_pred.shape[0] * 0.18)]
            # print("Number of Non Defaulters accepted: ", np.sum(pred[labels == 1] >= threshold), "Number of Defaulters accepted: ", np.sum(pred[labels == 0] >= threshold))
            # print("Number of Non Defaulters denied: ", np.sum(pred[labels == 1] < threshold), "Number of Defaulters denied: ", np.sum(pred[labels == 0] < threshold))
            thresholds.append([threshold] * 4)
    return trained_models, thresholds

def eval_models(models, thresholds, data, labels):
    accuracy = []
    m_accuracy = []
    f_accuracy = []
    for model, (m_l, m_h, f_l, f_h) in zip(models, thresholds):
        pred = model.predict(data)
        # pred[data['Gender'] == 1] = pred[data['Gender'] == 1] >= m_thresh
        # pred[data['Gender'] == 0] = pred[data['Gender'] == 0] >= f_thresh
        # print(pred)
        if m_l == m_h:
            pred[data['Gender'] == 1] = pred[data['Gender'] == 1] >= m_l
            pred[data['Gender'] == 0] = pred[data['Gender'] == 0] >= f_l
        else:
            for i in range(pred.shape[0]):
                if i in data['Gender'] == 1:
                    if pred[i] > m_h:
                        pred[i] = 1
                    elif pred[i] < m_l:
                        pred[i] = 0
                    else:
                        pred[i] = random.randint(0, 1)
                else:
                    if pred[i] > f_h:
                        pred[i] = 1
                    elif pred[i] < f_l:
                        pred[i] = 0
                    else:
                        pred[i] = random.randint(0, 1)

        accuracy.append(f1_score(labels, pred))
        m_accuracy.append(f1_score(labels[data['Gender'] == 1], pred[data['Gender'] == 1]))
        f_accuracy.append(f1_score(labels[data['Gender'] == 0], pred[data['Gender'] == 0]))
    print(accuracy)
    # print(m_accuracy)
    # print(f_accuracy)
    raise Exception("HERE")
    return accuracy
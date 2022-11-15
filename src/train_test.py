from fairlearn.reductions import GridSearch, DemographicParity, EqualizedOdds
from sklearn.metrics import f1_score, accuracy_score
from models import all_models

import numpy as np

def train_models(data, labels):
    trained_models = []
    thresholds = []
    for model, constraint in all_models():
        if constraint:
            model = GridSearch(model, constraint)
            # TODO remove sensitive feature from data? - keep these values, otherwise throws an error
            model.fit(data, labels, sensitive_features=data['Gender'])
            trained_models.append(model)
            pred = model.predict(data)
            # TODO better thresholding?
            if isinstance(constraint, DemographicParity):
                m_pred = pred[data['Gender'] == 1]
                f_pred = pred[data['Gender'] == 0]
                sub_m_pred = np.sort(m_pred[labels == 1])
                sub_f_pred = np.sort(f_pred[labels == 1])
                m_threshold = sorted_pred[int(sub_m_pred.shape[0] * 0.18)]
                f_threshold = sorted_pred[int(sub_f_pred.shape[0] * 0.18)]
                thresholds.append([m_threshold, m_threshold, f_threshold, f_threshold])
            elif isinstance(constraint, EqualizedOdds):
                m_pred = pred[data['Gender'] == 1]
                f_pred = pred[data['Gender'] == 0]
                sub_m_pred = np.sort(m_pred[labels == 1])
                sub_f_pred = np.sort(f_pred[labels == 1])
                m_threshold1 = sorted_pred[int(sub_m_pred.shape[0] * 0.18)]
                f_threshold1 = sorted_pred[int(sub_f_pred.shape[0] * 0.18)]

                sub_m_pred = np.sort(m_pred[labels == 0])
                sub_f_pred = np.sort(f_pred[labels == 0])
                m_threshold2 = sorted_pred[int(sub_m_pred.shape[0] * 0.82)]
                f_threshold2 = sorted_pred[int(sub_f_pred.shape[0] * 0.82)]
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
    for model, (m_thresh, f_thresh) in zip(models, thresholds):
        pred = model.predict(data)
        pred[data['Gender'] == 1] = pred[data['Gender'] == 1] >= m_thresh
        pred[data['Gender'] == 0] = pred[data['Gender'] == 0] >= f_thresh
        print(pred)
        accuracy.append(f1_score(labels, pred))
        m_accuracy.append(f1_score(labels[data['Gender'] == 1], pred[data['Gender'] == 1]))
        f_accuracy.append(f1_score(labels[data['Gender'] == 0], pred[data['Gender'] == 0]))
    print(accuracy)
    print(m_accuracy)
    print(f_accuracy)
    raise Exception("HERE")
    return accuracy
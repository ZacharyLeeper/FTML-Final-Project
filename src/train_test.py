from fairlearn.reductions import GridSearch
from sklearn.metrics import f1_score, accuracy_score
from models import all_models

import numpy as np

def train_models(data, labels):
    trained_models = []
    thresholds = []
    for model, constraint in all_models():
        if constraint:
            model = GridSearch(model, constraint)
            # TODO remove sensitive feature from data?
            model.fit(data, labels, sensitive_features=data['Gender'])
            trained_models.append(model)
            pred = model.predict(data)
            # TODO better thresholding?
            m_threshold = np.mean(pred[data['Gender'] == 1])
            f_threshold = np.mean(pred[data['Gender'] == 0])
            thresholds.append((m_threshold, f_threshold))
        else:
            model.fit(data, labels)
            trained_models.append(model)
            pred = model.predict(data)
            threshold = np.mean(pred)
            thresholds.append((threshold, threshold))
    
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
        m_accuracy.append(accuracy_score(labels[data['Gender'] == 1], pred[data['Gender'] == 1]))
        f_accuracy.append(accuracy_score(labels[data['Gender'] == 0], pred[data['Gender'] == 0]))
    print(accuracy)
    print(m_accuracy)
    print(f_accuracy)
    return accuracy
from fairlearn.reductions import GridSearch
from sklearn.metrics import log_loss, accuracy_score, f1_score
from models import all_models, MODELS
from sklearn.linear_model import LogisticRegressionCV

def train_models(data, labels):
    trained_models = []
    for model, constraint in all_models():
        if constraint:
            model = GridSearch(model, constraint)
            model.fit(data, labels, sensitive_features=data['Gender'])
            trained_models.append(model)
        else:
            model.fit(data, labels)
            trained_models.append(model)
    return trained_models

def eval_models(models, data, labels):
    accuracy = []
    threshold = 0.82
    for model in models:
        pred = model.predict(data)
        print(type(model))
        pred = pred >= 0.60
        print(pred)
        accuracy.append(f1_score(labels, pred))
    print(accuracy)
    raise Exception("STROP")
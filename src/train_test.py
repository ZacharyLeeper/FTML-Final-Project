from fairlearn.reductions import GridSearch
from sklearn.metrics import log_loss
from models import all_models, MODELS

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
    nll = []
    for model in models:
        pred = model.predict(data)
        nll.append(log_loss(labels, pred))
    return nll
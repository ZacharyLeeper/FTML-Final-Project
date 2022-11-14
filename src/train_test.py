from sklearn.metrics import accuracy_score
from .models import all_models

def train_models(data, labels):
    trained_models = []
    for model in all_models():
        trained_models.append(model.fit(data, labels))
    return trained_models

def eval_models(models, data, labels):
    accuracy = []
    for model in models:
        pred = model.predict(data)
        accuracy.append(accuracy_score(labels, pred))
    return accuracy
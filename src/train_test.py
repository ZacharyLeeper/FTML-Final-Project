from fairlearn.reductions import GridSearch
from sklearn.metrics import accuracy_score
from models import all_models, MODELS

def train_models(data, labels):
    trained_models = []
    for (model, constraint), model_name in zip(all_models(), MODELS):
        if constraint:
            model = GridSearch(model, constraint)
            trained_models.append(model.fit(data, labels, sensitive_features=data['Gender']))
        else:
            trained_models.append(model.fit(data, labels))
    return trained_models

def eval_models(models, data, labels):
    accuracy = []
    for model in models:
        pred = model.predict(data)
        accuracy.append(accuracy_score(labels, pred))
    return accuracy
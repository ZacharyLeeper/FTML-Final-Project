from fairlearn.reductions import GridSearch
from sklearn.metrics import f1_score
from models import all_models

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
    threshold = 0.5
    for model in models:
        pred = model.predict(data)
        print(pred)
        pred = pred >= threshold
        accuracy.append(f1_score(labels, pred))
    print(accuracy)
    return accuracy
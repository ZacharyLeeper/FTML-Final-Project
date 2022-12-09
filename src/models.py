from fairlearn.reductions import DemographicParity, EqualizedOdds
from joblib import dump, load
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

__all__ = ['logistic_reg_eo', 'nn_reg', 'nn_reg_dp', 'nn_reg_eo', 'all_models', 'MODELS']

MODELS = ['lin_reg_eo', 'nn_reg', 'nn_reg_dp', 'nn_reg_eo']
LR_PARAMS = {'solver':'liblinear'}
NN_PARAMS = {'hidden_layer_sizes':(128,64,), 'solver':'sgd', 'activation':'logistic', 'learning_rate':'invscaling', 'max_iter':500, 'random_state':0}

# Return models with a fairness constraint

def linear_reg():
    model = LinearRegression()
    return model, None

def nn_reg():
    model = MLPRegressor(**NN_PARAMS)
    return model, None

def nn_reg_dp():
    dp = DemographicParity()
    model = MLPRegressor(**NN_PARAMS)
    return model, dp

def nn_reg_eo():
    eo = EqualizedOdds()
    model = MLPRegressor(**NN_PARAMS)
    return model, eo

# Return a list of all models used
# Note that we only ended up using the NN
def all_models():
    return [linear_reg(), nn_reg()]

# Save a model for presenting its results to stakeholders
def save_models(models, thresholds, filenames):
    for model, t, name in zip(models, thresholds, filenames):
        dump(model, f'{name}.joblib')
        with open(f'{name}_t.txt', 'w+') as f:
            f.writelines(str(t))

# Load a model for presentation
def load_model(model_path):
    model = load(f'{model_path}.joblib')
    threshold = None
    with open (f'{model_path}_t.txt') as f:
        line = f.readline()
        line = line.strip('( ,)').split(', ')
        threshold = (float(line[0]), float(line[1]), float(line[2]), float(line[3]))
    return model, threshold
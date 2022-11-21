from fairlearn.reductions import DemographicParity, EqualizedOdds
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.tree import DecisionTreeRegressor

__all__ = ['logistic_reg_eo', 'nn_reg', 'nn_reg_dp', 'nn_reg_eo', 'all_models', 'MODELS']

MODELS = ['lin_reg_eo', 'nn_reg', 'nn_reg_dp', 'nn_reg_eo']
LR_PARAMS = {'solver':'liblinear'}
# TODO figure out parameters
NN_PARAMS = {'hidden_layer_sizes':(128,64,), 'solver':'sgd', 'activation':'logistic', 'learning_rate':'invscaling', 'max_iter':500, 'random_state':0}
RF_PARAMS = {'random_state':0, 'max_depth':3}

def linear_reg():
    model = LinearRegression()
    return model, None

def random_forest():
    model = DecisionTreeRegressor(**RF_PARAMS)
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
    
def all_models():
    return [linear_reg(), nn_reg()]

def save_models(models, thresholds, filenames):
    for model, t, name in zip(models, thresholds, filenames):
        dump(model, f'{name}.joblib')
        with open(f'{name}_t.txt', 'w+') as f:
            f.writelines(str(t))

# We'll worry about this one when we get there
# def load_models(model_dir):
#     models = []
#     thresholds = []
    
#     model = load(filename)
#     return models, thresholds
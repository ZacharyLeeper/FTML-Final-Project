from fairlearn.reductions import DemographicParity, EqualizedOdds
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor

__all__ = ['logistic_reg_eo', 'nn_reg', 'nn_reg_dp', 'nn_reg_eo', 'all_models', 'MODELS']

MODELS = ['Logistic reg. with EO', 'NN reg.', 'NN reg. with DP', 'NN reg. with EO']
LR_PARAMS = {'solver':'liblinear', 'cv':KFold(), 'fit_intercept':True}
# TODO figure out parameters
NN_PARAMS = {'hidden_layer_sizes':(512,256,128,), 'solver':'adam', 'activation':'relu', 'learning_rate':'invscaling', 'max_iter':500, 'random_state':0}

def logistic_reg_eo():
    eo = EqualizedOdds()
    model = LogisticRegressionCV(**LR_PARAMS)
    return model, eo

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
    return [logistic_reg_eo(), nn_reg(), nn_reg_dp(), nn_reg_eo()]
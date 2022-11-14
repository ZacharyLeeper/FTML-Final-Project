from fairlearn.reductions import DemographicParity, EqualizedOdds, GridSearch
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor

__all__ = ['logistic_reg_eo', 'nn_reg', 'nn_reg_dp', 'nn_reg_eo', 'all_models', 'MODELS']

MODELS = ['Logistic reg. with EO', 'NN reg.', 'NN reg. with DP', 'NN reg. with EO']

def logistic_reg_eo():
    cv = KFold()
    eo = EqualizedOdds()

    model = LogisticRegressionCV(solver='liblinear', Cs=1.0, cv=cv, fit_intercept=True)
    algorithm = GridSearch(model, constraints=eo)
    return algorithm

def nn_reg():
    model = MLPRegressor(solver='SGD', learning_rate='invscaling')
    algorithm = GridSearch(model)
    return algorithm

def nn_reg_dp():
    dp = DemographicParity()

    model = MLPRegressor(solver='SGD', learning_rate='invscaling')
    algorithm = GridSearch(model, constraints=dp)
    return algorithm

def nn_reg_eo():
    eo = EqualizedOdds()

    model = MLPRegressor(solver='SGD', learning_rate='invscaling')
    algorithm = GridSearch(model, constraints=eo)
    return algorithm
    
def all_models():
    return [logistic_reg_eo(), nn_reg(), nn_reg_dp(), nn_reg_eo()]
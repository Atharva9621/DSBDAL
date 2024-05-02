import numpy as np

#Regression Metrics

def mse(y, yhat):
    m = y.shape[0]
    return np.sum((yhat-y)**2, 0)/m

def rmse(y, yhat):
    m = y.shape[0]
    return np.sqrt(mse(y, yhat))

def mae(y, yhat):
    m = y.shape[0]
    return np.sum(y-yhat, axis=0)/m
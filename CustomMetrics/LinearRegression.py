import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, epochs=2000, verbosity = 1)->None:
        self.weights = None
        self.bias = None
        self.lr = lr
        self.epochs = epochs 
        self.__init_logger()

    def _init__weights(self, n):
        self.w = np.random.rand(n, 1)
        self.b = np.random.rand(1)

    def predict(self, X):
        yhat = np.dot(X, self.w)+self.b
        return yhat
    
    def _get_grads(self, X, y, yhat, m):
        dJ_dw = 1/m*np.dot(X.T, yhat-y)
        dJ_db = 1/m*np.sum(yhat-y)
        return dJ_dw, dJ_db
    
    def _get_loss(self, y, yhat, m):
        return 1/m*np.sqrt(np.sum((yhat-y)**2, axis=0))
    
    def fit(self, X, y, epochs=None):
        m, n = X.shape
        if epochs is not None:
            self.epochs = epochs
        self._init__weights(n)
        for epoch in range (self.epochs):
            yhat = self.predict(X)
            dJ_dw, dJ_db = self._get_grads(X, y, yhat, m)
            self.w -= self.lr*dJ_dw
            self.b -= self.lr*dJ_db

            self.log(epoch, self._get_loss(y, yhat, m))

    def __init_logger(self):
        pass

    def log(self, epoch, loss):
        if epoch%10 == 0:
            print(f'Epoch {epoch} : {loss}')
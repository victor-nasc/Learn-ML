import numpy as np

# Victor Nascimento Ribeiro
# 01/2024


class naive_bayes:
    
    def __init__(self):
        self._K = None
        self._mean = None
        self._var = None
        self._prior = None
    
    
    # "Trainig" the model
    def fit(self, X, y):
        N = np.shape(X)[0]
        self._K = np.unique(y)
        
        self._mean = np.array([X[y == k].mean(axis=0) for k in self._K])
        self._var = np.array([X[y == k].var(axis=0) for k in self._K])
        self._prior = np.array([np.sum(y == k) / float(N) for k in self._K])

    
    # Make predictions
    def predict(self, X):
        y_pred = np.zeros((len(X), len(self._K)))

        for i, x in enumerate(X):
            for j, k in enumerate(self._K):
                prior = np.log(self._prior[k])
                posterior = np.sum(np.log(self._gaussian(k, x)))
                y_pred[i, j] = posterior + prior

        y_pred = self._K[np.argmax(y_pred, axis=1)]
        return y_pred


    # Evaluate the model
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy
    

    # Gaussian Probability Density Function
    def _gaussian(self, k, x):
        mean = self._mean[k]
        var = self._var[k]
        
        gauss = np.exp(-((x - mean) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)
        return gauss
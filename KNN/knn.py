import numpy as np

# Victor Nascimento Ribeiro
# 01/2024 


class KNN:
    
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
        
        
    # "Trainig" the model
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    
    # Make predictions
    def predict(self, X):
        y_pred = []
        for x in X:
            dist = self._euclidean_distance(self.X_train, x)
            k_indx = np.argsort(dist)[:self.k]
            k_nearest = self.y_train[k_indx]
            most_common = np.bincount(k_nearest).argmax()
            y_pred.append(most_common)
                        
        return np.array(y_pred)
    
    
    # Evaluate the model
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy
    
    
    # Euclidian Distance
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2, axis=1))
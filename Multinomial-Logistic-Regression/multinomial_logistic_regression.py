import numpy as np

# Victor Nascimento Ribeiro
# 01/2024 


class multinomial_logistic_regression:
    
    def __init__(self):
        self.w = None
        self.loss_history = []
        
        
    # Trainig the model
    def fit(self, X, y, lr=0.001, epochs=500):
        # Add a bias term to the input features (X_0 = 1)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Get the number of samples (N) and features (d)
        N, d = np.shape(X)
        
        # Get the number of classes
        K = len(np.unique(y))
        
        # Initialize weights
        self.w = np.random.rand(d, K)
        
        # One-hot encoding
        y_one_hot = self._one_hot_encode(y, K)
        
        # Batch Gradient Descent
        for _ in range(epochs):
            # Predicted values using current weights
            y_pred = self._softmax(np.dot(X, self.w))
            
            # Compute gradient of the Cross-Entropy Loss
            grad_w = (1 / N) * np.dot(X.T, y_pred - y_one_hot)
            
            # Update weights 
            self.w -= lr * grad_w
            
            # store loss history
            loss = self._cross_entropy_loss(y_one_hot, y_pred)
            self.loss_history.append(loss)
            
    
    # Make predictions
    def predict(self, X):
        # Add a bias term to the input features (X_0 = 1)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Predicted values using learned weights
        y_pred = self._softmax(np.dot(X, self.w))
        
        return y_pred
    
    
    # Evaluate the model
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        y_pred = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_pred == y)
        return accuracy
    
    
    # Cross-Entropy Loss
    def _cross_entropy_loss(self, y_one_hot, y_pred):
        return -np.sum(y_one_hot * np.log(y_pred)) / len(y_one_hot)
        
    
    # Softmax activation function
    def _softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    
    
    # One hot encode
    def _one_hot_encode(self, y, K):
        return np.eye(K)[y]
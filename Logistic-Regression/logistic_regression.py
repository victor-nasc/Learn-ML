import numpy as np

# Victor Nascimento Ribeiro
# 01/2024

# Formulation {0,1} !!
class logistic_regression:

    def __init__(self):
        self.w = None
        self.loss_history = []


    # Trainig the model
    def fit(self, X, y, lr=0.001, epochs=500):
        # Add a bias term to the input features (X_0 = 1)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Get the number of samples (N) and features (d)
        N, d = np.shape(X)
        
        # Initialize weights
        self.w = np.random.rand(d)
        
        # Batch Gradient Descent
        for _ in range(epochs):
            # Predicted values using current weights
            y_pred = self._sigmoid(np.dot(X, self.w))

            # Compute gradient of the Cross-Entropy Loss
            grad_w = (1 / N) * np.dot(X.T, y_pred - y)
            
            # Update weights 
            self.w -= lr * grad_w
            
            # store loss history
            loss = self._cross_entropy_loss(y, y_pred)
            self.loss_history.append(loss)



    # Make predictions
    def predict(self, X):
        # Add a bias term to the input features (X_0 = 1)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Predicted values using learned weights
        y_pred = self._sigmoid(np.dot(X, self.w)) > 0.5
        
        return y_pred

    
    # Evaluate the model
    def evaluate(self, X, y):
        # Add a bias term to the input features (X_0 = 1)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Predicted values using learned weights
        y_pred = self._sigmoid(np.dot(X, self.w))
        
        # Compute the accuracy
        y_pred = (y_pred > 0.5).astype(int)
        accuracy = np.mean(y_pred == y)
        
        return accuracy
    
    
    
    # Sigmoid Activation Function
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    
    # Binary Cross-Entropy Loss
    def _cross_entropy_loss(self, y, y_pred):
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
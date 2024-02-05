import numpy as np

# Victor Nascimento Ribeiro
# 01/2024

class linear_regression:

    def __init__(self):
        self.w = None
        self.loss_history = []


    # Analytical solution 
    def fit_analytical(self, X, y):
        # Add a bias term to the input features (X_0 = 1)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Calculate the pseudo-inverse of X
        X_pinv = np.linalg.pinv(X)
        
        # Calculate weights
        self.w = np.dot(X_pinv, y)


    # Gradient Descent solution 
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
            y_pred = np.dot(X, self.w)
            
            # Compute gradient of the MSE
            grad_w = (1 / N) * np.dot(X.T, y_pred - y)
            
            # Update weights 
            self.w -= lr * grad_w
            
            # store loss history
            loss = self._mean_squared_error(y, y_pred)
            self.loss_history.append(loss)
            
            
    # Make predictions
    def predict(self, X):
        # Add a bias term to the input features (X_0 = 1)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Predicted values using learned weights
        y_pred = np.dot(X, self.w)
        return y_pred


    # Evaluate the model on a dataset
    def evaluate(self, X, y):
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate MSE
        mse = self._mean_squared_error(y, y_pred)
        
        # Calculate R2
        r2 = 1 - mse / np.var(y)
        
        return r2, mse

    
    # Mean Squared Error function
    def _mean_squared_error(self, y_true, y_pred):
        N = len(y_true)
        mse = np.sum((y_true - y_pred) ** 2) / N
        return mse